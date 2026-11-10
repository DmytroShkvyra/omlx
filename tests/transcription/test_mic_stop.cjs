// Exercise both sides of the actual AudioWorklet port, including queued PCM
// and the worklet's sub-2048-sample tail at the moment Stop is pressed.
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {execFileSync} = require('node:child_process');
const page = require('node:fs').readFileSync(require('node:path').join(__dirname,'../../apps/omlx-web/omlx_web/templates/chat.html'),'utf8');
const a=page.indexOf('    _micWorkletSource()'), b=page.indexOf('    async startMicTranscription()',a),c=page.indexOf('    formatRecordingTime(secs)',b);
const parts={worklet:page.slice(a,b),methods:page.slice(b,c)};
const methods = Function('return ({' + parts.worklet + parts.methods + '})')();
const node = () => ({connect(){}, disconnect(){}});

async function check(inputRate) {
    let Processor;
    let worklet;
    const sent = [];
    const trace = [];
    const track = {stopped: false, stop(){this.stopped = true;}};
    const ctx = {state:'running', sampleRate:inputRate, destination:node(),
        createMediaStreamSource:node, createGain:()=>({...node(),gain:{value:1}}),
        close(){trace.push('close');this.state='closed';return Promise.resolve();},
        audioWorklet:{async addModule(url){
            const source = await (await fetch(url)).text();
            vm.runInNewContext(source, {
                AudioWorkletProcessor:class {constructor(){this.port=worklet.processorPort;}},
                registerProcessor(_name, cls){Processor=cls;}, Float32Array,
            });
        }}};
    global.AudioWorkletNode = function() {
        worklet = this;
        Object.assign(this, node());
        this.port = {onmessage:null,postMessage:data=>queueMicrotask(()=>this.processorPort.onmessage({data}))};
        this.processorPort = {onmessage:null,postMessage:data=>queueMicrotask(()=>this.port.onmessage({data}))};
        this.processor = new Processor();
    };
    global.WebSocket = {OPEN:1};
    const session = {audioContext:ctx, chatId:'test', advancedMic:true, diarize:true,
        committed:'', partial:'', mediaStream:{getTracks:()=>[track]},
        ws:{readyState:1,send(value){
            trace.push(typeof value === 'string' ? 'stop' : 'pcm');sent.push(value);
        }}};
    const stream = {};
    const app = {...methods,micSession:session,micAudioSeconds:0,
        getStreamSession:()=>stream,getChatSession:()=>({messages:[]}),setEngineStatus(){},
        _failMicTranscription(_s,_st,_chat,message){throw Error(message);}};
    await app._startMicCapture(session);
    const quantum = new Float32Array(128).fill(0.125);
    // 2048 samples queued to main thread, another 384 still inside the worklet.
    for(let n=0;n<19;n++)worklet.processor.process([[quantum]]);
    const stopped = app.stopMicTranscription();
    assert(track.stopped, 'Physical microphone stops synchronously');
    assert.equal(ctx.state,'running','Keep audio context alive until buffered PCM arrives');
    await app.stopMicTranscription(); // accidental duplicate click must not finalize
    await stopped;
    assert.equal(ctx.state,'closed');
    const audio = sent.filter(x=>typeof x!=='string');
    const frames = audio.reduce((n,x)=>n+x.byteLength/2,0);
    assert.equal(frames, Math.ceil(2432*16000/inputRate), 'No queued or worklet-buffered sample lost');
    for(const buffer of audio)for(const value of new Int16Array(buffer))assert.equal(value,4095);
    const controls = sent.filter(x=>typeof x==='string').map(JSON.parse);
    assert.equal(controls.length,1);
    assert.equal(controls[0].type,'stop');
    assert.equal(controls[0].audio_bytes,frames*2);
    assert.equal(trace.at(-1),'stop','Stop command must follow the last PCM packet');
    assert(trace.indexOf('pcm') < trace.indexOf('close'),'Do not close before receiving final PCM');
    assert(session.stopping && !session.finalized,'Wait for full model result after draining');
    assert.equal(session.stopTimeout,undefined,'No time limit discarding advanced results');
    const before = sent.length;
    for(let n=0;n<20;n++)worklet.processor.process([[quantum]]);
    await Promise.resolve();
    assert.equal(sent.length,before,'Worklet freezes input after flush');
    console.log('PASS:',inputRate,'Hz: queued PCM + 384-sample tail drained before a single Stop');
}
(async()=>{await check(16000);await check(48000);})().catch(error=>{console.error(error);process.exitCode=1;});
