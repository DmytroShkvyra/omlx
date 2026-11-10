const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const page = require('node:fs').readFileSync(require('node:path').join(__dirname,'../../apps/omlx-web/omlx_web/templates/chat.html'),'utf8');
const a=page.indexOf('    async startMicTranscription()'),b=page.indexOf('    formatRecordingTime(secs)',a);
const methods=Function('return ({'+page.slice(a,b)+'})')();
// Use the actual shipped Alpine runtime. Plain-object mocks missed this bug.
const alpineSource = fs.readFileSync(require('node:path').join(__dirname,'../../apps/omlx-web/omlx_web/static/js/alpine.min.js'), 'utf8');
const sandbox = {window:{Element:class{}}, document:{addEventListener(){},createElement(){return{};}},
 MutationObserver:class{observe(){}disconnect(){}},queueMicrotask(){},console,setTimeout,clearTimeout};
vm.runInNewContext(alpineSource,sandbox);
const Alpine = sandbox.window.Alpine;
const raw = {}, proof = Alpine.reactive({session:raw});
assert.notEqual(proof.session,raw);
console.log('REPRODUCED: Alpine',Alpine.version,'wraps raw session; original guard drops ready/done');
const saved = {setTimeout,clearTimeout,setInterval,clearInterval,now:Date.now};
function node(){return{connect(){},disconnect(){}};};
function fixture({hangingModule=false,diarize=false,translate=false,visual=false,model=null}={}) {
 let time=1000, counter=0, ws, resolveModule;
 const timers=new Map(),intervals=new Map();
 global.setTimeout=(fn,ms)=>{const id=++counter;timers.set(id,{fn,at:time+ms});return id;};
 global.clearTimeout=id=>timers.delete(id);
 global.setInterval=fn=>{const id=++counter;intervals.set(id,fn);return id;};
 global.clearInterval=id=>intervals.delete(id);
 Date.now=()=>time;
 const ctx={ [Symbol.toStringTag]:'AudioContext',state:'running',sampleRate:48000,destination:node(),
   resume(){return Promise.resolve();},close(){this.state='closed';return Promise.resolve();},
   createMediaStreamSource(){return node();},createGain(){return{...node(),gain:{value:1}};},
   audioWorklet:{addModule:()=>hangingModule?new Promise(resolve=>{resolveModule=resolve;}):Promise.resolve()}};
 const track={stopped:false,stop(){this.stopped=true;}};;
 const media={ [Symbol.toStringTag]:'MediaStream',getTracks:()=>[track],getAudioTracks:()=>[{label:'test input'}]};
 global.location={protocol:'http:',host:'127.0.0.1:18081'};
 global.window={AudioContext:function(){return ctx;},t:x=>x,addEventListener(){},removeEventListener(){}};
 Object.defineProperty(global,'navigator',{configurable:true,value:{mediaDevices:{getUserMedia:async()=>media}}});
 global.alert=message=>{throw Error('Unexpected alert: '+message);};
 global.WebSocket=class{
   static OPEN=1;static CONNECTING=0;
   get [Symbol.toStringTag](){return 'WebSocket';}
   constructor(url){ws=this;this.url=url;this.readyState=0;this.sent=[];}
   send(value){if(this.throwOnSend)throw Error('socket send fixture');this.sent.push(value);}
   close(){this.readyState=3;if(this.onclose)this.onclose();}
 };
 global.AudioWorkletNode=function(){const port={onmessage:null,postMessage(){queueMicrotask(()=>port.onmessage({data:{type:'flushed'}}));}};return{...node(),port};};
 const app=Alpine.reactive({...methods,micSession:null,micSeconds:0,micLevel:0,micAudioSeconds:0,diarizeUpload:diarize,translateUpload:translate,targetLanguage:'Ukrainian',
   visualEnabled:visual,visualScreen:visual?{}:null,visualFrames:[],visionModel:'chosen-vlm',
   _startVisualSampling(s){s.visualSampleStarted=true;},_stopVisualSampling(s){s.visualSampleStarted=false;},
   acceptVisualResult(v){this.visualResult=v;},
   currentModel:model || ((diarize || translate || visual) ? 'Nemotron-3.5-Transcribe-30B-A4B-MLX' : 'Qwen3-ASR'),currentChatId:'chat',uploadAudio:null,counter:0,
   streams:{chat:{isStreaming:false}},chats:{chat:{messages:[],model:'Nemotron-3.5-Transcribe-30B-A4B-MLX'}},
   micActive(){return this.micSession!==null;},isCurrentChatStreaming(){return this.streams.chat.isStreaming;},
   resolveGatewayModelId:x=>x,getChatSession(){return this.chats.chat;},getStreamSession(){return this.streams.chat;},
   resetStreamSession(s){Object.assign(s,{isStreaming:false,streamingContent:'',finalContent:false,statusLog:[]});},
   newMessageId(){return String(++this.counter);},setEngineStatus(id,status){this.streams[id].engineStatus=status;},
   forceScrollToBottom(){},scrollToBottom(){},saveCurrentChat(){},scheduleEnhanceMessages(){},getApiKey:()=>'',
   _micWorkletSource:()=>''});
 return{app,ctx,track,timers,intervals,get ws(){return ws;},get resolveModule(){return resolveModule;},
   async start(){await app.startMicTranscription();ws.readyState=1;ws.onopen();},
   event(data){if(data.type==='transcript.done'&&data.audio_bytes===undefined)data.audio_bytes=app.micSession?.sentMicBytes||0;return ws.onmessage({data:JSON.stringify(data)});},
   audio(){app.micSession.workletNode.port.onmessage({data:new Float32Array(48000).fill(0.1)});},
   advance(ms){time+=ms;for(const[id,item]of[...timers])if(item.at<=time&&timers.has(id)){timers.delete(id);item.fn();}},
   assertStopped(){assert.equal(app.micSession,null);assert.equal(app.streams.chat.isStreaming,false);assert(track.stopped);assert.equal(ctx.state,'closed');assert.equal(ws.readyState,3);assert.equal(timers.size,0);assert.equal(intervals.size,0);}};
}
(async()=>{
 let f=fixture();await f.start();
 await f.event({type:'ready'});
 assert.equal(f.app.micSession.phase,'listening');assert(f.app.micSession.workletNode);
 f.audio();assert.equal(f.app.micAudioSeconds,1);
 await f.event({type:'transcript.delta',delta:'Live text'});
 assert.equal(f.app.streams.chat.streamingContent,'Live text');
 await f.app.stopMicTranscription();assert(f.track.stopped);assert.equal(f.ctx.state,'closed');
 assert.equal(f.app.micSession.phase,'stopping');assert(JSON.parse(f.ws.sent.at(-1)).type==='stop');
 await f.event({type:'transcript.done',text:'Live text'});f.assertStopped();
 assert.equal(f.app.chats.chat.messages.at(-1).content,'Live text');
 console.log('PASS: actual Alpine reactive app receives ready, audio, delta, stop acknowledgement and saves text');

 f=fixture({model:'Nemotron-3.5-Transcribe-30B-A4B-MLX'});await f.start();
 assert(f.ws.url.endsWith('/realtime/diarize'));
 const originalStart=JSON.parse(f.ws.sent[0]);
 assert.equal(originalStart.translate,false);assert.equal(originalStart.diarize,false);
 await f.event({type:'ready'});f.audio();await f.app.stopMicTranscription();
 await f.event({type:'transcript.done',text:'Це український оригінал.'});f.assertStopped();
 assert.equal(f.app.chats.chat.messages.at(-1).content,'Це український оригінал.');
 console.log('PASS: unchecked translation uses original-language Nemotron ASR and sends translate:false');


 f=fixture();await f.start();await f.app.stopMicTranscription();f.assertStopped();
 await f.event({type:'ready'});assert.equal(f.app.micSession,null);
 console.log('PASS: stop before ready immediately releases microphone and ignores late ready');

 f=fixture();await f.start();f.advance(60000);f.assertStopped();
 assert(f.app.chats.chat.messages.at(-1).content.includes('60 секунд'));
 console.log('PASS: missing ready deadline clears pending UI state');

 f=fixture({hangingModule:true});await f.start();const starting=f.event({type:'ready'});
 await Promise.resolve();await f.app.stopMicTranscription();f.assertStopped();
 f.resolveModule();await starting;assert.equal(f.app.micSession,null);
 console.log('PASS: stop during AudioWorklet load cannot reconnect a closed microphone');

 f=fixture({hangingModule:true});await f.start();const waiting=f.event({type:'ready'});
 await Promise.resolve();f.advance(7000);f.assertStopped();f.resolveModule();await waiting;
 console.log('PASS: hanging AudioWorklet startup deadline releases microphone');

 for(const cancel of ['timeout','send-failure']) {
   f=fixture();await f.start();await f.event({type:'ready'});f.audio();
   await f.event({type:'transcript.delta',delta:'Keep partial'});
   if(cancel==='send-failure')f.ws.throwOnSend=true;
   await f.app.stopMicTranscription();
   if(cancel==='timeout')f.advance(10000);
   f.assertStopped();assert(f.app.chats.chat.messages.at(-1).content.includes('Keep partial'));
   console.log('PASS:',cancel,'releases input/socket and retains committed text');
 }
 const source=page;
 assert(source.includes('x-show="isCurrentChatStreaming() && !micActive()" @click="stopStreaming()"'));
 assert(source.includes('>Стоп</span>'));
 console.log('PASS: generic stop hidden while the single microphone stop is visible');

 for (const mode of [{diarize:true},{translate:true}]) {
   f=fixture(mode);await f.start();await f.event({type:'ready'});f.audio();
   await f.event({type:'transcript.partial',text:'Keep preview'});
   await f.app.stopMicTranscription();
   await f.app.stopMicTranscription();f.advance(600001);
   assert(f.app.micSession,'Repeated stop/time elapsed must not drop unprocessed audio');
   assert.equal(f.ws.sent.filter(x=>typeof x==='string'&&JSON.parse(x).type==='stop').length,1);
   await f.event({type:'transcript.done',text:'Full recording including tail'});f.assertStopped();
   assert.equal(f.app.chats.chat.messages.at(-1).content,'Full recording including tail');
 }
 console.log('PASS: repeated Stop and >10 minutes waiting preserve pending finalization');

 f=fixture({diarize:true});await f.start();await f.event({type:'ready'});f.audio();
 await f.event({type:'transcript.partial',text:'Keep preview'});
 await f.app.stopMicTranscription();
 await f.event({type:'transcript.done',text:'Incorrect completion',audio_bytes:2});f.assertStopped();
 assert(f.app.chats.chat.messages.at(-1).content.includes('не підтвердив'));
 console.log('PASS: incomplete audio acknowledgement cannot be reported as a final result');

 f=fixture({diarize:true});await f.start();await f.event({type:'ready'});f.audio();
 assert(f.ws.url.endsWith('/realtime/diarize'));
 await f.event({type:'transcript.partial',text:'**Мовець 1:** Preview'});
 await f.app.stopMicTranscription();assert(f.track.stopped);assert.equal(f.ctx.state,'closed');
 f.advance(10001);assert(f.app.micSession,'Final diarization must not inherit the ordinary 10-second cutoff');
 await f.event({type:'status',detail:'Final pass'});assert.equal(f.app.streams.chat.engineStatus.text,'Final pass');
 await f.event({type:'transcript.done',text:'**Мовець 1:** Final'});f.assertStopped();
 assert.equal(f.app.chats.chat.messages.at(-1).content,'**Мовець 1:** Final');
 console.log('PASS: diarization keeps a provisional preview, waits beyond 10 s, and replaces it with final text');

 // Translation only (diarize=false, translate=true)
 f=fixture({diarize:false,translate:true});
 f.app.translationModel='chosen-translator';f.app.translationContext='Тема: медицина';
 await f.start();await f.event({type:'ready'});f.audio();
 f.app.translationModel='other';f.app.translationContext='changed';
 assert(f.ws.url.endsWith('/realtime/diarize'));
 const startMsg = JSON.parse(f.ws.sent[0]);
 assert.equal(startMsg.translate, true); assert.equal(startMsg.diarize, false);
 assert.equal(startMsg.target_language, 'Ukrainian');
 assert.equal(startMsg.translation_model, 'chosen-translator');
 assert.equal(startMsg.translation_context, 'Тема: медицина');
 assert.equal(f.app.micSession.translationModel, 'chosen-translator');
 assert.equal(f.app.micSession.translationContext, 'Тема: медицина');
 await f.event({type:'transcript.partial',text:'Попередній переклад'});
 await f.app.stopMicTranscription();
 await f.event({type:'transcript.done',text:'Остаточний переклад'});f.assertStopped();
 assert.equal(f.app.chats.chat.messages.at(-1).content,'Остаточний переклад');
 console.log('PASS: translation-only mic session connects to advanced endpoint, sends target_language Ukrainian and receives translation');

 f=fixture({diarize:true,translate:true,visual:true});await f.start();await f.event({type:'ready'});f.audio();
 const visualStart=JSON.parse(f.ws.sent[0]);
 assert.equal(visualStart.visual_enabled,true);assert.equal(visualStart.vision_model,'chosen-vlm');
 assert(f.app.micSession.visualSampleStarted);
 await f.event({type:'visual.context',visual:{context:'Release planning',labels:{0:'Андрій'}}});
 assert.equal(f.app.visualResult.context,'Release planning');
 await f.app.stopMicTranscription();assert(!f.app.micSession.visualSampleStarted);
 await f.event({type:'transcript.done',text:'**Мовець 1 · Андрій:** Final',visual:{context:'Release planning',labels:{0:'Андрій'}}});f.assertStopped();
 assert.equal(f.app.chats.chat.messages.at(-1)._visual.labels[0],'Андрій');
 console.log('PASS: visual mic mode snapshots VLM, forwards context, stops sampling and saves labels');

 for(const action of ['disconnect','error']) {
   f=fixture({diarize:true,translate:true});await f.start();await f.event({type:'ready'});f.audio();
   await f.event({type:'transcript.partial',text:'**Мовець 2:** Keep preview'});
   if(action==='disconnect')f.ws.close();
   else if(action==='error')await f.event({type:'error',detail:'inference fixture'});
   f.assertStopped();const result=f.app.chats.chat.messages.at(-1).content;
   assert(result.includes('Keep preview'));assert(result.includes('Попередній результат'));
   console.log('PASS: diarization+translate',action,'preserves preview as incomplete and releases resources');
 }
})().catch(error=>{console.error(error.stack);process.exitCode=1;}).finally(()=>{
 global.setTimeout=saved.setTimeout;global.clearTimeout=saved.clearTimeout;
 global.setInterval=saved.setInterval;global.clearInterval=saved.clearInterval;Date.now=saved.now;
});
