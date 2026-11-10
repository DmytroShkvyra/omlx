const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const page=require('node:fs').readFileSync(require('node:path').join(__dirname,'../../apps/omlx-web/omlx_web/templates/chat.html'),'utf8');
const a=page.indexOf('    loadTranslationPreferences()'),b=page.indexOf('    // ===== Realtime microphone transcription =====',a);
const methods=Function('return ({'+page.slice(a,b)+'})')();
global.window={t:x=>x};
const make=()=>({...methods,currentModel:'Nemotron-3.5-Transcribe-30B-A4B-MLX',currentChatId:'chat',
    visualEnabled:true,visionModel:'selected-vlm',visualFrames:[],visualSpeakerLabels:{0:'Old recording'},
    diarizeUpload:true,translateUpload:true,targetLanguage:'Ukrainian',translationContext:'Initial glossary',
    streams:{chat:{}},chats:{chat:{messages:[]}},getApiKey:()=>'',micActive:()=>false,
    getChatSession(){return this.chats.chat;},getStreamSession(){return this.streams.chat;},
    isCurrentChatStreaming(){return !!this.streams.chat.isStreaming;},resolveGatewayModelId:x=>x,
    newMessageId:()=>crypto.randomUUID(),resetStreamSession(s){s.isStreaming=false;s.streamingContent='';},
    setEngineStatus(){},forceScrollToBottom(){},scrollToBottom(){},saveCurrentChat(){},scheduleEnhanceMessages(){}});
(async()=>{
    let form,url;
    const visual={context:'Visible topic',labels:{0:'Andrii (ймовірно)'},speaker_ids:[0],candidates:['Andrii']};
    global.fetch=async(path,opts)=>{url=path;form=opts.body;return new Response([
        {type:'transcript.visual',visual},
        {type:'transcript.text.done',text:'**Andrii:** Hello',visual,segments:[{speaker:0,text:'Hello'}]},
    ].map(v=>'data: '+JSON.stringify(v)+'\n\n').join(''));};
    let app=make();
    app.videoFrameSelection='keyframes';
    const file=new File(['video'],'meeting.mp4',{type:'video/mp4'});
    app.uploadAudio={file,filename:file.name};await app.sendTranscriptionMessage();
    assert.equal(url,'/v1/audio/transcriptions/diarize');assert.equal(form.get('visual_enabled'),'true');
    assert.equal(form.get('vision_model'),'selected-vlm');assert.equal(form.get('translation_context'),'Initial glossary');
    assert.equal(form.get('frame_selection'),'keyframes');
    assert.deepEqual(app.uniqueSpeakerLabels([' Bordiuhov, Andrii [External] ','bordiuhov andrii','Bordiuhov,  Andrii External']),['Bordiuhov, Andrii External']);
    assert.equal(form.get('speaker_labels'),'{}','Voice labels must not carry between recordings');
    assert.equal(app.visualResult.context,'Visible topic');assert.equal(app.chats.chat.messages.at(-1)._visual.labels[0],'Andrii (ймовірно)');
    app.visualSpeakerLabels={0:'Андрій'};app.sendVisualLabels();
    assert.equal(app.chats.chat.messages.at(-1).content,'**Андрій:** Hello');
    app.visualSpeakerLabels={0:'Інше ім’я'};app.sendVisualLabels();
    assert.equal(app.chats.chat.messages.at(-1).content,'**Інше ім’я:** Hello');
    assert.equal(app.formatTranscriptionHeadings('**Мовець 3 · Bordiuhov, Andrii External (ймовірно):** Text'),
        '**Bordiuhov, Andrii External:** Text');
    const legacy=app.chats.chat.messages.at(-1);legacy.content='**Мовець 1 · Andrii (ймовірно):** Hello';delete legacy._transcriptionSegments;
    app.visualSpeakerLabels={0:'Андрій'};app.sendVisualLabels();app.visualSpeakerLabels={0:'Нове ім’я'};app.sendVisualLabels();
    assert.equal(legacy.content,'**Нове ім’я:** Hello');
    app=make();const audio=new File(['audio'],'sound.wav',{type:'audio/wav'});
    app.uploadAudio={file:audio,filename:audio.name};await app.sendTranscriptionMessage();
    assert.equal(form.has('visual_enabled'),false,'Audio-only input stays on old pipeline even with visual toggle checked');
    app=make();app.videoFrameSelection='changes';app.uploadAudio={file,filename:file.name};await app.sendTranscriptionMessage();
    assert.equal(form.get('frame_selection'),'changes');
    assert(page.includes('За змістовними змінами'));
    assert(page.includes('autocomplete="off"'));
    app=make();app.visualFrames=[{time:null,images:['data:image/jpeg;base64,/9j/dGVzdA==']}];
    app.uploadAudio={file:audio,filename:audio.name};await app.sendTranscriptionMessage();
    assert.equal(form.get('visual_enabled'),'true');assert.equal(JSON.parse(form.get('visual_frames')).length,1);
    let stopped=false;app.visualScreen={media:{getTracks:()=>[{stop(){stopped=true;}}]},video:{srcObject:{}}};
    app.stopVisualScreen();assert(stopped);assert.equal(app.visualScreen,null);
    console.log('PASS: visual video/frame transport, initial-context preservation, manual labels, session reset and audio-only fallback');
})().catch(e=>{console.error(e);process.exitCode=1;});
