const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const page = require('node:fs').readFileSync(require('node:path').join(__dirname,'../../apps/omlx-web/omlx_web/templates/chat.html'),'utf8');
const start=page.indexOf('    loadTranslationPreferences()');
const end=page.indexOf('    // ===== Realtime microphone transcription =====',start);
const methods=Function('return ({'+page.slice(start,end)+'})')();
const storage=new Map();
global.localStorage={getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)};
global.window={t:x=>x};
const make=()=>({...methods,currentModel:'Nemotron-3.5-Transcribe-30B-A4B-MLX',
    currentChatId:'test',diarizeUpload:true,translateUpload:true,targetLanguage:'Polish',
    translationModel:'chosen',translationContext:'Тема: розробка. API не перекладати.',
    streams:{test:{}},chats:{test:{messages:[]}},micActive:()=>false,
    resolveGatewayModelId:x=>x,getChatSession(){return this.chats.test;},getStreamSession(){return this.streams.test;},
    isCurrentChatStreaming(){return !!this.streams.test.isStreaming;},newMessageId:()=>crypto.randomUUID(),
    resetStreamSession(s){Object.assign(s,{isStreaming:false,streamingContent:'',statusLog:[]});},
    setEngineStatus(){},getApiKey:()=>'',scrollToBottom(){},forceScrollToBottom(){},saveCurrentChat(){},scheduleEnhanceMessages(){}});
(async()=>{
    const app=make();app.saveTranslationPreferences();
    const restored=make();restored.translationModel='';restored.translationContext='';restored.loadTranslationPreferences();
    assert.equal(restored.translationModel,'chosen');assert.equal(restored.translationContext,app.translationContext);
    global.fetch=async url=>{
        assert.equal(url,'/v1/audio/translation/models');
        return Response.json({models:['chosen','other'],default_model:'mimo_9b'});
    };
    await app.loadTranslationModels();assert.deepEqual(app.translationModels,['chosen','other']);
    assert.equal(app.translationModel,'chosen');assert.equal(app.translationDefaultModel,'mimo_9b');
    global.fetch=async()=>{throw Error('offline');};
    await app.loadTranslationModels();assert.equal(app.translationModelsError,'offline');
    assert.equal(app.translationModel,'chosen');assert.equal(app.translationModelsLoading,false);
    let request;
    global.fetch=async(url,options)=>{
        request={url,form:options.body};
        return new Response('data: '+JSON.stringify({type:'transcript.text.done',text:'Translated'})+'\n\n');
    };
    const file=new File(['fixture'],'test.wav',{type:'audio/wav'});
    app.uploadAudio={file,filename:file.name,size:file.size};
    const pending=app.sendTranscriptionMessage();
    app.translationModel='changed';app.translationContext='changed';
    await pending;
    assert.equal(request.url,'/v1/audio/transcriptions/diarize');
    assert.equal(request.form.get('translation_model'),'chosen');
    assert.equal(request.form.get('translation_context'),'Тема: розробка. API не перекладати.');
    assert.equal(request.form.get('target_language'),'Polish');
    assert.equal(request.form.has('visual_enabled'),false,'Audio-only request must remain unchanged');
    assert.equal(app.chats.test.messages.at(-1).content,'Translated');
    assert(page.includes('x-model="translationModel"'));assert(page.includes('x-model="translationContext"'));
    assert(page.includes('maxlength="6000"'));
    console.log('PASS: translation settings discovery, persistence, failure, file transport and settings locked at send');
})().catch(error=>{console.error(error);process.exitCode=1;});
