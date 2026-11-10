# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from omlx.transcription.native_timestamps import capture_native_timestamps, native_turn_intervals
from omlx.transcription.speech_timing import resolve_turn_intervals, timing_metadata, interval_text


TURNS = [{'speaker':7,'text':'Hello there.'},{'speaker':2,'text':'Good morning.'}]
PAYLOAD = {'error':None,'words':[
    {'word':'Hello','start':1.1,'end':1.4,'speaker':7},
    {'word':'there','start':1.4,'end':1.8,'speaker':7},
    {'word':'Good','start':4,'end':4.4,'speaker':2},
    {'word':'morning','start':4.4,'end':5,'speaker':2}]}


class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_ctc_has_priority_and_skips_approximate_asr(self):
        with patch('omlx.transcription.speech_timing.command_bytes',new=AsyncMock(return_value=b'{"format":{"duration":"10"}}')), patch(
                'omlx.transcription.speech_timing.estimate_turn_intervals',new=AsyncMock(side_effect=AssertionError('Must not approximate valid CTC'))):
            result=await resolve_turn_intervals(None,'test.wav',TURNS,native_payload=PAYLOAD)
        self.assertEqual([(t['start'],t['end']) for t in result],[(1.1,1.8),(4,5)])
        self.assertEqual([t['speaker'] for t in result],[7,2])
        self.assertEqual(timing_metadata(result)['timestamp_type'],'native_ctc')
        self.assertNotIn('≈',interval_text(result))

    async def test_ctc_alignment_error_or_out_of_file_bounds_falls_back(self):
        fallback=[{**t,'start':0,'end':10,'timing_method':'windowed_asr'} for t in TURNS]
        for bad in ({**PAYLOAD,'error':'alignment failed'}, {**PAYLOAD,'words':[{**w,'end':100} for w in PAYLOAD['words']]}):
            with patch('omlx.transcription.speech_timing.command_bytes',new=AsyncMock(return_value=b'{"format":{"duration":"10"}}')), patch(
                    'omlx.transcription.speech_timing.estimate_turn_intervals',new=AsyncMock(return_value=fallback)) as estimate:
                result=await resolve_turn_intervals(None,'test.wav',TURNS,native_payload=bad)
            self.assertEqual(result,fallback);estimate.assert_awaited_once()

    async def test_partial_ctc_keeps_native_turn_and_only_falls_back_for_missing_turn(self):
        fallback=[{**t,'start':0,'end':10,'timing_method':'windowed_asr'} for t in TURNS]
        payload={**PAYLOAD,'words':PAYLOAD['words'][:2]}
        with patch('omlx.transcription.speech_timing.command_bytes',new=AsyncMock(return_value=b'{"format":{"duration":"10"}}')), patch(
                'omlx.transcription.speech_timing.estimate_turn_intervals',new=AsyncMock(return_value=fallback)):
            result=await resolve_turn_intervals(None,'test.wav',TURNS,native_payload=payload)
        self.assertEqual(result[0]['timing_method'],'native_ctc')
        self.assertEqual(result[1]['timing_method'],'windowed_asr')
        self.assertEqual(timing_metadata(result)['timestamp_type'],'mixed_native_and_approximate')

    def test_malformed_or_unverified_ctc_is_rejected(self):
        for bad in ({'words':PAYLOAD['words']},{'error':None,'words':[]},
                    {'error':None,'words':[{'word':'Hello','start':float('nan'),'end':2}]},
                    {'error':None,'words':[{'word':'Hello','start':-1,'end':2}]}):
            self.assertEqual(native_turn_intervals(TURNS,bad),[None,None])

    def test_runtime_capture_requests_ctc_and_keeps_metadata_dropped_by_text_adapter(self):
        class Model:
            def generate(self,path,stream=True,capture_ctc_timestamps=False):
                self.requested=capture_ctc_timestamps
                yield {'text':'<spk:7>Hello there.<spk:2>Good morning.','ctc_timestamps':PAYLOAD}
        class Engine:pass
        engine=Engine();engine._model=Model()
        with capture_native_timestamps(engine,'test.wav') as capture:
            # Simulate oMLX consuming the raw result but forwarding only text.
            text=[{'text':chunk['text']} for chunk in engine._model.generate('test.wav')]
        self.assertTrue(engine._model.requested)
        self.assertEqual(capture.payload,PAYLOAD)
        self.assertEqual(native_turn_intervals(TURNS,capture.payload)[0]['start'],1.1)
        self.assertNotIn('ctc_timestamps',text[0])

    def test_capture_is_scoped_to_the_audio_file(self):
        class Model:
            def generate(self,path,capture_ctc_timestamps=False):
                self.requested=capture_ctc_timestamps
                yield {'ctc_timestamps':PAYLOAD}
        class Engine:pass
        engine=Engine();engine._model=Model()
        with capture_native_timestamps(engine,'one.wav') as first, capture_native_timestamps(engine,'two.wav') as second:
            list(engine._model.generate('one.wav'))
            self.assertEqual(first.payload,PAYLOAD);self.assertIsNone(second.payload)
        list(engine._model.generate('one.wav'))
        self.assertFalse(engine._model.requested,'Capture flag must not leak into other requests')

    async def test_pipeline_uses_native_times_to_select_per_turn_visual_context(self):
        from omlx.transcription.diarization import transcription_events
        from omlx.transcription.visual_context import VisualContext
        state=VisualContext(time_basis='video')
        for time,context in ((0,'EARLY'),(3,'LATER'),(9,'FUTURE')):
            state.observations.append({'time':time,'context':context,'people':[]})
        audio=types.SimpleNamespace(_sse_event=lambda x:'data: '+json.dumps(x),
            _cleanup_tempfile=Mock(),_record_audio_request=Mock())
        class Engine:
            async def transcribe_stream(self,*args,**kwargs):
                yield {'text':'<spk:7>Hello there.<spk:2>Good morning.','ctc_timestamps':PAYLOAD}
        async def translated(turns,*args,**kwargs):
            return [{**t,'text':'Переклад.'} for t in turns]
        translator=AsyncMock(side_effect=translated)
        with patch.dict(sys.modules,{'omlx.api':types.SimpleNamespace(audio_routes=audio)}), patch(
                'omlx.transcription.speech_timing.command_bytes',new=AsyncMock(return_value=b'{"format":{"duration":"10"}}')), patch(
                'omlx.transcription.speech_timing.estimate_turn_intervals',new=AsyncMock(side_effect=AssertionError('No fallback for native CTC'))), patch(
                'omlx.transcription.translation.translate_turns',new=translator):
            events=[json.loads(e[6:]) async for e in transcription_events(Engine(),'test.wav','native-model',100,
                visual=state,translate=True)]
        result=events[-1]
        self.assertEqual(result['timestamp_type'],'native_ctc')
        self.assertEqual([t['speaker'] for t in result['segments']],[7,2])
        self.assertEqual([t['visual_context_times'] for t in result['segments']],[[0],[3]])
        contexts=translator.await_args.kwargs['turn_contexts']
        self.assertIn('EARLY',contexts[0]);self.assertNotIn('LATER',contexts[0])
        self.assertIn('LATER',contexts[1]);self.assertNotIn('EARLY',contexts[1])
        self.assertNotIn('FUTURE',''.join(contexts))


if __name__=='__main__':unittest.main()
