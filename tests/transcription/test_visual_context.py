# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from omlx.transcription.visual_context import VisualContext, frame_timestamp, image_uri, labeled_text, match_speaker, visible_references
from omlx.transcription.diarization import transcription_events
import omlx.transcription.microphone_diarization as mic
from tests.transcription.test_microphone_diarization import Socket, Engine, ARTIFACTS

JPEG = 'data:image/jpeg;base64,' + base64.b64encode(b'\xff\xd8\xfftest').decode()
QUOTE = 'Today we will discuss the new software release'


def observation(label='Andrii', quote=QUOTE, active=True):
    return {'time': None, 'context': 'Software release', 'people': [
        {'label': label, 'name': label, 'description': 'Людина в окулярах',
         'active': active, 'evidence': 'named_caption', 'confidence': .95, 'quote': quote}]}


class VisualTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        connection = patch('omlx.transcription.visual_context.translation_connection', return_value=('http://127.0.0.1:8000/v1', {}))
        connection.start()
        self.addCleanup(connection.stop)

    def test_references_preserve_exact_spelling_and_reject_invalid_data(self):
        def ref(value, kind='id', legibility='complete'):
            return {'kind': kind, 'value': value, 'legibility': legibility}
        originals = [ref('https://Example.org/A_b?q=X%2FY&n=02#Sec[1]', 'url'),
                     ref('s3://Bucket/Some_Key-01', 'uri'), ref('REQ-AbC_007'), ref('REQ-abc_007'),
                     ref('https://host/visible…', 'url', 'partial')]
        self.assertEqual(visible_references(originals + [originals[0], ref('bad\nvalue'),
            ref('x' * 1025), ref('hidden', legibility='guessed'), None]), originals)
        self.assertEqual(visible_references('invalid'), [])
        bounded = visible_references([ref('"' * 1000 + str(i)) for i in range(12)])
        self.assertLessEqual(len(json.dumps(bounded, ensure_ascii=False)), 3602)
        self.assertTrue(all(len(item['value']) == 1001 for item in bounded))

    async def test_vlm_references_survive_metadata_and_scoped_context(self):
        references = [{'kind': 'url', 'value': 'https://Host/A_b?ID=Ab%2F1#part[0]', 'legibility': 'complete'},
                      {'kind': 'id', 'value': 'Trace_ID-00aB', 'legibility': 'partial'}]
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=Mock(json=Mock(return_value={'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps(
                {'context': '', 'people': [], 'references': references})}}]})))
        with patch.dict(sys.modules, {'httpx': types.SimpleNamespace(AsyncClient=Mock(return_value=client))}):
            state = VisualContext(time_basis='video')
            result = await state.analyze({'time': 12, 'images': [JPEG]})
        self.assertEqual(result['references'], references)
        self.assertEqual(state.metadata()['observations'][0]['references'], references)
        self.assertEqual(state.context_for_interval(0, 10), '')
        context = state.context_for_interval(12, 20)
        self.assertIn('[00:00:12.000]', context)
        for reference in references:
            self.assertIn(reference['value'], context)
        self.assertIn('partial', context)
        state.observations.append({'time': 30, 'people': [], 'context': 'Next slide'})
        self.assertNotIn(references[0]['value'], state.context_for_interval(31, 40))
        instruction = client.post.await_args.kwargs['json']['messages'][0]['content']
        self.assertIn('Pay special attention to URLs', instruction)
        self.assertIn('never guess hidden characters', instruction)

    def test_context_timestamps_follow_video_positions_and_keep_distinct_times(self):
        state = VisualContext(time_basis='video')
        for time in (5.005, 120.125, None):
            state.observations.append({**observation(), 'time': time})
        text = state.context()
        self.assertIn('[00:00:05.005]', text)
        self.assertIn('[00:02:00.125]', text)
        self.assertIn('[Кадр без часової прив’язки]', text)
        self.assertEqual(state.metadata()['observations'][1]['timestamp'], '00:02:00.125')
        self.assertEqual(state.metadata()['time_basis'], 'video')
        self.assertEqual(frame_timestamp(3599.9996), '01:00:00.000')

    def test_unique_long_phrase_required_for_audio_voice_id(self):
        segments = [{'speaker': 7, 'text': QUOTE + '.'}, {'speaker': 2, 'text': 'Yes, thanks.'}]
        self.assertEqual(match_speaker(QUOTE, segments), 7)
        self.assertIsNone(match_speaker('Yes thanks', segments))
        self.assertIsNone(match_speaker(QUOTE, segments + [{'speaker': 2, 'text': QUOTE}]))

    def test_frames_are_validated_and_queue_bounded(self):
        state = VisualContext()
        for i in range(20):
            state.enqueue({'time': i, 'images': [JPEG]})
        self.assertEqual(len(state.pending), 12)
        self.assertEqual(state.pending[0]['time'], 8)
        for value in ('https://example.com/image.jpg', 'data:image/jpeg;base64,bad'):
            with self.assertRaises(ValueError):
                image_uri(value)
        with self.assertRaises(ValueError):
            state.enqueue({'time': float('nan'), 'images': [JPEG]})

    async def test_visible_person_without_active_evidence_is_not_assigned(self):
        state = VisualContext()
        state.observations.append(observation(active=False))
        state.audio_quote = AsyncMock()
        self.assertEqual(await state.associate(None, 'unused', [{'speaker': 7, 'text': QUOTE}]), {})
        state.audio_quote.assert_not_awaited()
        self.assertIn('Andrii', state.context())

    async def test_provisional_label_keeps_audio_id_and_manual_override(self):
        state = VisualContext()
        state.observations.append(observation())
        segments = [{'speaker': 7, 'text': QUOTE}]
        self.assertEqual(await state.associate(None, 'unused', segments), {7: 'Andrii (ймовірно)'})
        state.update_labels({'7': 'Андрій'})
        self.assertEqual(await state.associate(None, 'unused', segments), {7: 'Андрій'})
        self.assertEqual(labeled_text('<spk:7>Hello.', state.labels), '**Андрій:** Hello.')

    async def test_conflicting_people_do_not_overwrite_audio_labels(self):
        state = VisualContext()
        state.observations.extend([observation(), observation('Daniel')])
        self.assertEqual(await state.associate(None, 'unused', [{'speaker': 0, 'text': QUOTE}]), {})

    async def test_single_open_mouth_is_not_sufficient_evidence(self):
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        payload = {'context': 'Meeting', 'people': [dict(name='Visible Name', description='Окуляри',
                    active=True, evidence='mouth_motion', confidence=.99, quote=QUOTE)]}
        client.post = AsyncMock(return_value=Mock(json=Mock(return_value={'choices':[
            {'finish_reason': 'stop', 'message': {'content': '◁think▷private reasoning◁/think▷```json\n'+json.dumps(payload)+'\n```'}}]})))
        module = types.SimpleNamespace(AsyncClient=Mock(return_value=client))
        with patch.dict(sys.modules, {'httpx': module}):
            state = VisualContext('chosen-vlm')
            value = await state.analyze({'time': 2, 'images': [JPEG]})
        self.assertFalse(value['people'][0]['active'])
        self.assertEqual(client.post.await_args.kwargs['json']['model'], 'chosen-vlm')

    async def test_vlm_error_keeps_audio_usable(self):
        state = VisualContext()
        state.enqueue({'time': None, 'images': [JPEG]})
        state.analyze = AsyncMock(side_effect=ValueError('invalid JSON'))
        await state.drain()
        self.assertIn('invalid JSON', state.warnings[0])
        self.assertFalse(state.pending)

    async def test_visual_context_is_separate_from_user_context_in_translation(self):
        state = VisualContext()
        state.observations.append(observation())
        state.observations.append({'time':100,'context':'FUTURE_SLIDE','people':[]})
        audio = types.SimpleNamespace(_sse_event=lambda x: 'data: '+json.dumps(x),
                                      _cleanup_tempfile=Mock(), _record_audio_request=Mock())
        engine = Mock()
        async def stream(*args, **kwargs):
            yield {'text': '<spk:7>' + QUOTE}
        engine.transcribe_stream = stream
        translator = AsyncMock(return_value=[{'speaker': 7, 'text': 'Сьогодні обговоримо новий випуск.', 'start':0, 'end':15}])
        with patch.dict(sys.modules, {'omlx.api': types.SimpleNamespace(audio_routes=audio)}), patch('omlx.transcription.translation.translate_turns', new=translator), patch(
                 'omlx.transcription.speech_timing.estimate_turn_intervals',new=AsyncMock(return_value=[{'speaker':7,'text':QUOTE,'start':0,'end':15}])):
            events = [json.loads(x[6:]) async for x in transcription_events(engine, 'unused', 'test', 100,
                translate=True, translation_context='My initial glossary', visual=state)]
        self.assertEqual(translator.await_args.kwargs['context'], 'My initial glossary')
        self.assertIn('Software release', translator.await_args.kwargs['turn_contexts'][0])
        self.assertNotIn('FUTURE_SLIDE', translator.await_args.kwargs['turn_contexts'][0])
        self.assertIn('**Andrii:**', events[-1]['text'])
        self.assertEqual(events[-1]['segments'][0]['speaker'], 7)
        self.assertTrue(events[-1]['timestamps'])

    def test_temporal_context_excludes_future_and_obsolete_scenes(self):
        state = VisualContext(time_basis='video')
        for time, topic in ((0,'INTRO'),(15,'API'),(30,'DATABASE'),(60,'FUTURE')):
            state.observations.append({'time':time,'context':topic,'people':[]})
        middle = state.context_for_interval(20,40)
        self.assertIn('API',middle)
        self.assertIn('DATABASE',middle)
        self.assertNotIn('INTRO',middle)
        self.assertNotIn('FUTURE',middle)
        self.assertEqual(state.context_for_interval(None,None),'')
        self.assertNotIn('API',state.context_for_interval(0,15))
        self.assertIn('API',state.context_for_interval(15,30))
        self.assertNotIn('DATABASE',state.context_for_interval(15,30))

    async def test_stop_reprocesses_labels_updated_during_inference(self):
        state, socket, engine = VisualContext(), Socket(), Engine()
        entered, proceed = asyncio.Event(), asyncio.Event()
        revisions = []
        async def inference(engine, path, *args, **kwargs):
            revision = state.version
            revisions.append(revision)
            try:
                entered.set()
                await proceed.wait()
                yield 'data: '+json.dumps({'type':'transcript.text.done', 'text':str(revision),
                    'segments':[], 'visual':{'processed_version':revision}})
            finally:
                Path(path).unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as folder, patch.object(mic, 'PREVIEW_SECONDS', 1), patch.object(mic,'transcription_events',inference):
            task=asyncio.create_task(mic.run_microphone_session(socket,engine,'test',folder,100000,visual=state))
            await socket.feed({'bytes':b'\x00\x00'*16000})
            await asyncio.wait_for(entered.wait(),2)
            await socket.feed({'text':json.dumps({'type':'visual.labels','labels':{'0':'Андрій'}})})
            await socket.feed({'text':'{"type":"stop","audio_bytes":32000}'})
            await asyncio.sleep(0)
            proceed.set()
            await asyncio.wait_for(task,2)
        self.assertEqual(revisions,[1,2])
        self.assertEqual(socket.sent[-1]['text'],'2')


if __name__ == '__main__':
    unittest.main()
