# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from omlx.transcription.diarization import transcription_events
from omlx.transcription.translation import check_language, parse_translation, translate_turns, translation_options, available_translation_models


class TranslationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        connection = patch('omlx.transcription.translation.translation_connection', return_value=('http://127.0.0.1:8000/v1', {}))
        connection.start()
        self.addCleanup(connection.stop)

    def test_translation_options_default_and_validation(self):
        with patch.dict('os.environ', {'TRANSLATION_MODEL': 'server-default'}):
            self.assertEqual(translation_options('', '  Терміни  '), ('server-default', 'Терміни'))
            self.assertEqual(translation_options('chosen', ''), ('chosen', ''))
        for model, context in (([], ''), ('m' * 513, ''), ('chosen', None), ('chosen', 'x' * 6001)):
            with self.assertRaises(ValueError):
                translation_options(model, context)

    async def test_model_discovery_filters_audio_and_preserves_model_ids(self):
        client, module = self.fake_client([])
        client.get = AsyncMock(side_effect=[Mock(json=Mock(return_value={'data':[
            {'id':'text-b'}, {'id':'speech'}, {'id':'text-a'}, {'id':'text-b'}]})),
            Mock(json=Mock(return_value={'models':[
                {'id':'text-b','model_type':'vlm'}, {'id':'speech','model_type':'audio_stt'},
                {'id':'text-a','model_type':'llm'}]}))])
        with patch.dict(sys.modules, {'httpx': module}):
            result = await available_translation_models()
        self.assertEqual(result['models'], ['text-a', 'text-b'])

    def test_single_object_response_from_screenshot_is_accepted(self):
        response = '{"id": 0, "text": "Dziękuję za obejrzenie."}'
        self.assertEqual(parse_translation(response, [{'id': 0}]), ['Dziękuję za obejrzenie.'])
        self.assertEqual(parse_translation('```json\n' + response + '\n```', [{'id': 0}]),
                         ['Dziękuję za obejrzenie.'])

    def test_single_object_still_requires_exact_id_and_nonempty_text(self):
        for response in ({'id': 1, 'text': 'Dziękuję.'}, {'id': '0', 'text': 'Dziękuję.'},
                         {'text': 'Dziękuję.'}, {'id': 0, 'text': ''}):
            with self.assertRaises(ValueError):
                parse_translation(json.dumps(response), [{'id': 0}])

    def test_single_object_cannot_replace_multiple_turns(self):
        with self.assertRaisesRegex(ValueError, 'кількість'):
            parse_translation('{"id": 0, "text": "Dziękuję."}', [{'id': 0}, {'id': 1}])

    async def test_single_turn_object_keeps_original_speaker_without_retry(self):
        client, module = self.fake_client([{'id': 0, 'text': 'Dziękuję za obejrzenie.'}])
        with patch.dict(sys.modules, {'httpx': module}):
            result = await translate_turns([{'speaker': 5, 'text': 'Дякую за перегляд.'}], 'Polish')
        self.assertEqual(result, [{'speaker': 5, 'text': 'Dziękuję za obejrzenie.'}])
        self.assertEqual(client.post.await_count, 1)

    async def test_wrong_language_is_rejected(self):
        with patch('omlx.transcription.translation.detect_language', return_value={'en': .99, 'pl': .01}):
            with self.assertRaisesRegex(ValueError, 'визначено мову en'):
                await check_language('This is English speech and it must not be accepted as a Polish translation.', 'Polish')

    def test_dropped_reordered_or_invented_turns_are_rejected(self):
        batch = [{'id': 0}, {'id': 1}]
        for values in ([{'id': 0, 'text': 'ok'}],
                       [{'id': 1, 'text': 'ok'}, {'id': 0, 'text': 'ok'}],
                       [{'id': 0, 'text': 'ok'}, {'id': 1, 'text': '<spk:7>other'}],
                       [{'id': 0, 'text': 'ok'}, {'id': 1, 'text': ''}]):
            with self.assertRaises(ValueError):
                parse_translation(json.dumps(values), batch)

    def fake_client(self, replies):
        client = Mock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        responses = [Mock(json=Mock(return_value={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(reply)}}]})) for reply in replies]
        client.post = AsyncMock(side_effect=responses)
        module = types.SimpleNamespace(AsyncClient=Mock(return_value=client))
        return client, module

    async def test_retry_retains_nonsequential_speaker_ids(self):
        turns = [{'speaker': 7, 'text': 'Hello'}, {'speaker': 2, 'text': 'Good morning'}]
        client, module = self.fake_client([
            [{'id': 0, 'text': 'Hello'}, {'id': 1, 'text': 'Good morning'}],
            [{'id': 0, 'text': 'Cześć'}, {'id': 1, 'text': 'Dzień dobry'}]])
        with patch.dict(sys.modules, {'httpx': module}), patch('omlx.transcription.translation.check_language',
                new=AsyncMock(side_effect=[ValueError('English instead of Polish'), None])):
            result = await translate_turns(turns, 'Polish')
        self.assertEqual(result, [{'speaker': 7, 'text': 'Cześć'}, {'speaker': 2, 'text': 'Dzień dobry'}])
        self.assertEqual(client.post.await_count, 2)
        self.assertNotIn('speaker', client.post.call_args.kwargs['json']['messages'][1]['content'])
        self.assertEqual(turns[0]['text'], 'Hello')

    async def test_second_wrong_language_fails_not_silent_fallback(self):
        reply = [{'id': 0, 'text': 'Wrong language'}]
        client, module = self.fake_client([reply, reply])
        with patch.dict(sys.modules, {'httpx': module}), patch('omlx.transcription.translation.check_language',
                new=AsyncMock(side_effect=ValueError('English instead of Polish'))):
            with self.assertRaisesRegex(RuntimeError, 'не пройшов перевірку'):
                await translate_turns([{'text': 'Hello'}], 'Polish')
        self.assertEqual(client.post.await_count, 2)

    async def test_omitted_short_turn_is_recovered_individually(self):
        turns = [{'speaker': 7, 'text': 'Hello'}, {'speaker': 2, 'text': 'Yes'}]
        client, module = self.fake_client([
            [{'id': 0, 'text': 'Cześć'}], [{'id': 0, 'text': 'Cześć'}],
            [{'id': 0, 'text': 'Cześć'}], [{'id': 0, 'text': 'Tak'}]])
        with patch.dict(sys.modules, {'httpx': module}), patch('omlx.transcription.translation.check_language', new=AsyncMock()):
            result = await translate_turns(turns, 'Polish', model='chosen-translator',
                                           context='API залишати як API. Тема: розробка.', dynamic_context='Visible slide: Clickstream')
        self.assertEqual(result, [{'speaker': 7, 'text': 'Cześć'}, {'speaker': 2, 'text': 'Tak'}])
        self.assertEqual(client.post.await_count, 4)
        for call in client.post.await_args_list:
            payload = call.kwargs['json']
            self.assertEqual(payload['model'], 'chosen-translator')
            self.assertIn('API залишати як API', payload['messages'][0]['content'])
            self.assertIn('Visible slide: Clickstream', payload['messages'][0]['content'])

    async def test_long_turn_is_reassembled_with_same_speaker(self):
        turns = [{'speaker': 3, 'text': 'Hello. ' * 500}, {'speaker': 0, 'text': 'Goodbye'}]
        from omlx.transcription.translation import split_turns
        pieces = split_turns(turns)
        reply = [{'id': p['id'], 'text': f"Fragment {p['id']}"} for p in pieces]
        _, module = self.fake_client([reply[:1], reply[1:]])
        with patch.dict(sys.modules, {'httpx': module}), patch('omlx.transcription.translation.check_language', new=AsyncMock()):
            result = await translate_turns(turns, 'Polish')
        self.assertEqual([t['speaker'] for t in result], [3, 0])
        self.assertEqual(result[0]['text'], 'Fragment 0 Fragment 1')

    async def test_different_time_contexts_are_not_mixed_in_one_translation_request(self):
        turns=[{'speaker':7,'text':'Hello','start':0,'end':15},
               {'speaker':2,'text':'Good morning','start':30,'end':45}]
        client,module=self.fake_client([[{'id':0,'text':'Cześć'}],[{'id':1,'text':'Dzień dobry'}]])
        with patch.dict(sys.modules,{'httpx':module}):
            result=await translate_turns(turns,'Polish',context='USER_GLOSSARY',
                turn_contexts=['[00:00:00] EARLY_SLIDE','[00:00:30] LATER_SLIDE'])
        requests=[c.kwargs['json']['messages'][0]['content'] for c in client.post.await_args_list]
        self.assertEqual(len(requests),2)
        self.assertIn('EARLY_SLIDE',requests[0]);self.assertNotIn('LATER_SLIDE',requests[0])
        self.assertIn('LATER_SLIDE',requests[1]);self.assertNotIn('EARLY_SLIDE',requests[1])
        self.assertTrue(all('USER_GLOSSARY' in r for r in requests))
        self.assertEqual([t['start'] for t in result],[0,30])

    async def test_translation_receives_exact_technical_references_and_partial_guard(self):
        url = 'https://Host/A_b?ID=X%2FY#part[1]'
        translated = 'Дивись ' + url + ' та REQ-AbC_007.'
        client, module = self.fake_client([[{'id': 0, 'text': translated}]])
        with patch.dict(sys.modules, {'httpx': module}), patch('omlx.transcription.translation.check_language', new=AsyncMock()):
            result = await translate_turns([{'speaker': 2, 'text': 'See ' + url + ' and REQ-AbC_007.'}],
                'Ukrainian', turn_contexts=[json.dumps({'references': [
                    {'kind': 'url', 'value': url, 'legibility': 'complete'},
                    {'kind': 'id', 'value': 'REQ-AbC_007', 'legibility': 'partial'}]})])
        self.assertEqual(result[0]['text'], translated)
        instruction = client.post.await_args.kwargs['json']['messages'][0]['content']
        self.assertIn(url, instruction)
        self.assertIn('References marked partial', instruction)
        self.assertIn('Do not insert unspoken links or identifiers', instruction)
        self.assertIn('character for character', instruction)

    async def test_cancel_closes_translation_connection(self):
        client, module = self.fake_client([])
        entered = asyncio.Event()

        async def wait_for_reply(*args, **kwargs):
            entered.set()
            await asyncio.Future()

        client.post.side_effect = wait_for_reply
        with patch.dict(sys.modules, {'httpx': module}):
            task = asyncio.create_task(translate_turns([{'text': 'Hello'}], 'Polish'))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        client.__aexit__.assert_awaited_once()

    async def test_token_limit_is_not_accepted_as_complete_translation(self):
        reply = [{'id': 0, 'text': 'Cześć'}]
        client, module = self.fake_client([reply, reply])
        client.post.side_effect = [Mock(json=Mock(return_value={'choices': [
            {'finish_reason': 'length', 'message': {'content': json.dumps(reply)}}]}))] * 2
        with patch.dict(sys.modules, {'httpx': module}):
            with self.assertRaisesRegex(RuntimeError, 'обірвано'):
                await translate_turns([{'text': 'Hello'}], 'Polish')


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def run_pipeline(self, translator, **options):
        audio = types.SimpleNamespace(_sse_event=lambda v: 'data: ' + json.dumps(v),
            _cleanup_tempfile=Mock(), _record_audio_request=Mock())
        engine = Mock()
        closed = []

        async def stream(*args, **kwargs):
            self.assertEqual(kwargs['task'], 'saa')
            self.assertIsNone(kwargs['language'])
            self.assertIsNone(kwargs['prompt'])
            try:
                yield {'text': '<spk:7>Hello.<spk:2>Good morning.'}
            finally:
                closed.append(True)

        engine.transcribe_stream = stream
        with patch.dict(sys.modules, {'omlx.api': types.SimpleNamespace(audio_routes=audio)}), patch('omlx.transcription.translation.translate_turns', new=translator):
            events = [json.loads(event[6:]) async for event in transcription_events(
                engine, 'test.wav', 'test', 100, translate=True, diarize=True, target_language='Polish', **options)]
        self.assertTrue(closed)
        audio._cleanup_tempfile.assert_called_once_with('test.wav')
        return events

    async def test_audio_diarization_then_text_translation(self):
        translator = AsyncMock(return_value=[
            {'speaker': 7, 'text': 'Cześć.'}, {'speaker': 2, 'text': 'Dzień dobry.'}])
        events = await self.run_pipeline(translator, translation_model='chosen', translation_context='Names: Daniel')
        self.assertEqual(translator.await_args.kwargs, {'model':'chosen', 'context':'Names: Daniel'})
        result = events[-1]
        self.assertEqual(result['type'], 'transcript.text.done')
        self.assertEqual([s['speaker'] for s in result['segments']], [7, 2])
        self.assertEqual(result['source_segments'][0]['text'], 'Hello.')
        self.assertNotIn('Hello', ''.join(e.get('delta', '') for e in events))
        self.assertEqual(result['translation_model'], 'chosen')

    async def test_translation_failure_keeps_original_and_never_reports_success(self):
        events = await self.run_pipeline(AsyncMock(side_effect=RuntimeError('translator offline')))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('Hello.', events[-1]['error']['message'])
        self.assertIn('Мовець 8', events[-1]['error']['message'])
        self.assertFalse(any(e['type'] == 'transcript.text.done' for e in events))


if __name__ == '__main__':
    unittest.main()
