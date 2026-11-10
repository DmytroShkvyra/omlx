# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import omlx.transcription.microphone_diarization as mic

ARTIFACTS = Path(__file__).resolve().parents[2] / '.artifacts/transcription-tests'
ARTIFACTS.mkdir(parents=True, exist_ok=True)


class PCMTests(unittest.TestCase):
    def test_short_filesystem_writes_keep_the_complete_pcm_packet(self):
        class PartialWriter:
            def __init__(self, file): self.file=file
            def write(self, data): return self.file.write(data[:4096])
            def close(self): self.file.close()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            recording=mic.PCMRecording(directory,1_000_000)
            recording.file=PartialWriter(recording.file)
            data=b'\x00\x08'*128000
            try:
                recording.append(data)
                with wave.open(recording.snapshot(recording.size)) as audio:
                    self.assertEqual(audio.readframes(audio.getnframes()),data)
            finally:
                recording.close()

    def test_snapshot_is_a_quicktime_readable_pcm_wav_of_exact_prefix(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            recording = mic.PCMRecording(directory, 100000)
            try:
                recording.append(b'\x01\x00' * 16000)
                prefix = recording.size
                recording.append(b'\x02\x00' * 8000)
                path = recording.snapshot(prefix)
                with wave.open(path) as audio:
                    self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                    self.assertEqual(audio.getnframes(), 16000)
                    self.assertEqual(audio.readframes(16000), b'\x01\x00' * 16000)
                with self.assertRaises(ValueError):
                    recording.append(b'\x00')
                with self.assertRaises(ValueError):
                    recording.append(b'\x00' * 100000)
                self.assertEqual(recording.size, 48000)
            finally:
                recording.close()


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, value):
        self.sent.append(value)

    async def feed(self, value):
        await self.incoming.put(value)


class Engine:
    def __init__(self):
        self.active = set()

    def _begin_activity(self, *args, **kwargs):
        self.active.add('recording')
        return 'recording'

    async def _finish_activity(self, value):
        self.active.remove(value)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_includes_audio_received_while_preview_was_running(self):
        socket, engine = Socket(), Engine()
        entered, proceed = asyncio.Event(), asyncio.Event()
        sizes = []

        async def inference(engine, path, *args, **kwargs):
            try:
                with wave.open(path) as audio:
                    sizes.append(audio.getnframes())
                entered.set()
                await proceed.wait()
                yield 'data: ' + json.dumps({'type': 'transcript.text.delta', 'delta': 'preview'})
                yield 'data: ' + json.dumps({'type': 'transcript.text.done', 'text': f'frames={sizes[-1]}', 'segments': []})
            finally:
                Path(path).unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory, patch.object(mic, 'PREVIEW_SECONDS', 1), patch.object(mic, 'transcription_events', inference):
            task = asyncio.create_task(mic.run_microphone_session(socket, engine, 'test', directory, 100000))
            await socket.feed({'bytes': b'\x00\x00' * 16000})
            await asyncio.wait_for(entered.wait(), 2)
            await socket.feed({'bytes': b'\x00\x00' * 8000})
            await socket.feed({'text': '{"type":"stop"}'})
            await asyncio.sleep(0)
            proceed.set()
            await asyncio.wait_for(task, 2)
            self.assertEqual(sizes, [16000, 24000])
            self.assertEqual(socket.sent[-1]['type'], 'transcript.done')
            self.assertEqual(socket.sent[-1]['text'], 'frames=24000')
            self.assertFalse(socket.sent[-1]['provisional'])
            self.assertEqual(socket.sent[-1]['audio_bytes'], 48000)
            self.assertEqual(socket.sent[-1]['audio_seconds'], 1.5)
            self.assertTrue(any(event.get('provisional') for event in socket.sent))
            self.assertFalse(engine.active)
            self.assertEqual(list(Path(directory).glob('*.wav')), [])

    async def test_stop_before_first_preview_processes_entire_short_buffer(self):
        socket, engine = Socket(), Engine()
        snapshots = []

        async def inference(engine, path, *args, **kwargs):
            try:
                with wave.open(path) as audio:
                    snapshots.append(audio.readframes(audio.getnframes()))
                yield 'data: ' + json.dumps({'type': 'transcript.text.done',
                    'text': 'Last spoken words', 'segments': []})
            finally:
                Path(path).unlink(missing_ok=True)

        pcm = b'\x01\x00' * 8000 + b'\x02\x00' * 384
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory, patch.object(mic, 'transcription_events', inference):
            await socket.feed({'bytes': pcm[:-768]})
            await socket.feed({'bytes': pcm[-768:]})
            await socket.feed({'text': json.dumps({'type': 'stop', 'audio_bytes': len(pcm)})})
            await asyncio.wait_for(mic.run_microphone_session(socket, engine, 'test', directory, 100000), 2)
        self.assertEqual(snapshots, [pcm])
        self.assertEqual(socket.sent[-1]['type'], 'transcript.done')
        self.assertEqual(socket.sent[-1]['audio_bytes'], len(pcm))
        self.assertEqual(socket.sent[-1]['text'], 'Last spoken words')

    async def test_missing_tail_is_reported_instead_of_finalized(self):
        socket, engine = Socket(), Engine()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            await socket.feed({'bytes': b'\x00\x00' * 100})
            await socket.feed({'text': '{"type":"stop","audio_bytes":400}'})
            with self.assertRaisesRegex(RuntimeError, 'не весь аудіобуфер'):
                await asyncio.wait_for(mic.run_microphone_session(socket, engine, 'test', directory, 100000), 2)
        self.assertFalse(any(event['type'] == 'transcript.done' for event in socket.sent))
        self.assertFalse(engine.active)

    async def test_translation_session_sets_metadata_and_prompts(self):
        socket, engine = Socket(), Engine()
        captured_kwargs = {}

        async def inference(engine, path, *args, **kwargs):
            captured_kwargs.update(kwargs)
            try:
                yield 'data: ' + json.dumps({'type': 'transcript.text.delta', 'delta': 'переклад'})
                yield 'data: ' + json.dumps({'type': 'transcript.text.done', 'text': 'переклад', 'segments': []})
            finally:
                Path(path).unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory, patch.object(mic, 'PREVIEW_SECONDS', 1), patch.object(mic, 'transcription_events', inference):
            task = asyncio.create_task(mic.run_microphone_session(socket, engine, 'test', directory, 100000,
                                                                 diarize=False, translate=True, target_language='Ukrainian',
                                                                 translation_model='chosen', translation_context='Тема: медицина'))
            await socket.feed({'bytes': b'\x00\x00' * 16000})
            await socket.feed({'text': '{"type":"stop"}'})
            await asyncio.wait_for(task, 2)
            self.assertEqual(captured_kwargs.get('translate'), True)
            self.assertEqual(captured_kwargs.get('diarize'), False)
            self.assertEqual(captured_kwargs.get('target_language'), 'Ukrainian')
            self.assertEqual(captured_kwargs.get('translation_model'), 'chosen')
            self.assertEqual(captured_kwargs.get('translation_context'), 'Тема: медицина')
            self.assertEqual(socket.sent[-1]['text'], 'переклад')
            self.assertEqual(socket.sent[-1]['translate'], True)
            self.assertEqual(socket.sent[-1]['target_language'], 'Ukrainian')

    async def test_empty_stop_does_not_invoke_model(self):
        socket, engine = Socket(), Engine()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            await socket.feed({'text': '{"type":"stop"}'})
            await asyncio.wait_for(mic.run_microphone_session(socket, engine, 'test', directory, 100000), 2)
        self.assertEqual(socket.sent[-1]['text'], '')
        self.assertFalse(engine.active)

    async def test_disconnect_cancels_preview_and_releases_activity(self):
        socket, engine = Socket(), Engine()
        entered, closed = asyncio.Event(), asyncio.Event()

        async def inference(engine, path, *args, **kwargs):
            try:
                entered.set()
                await asyncio.Future()
                yield ''
            finally:
                Path(path).unlink(missing_ok=True)
                closed.set()

        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory, patch.object(mic, 'PREVIEW_SECONDS', 1), patch.object(mic, 'transcription_events', inference):
            task = asyncio.create_task(mic.run_microphone_session(socket, engine, 'test', directory, 100000))
            await socket.feed({'bytes': b'\x00\x00' * 16000})
            await asyncio.wait_for(entered.wait(), 2)
            await socket.feed({'type': 'websocket.disconnect'})
            await asyncio.wait_for(task, 2)
            self.assertTrue(closed.is_set())
            self.assertFalse(engine.active)
            self.assertFalse(any(event['type'] == 'transcript.done' for event in socket.sent))

    async def test_invalid_pcm_stops_without_running_inference(self):
        socket, engine = Socket(), Engine()
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            await socket.feed({'bytes': b'\x00'})
            with self.assertRaisesRegex(RuntimeError, 'парної'):
                await asyncio.wait_for(mic.run_microphone_session(socket, engine, 'test', directory, 100000), 2)
        self.assertFalse(engine.active)


if __name__ == '__main__':
    unittest.main()
