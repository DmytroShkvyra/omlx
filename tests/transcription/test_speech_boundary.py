# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from omlx.transcription.speech_boundary import SpeechBoundary
import omlx.transcription.microphone_diarization as mic
from tests.transcription.test_microphone_diarization import Socket, Engine, ARTIFACTS


def pcm(seconds, speech=True):
    return (b'\x00\x08' if speech else b'\x00\x00') * round(16000 * seconds)


class BoundaryTests(unittest.TestCase):
    def test_pause_after_speech_triggers_before_four_seconds(self):
        gate = SpeechBoundary()
        gate.feed(pcm(1)); gate.feed(pcm(.4, False))
        self.assertIsNone(gate.due(44800))
        gate.feed(pcm(.2, False))
        self.assertEqual(gate.due(51200), 'silence')
        gate.snapshot_started(51200)
        gate.feed(pcm(3, False))
        self.assertIsNone(gate.due(147200), 'The same pause must not repeatedly trigger translation')

    def test_continuous_speech_reaches_maximum_four_second_interval(self):
        gate = SpeechBoundary()
        gate.feed(pcm(3.8))
        self.assertIsNone(gate.due(121600))
        gate.feed(pcm(.2))
        self.assertEqual(gate.due(128000), 'max_interval')
        gate.snapshot_started(128000)
        gate.feed(pcm(3.8)); self.assertIsNone(gate.due(249600))
        gate.feed(pcm(.2)); self.assertEqual(gate.due(256000), 'max_interval')

    def test_packet_boundaries_do_not_change_detection_and_silence_is_idle(self):
        gate = SpeechBoundary()
        gate.feed(pcm(12, False)); self.assertIsNone(gate.due(384000))
        data = pcm(.4) + pcm(.6, False)
        for offset in range(0, len(data), 382):
            gate.feed(data[offset:offset+382])
        self.assertEqual(gate.due(416000), 'silence')


class StreamingBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def run_trigger(self, chunks, expected):
        socket, engine = Socket(), Engine()
        entered, proceed = asyncio.Event(), asyncio.Event()
        frames = []

        async def inference(engine, path, *args, **kwargs):
            try:
                with wave.open(path) as audio:
                    frames.append(audio.getnframes())
                self.assertTrue(kwargs['translate'])
                entered.set()
                await proceed.wait()
                yield 'data: '+json.dumps({'type':'transcript.text.done','text':'Переклад','segments':[]})
            finally:
                Path(path).unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory, patch.object(mic, 'transcription_events', inference):
            task = asyncio.create_task(mic.run_microphone_session(socket, engine, 'test', directory,
                                                                  1_000_000, translate=True))
            for chunk in chunks:
                await socket.feed({'bytes':chunk})
            await asyncio.wait_for(entered.wait(), 2)
            statuses = [e for e in socket.sent if e.get('trigger')]
            self.assertEqual(statuses[0]['trigger'], expected)
            self.assertEqual(frames[0], sum(map(len,chunks))//2)
            # New words plus pause arrive while an earlier inference is running.
            tail = pcm(.3)+pcm(.6,False)
            await socket.feed({'bytes':tail})
            await socket.feed({'text':json.dumps({'type':'stop','audio_bytes':sum(map(len,chunks))+len(tail)})})
            await asyncio.sleep(0)
            proceed.set()
            await asyncio.wait_for(task, 2)
            self.assertEqual(len(frames),2)
            self.assertEqual(frames[-1],(sum(map(len,chunks))+len(tail))//2)
            self.assertEqual(socket.sent[-1]['type'],'transcript.done')
            self.assertEqual(socket.sent[-1]['audio_bytes'],frames[-1]*2)
            ready=next(e for e in socket.sent if e['type']=='ready')
            self.assertEqual(ready['preview_seconds'],4)
            self.assertEqual(ready['silence_seconds'],.6)
            self.assertFalse(engine.active)

    async def test_pause_starts_translation_and_stop_keeps_pending_tail(self):
        await self.run_trigger([pcm(1),pcm(.6,False)],'silence')

    async def test_four_seconds_starts_translation_without_waiting_for_stop(self):
        await self.run_trigger([pcm(4)],'max_interval')
