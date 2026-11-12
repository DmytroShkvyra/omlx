# SPDX-License-Identifier: Apache-2.0
# Fork modification: migrated meeting transcription features from the local prototype.
"""PCM microphone diarization and translation with cumulative previews and a final SAA pass."""
import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import time
import wave

from .diarization import DEFAULT_MAX_TOKENS, transcription_events
from .speech_boundary import SpeechBoundary

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2
PREVIEW_SECONDS = 15
TRANSLATION_SECONDS = 4
TRANSLATION_SILENCE_SECONDS = .6


class PCMRecording:
    """Bounded disk spool; each inference gets an immutable WAV prefix."""
    def __init__(self, directory, max_bytes):
        self.directory = Path(directory)
        self.path = self.directory / 'microphone.pcm'
        self.file = self.path.open('wb', buffering=0)
        self.max_bytes = max_bytes
        self.size = 0

    def append(self, data):
        if len(data) % 2:
            raise ValueError('PCM16 потребує парної кількості байтів.')
        if self.size + len(data) > self.max_bytes:
            raise ValueError('Досягнуто ліміт запису мікрофона. Зупиніть запис і почніть новий.')
        # Unbuffered writes can be short on a shared filesystem. Count only
        # a fully written packet so Stop/snapshot never sees a missing tail.
        remaining = memoryview(data)
        while remaining:
            written = self.file.write(remaining)
            if not written:
                raise RuntimeError('Не вдалося зберегти аудіобуфер мікрофона.')
            remaining = remaining[written:]
        self.size += len(data)

    def snapshot(self, size):
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.wav', delete=False) as tmp:
            path = tmp.name
        try:
            with self.path.open('rb') as source, wave.open(path, 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(SAMPLE_RATE)
                remaining = size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError('Не вдалося прочитати повний запис мікрофона.')
                    out.writeframesraw(chunk)
                    remaining -= len(chunk)
            return path
        except BaseException:
            Path(path).unlink(missing_ok=True)
            raise

    def close(self):
        self.file.close()


async def run_microphone_session(websocket, engine, model, directory, max_bytes, language=None,
                                 diarize=True, translate=False, target_language="Ukrainian",
                                 translation_model=None, translation_context='', visual=None):
    if translate:
        from .translation import translation_options
        translation_model, translation_context = translation_options(translation_model, translation_context)
    recording = PCMRecording(directory, max_bytes)
    boundary = SpeechBoundary(SAMPLE_RATE, TRANSLATION_SECONDS, TRANSLATION_SILENCE_SECONDS) if translate else None
    changed = asyncio.Event()
    stopping = False
    client_gone = False
    receiver_error = None
    processor = None

    async def receive():
        nonlocal stopping, client_gone, receiver_error
        try:
            while not stopping:
                message = await websocket.receive()
                if message.get('type') == 'websocket.disconnect':
                    client_gone = True
                    return
                if message.get('bytes') is not None:
                    data = message['bytes']
                    if len(data) > 1024 * 1024:
                        raise ValueError('Завеликий пакет аудіо; максимум 1 МіБ.')
                    recording.append(data)
                    if boundary is not None:
                        boundary.feed(data)
                    changed.set()
                elif message.get('text'):
                    value = json.loads(message['text'])
                    if isinstance(value, dict) and value.get('type') == 'visual.frame' and visual is not None:
                        # Bound the frame timeline to audio already received in this session.
                        frame = value.get('frame')
                        if not isinstance(frame, dict):
                            raise ValueError('Некоректний кадр.')
                        if frame.get('time') is not None and frame['time'] > recording.size / BYTES_PER_SECOND + 2:
                            raise ValueError('Час кадру поза аудіозаписом.')
                        visual.enqueue(frame)
                        changed.set()
                    if isinstance(value, dict) and value.get('type') == 'visual.labels' and visual is not None:
                        visual.update_labels(value.get('labels'))
                        changed.set()
                    if isinstance(value, dict) and value.get('type') == 'stop':
                        expected = value.get('audio_bytes')
                        if expected is not None and (type(expected) is not int or expected != recording.size):
                            raise ValueError('Отримано не весь аудіобуфер мікрофона; фіналізацію зупинено.')
                        stopping = True
                        await websocket.send_json({'type': 'status',
                            'detail': 'Увесь аудіобуфер отримано. Завершую обробку запису…',
                            'audio_bytes': recording.size,
                            'audio_seconds': recording.size / BYTES_PER_SECOND})
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            receiver_error = str(exc)
        finally:
            changed.set()
            if (client_gone or receiver_error) and processor is not None:
                processor.cancel()

    async def process():
        last_size = 0
        last_visual_version = -1
        result = None
        while True:
            await changed.wait()
            changed.clear()
            due = boundary.due(recording.size) if boundary is not None else (
                'interval' if recording.size - last_size >= PREVIEW_SECONDS * BYTES_PER_SECOND else None)
            if not stopping and not due:
                continue
            if recording.size == 0 and stopping:
                await websocket.send_json({'type': 'transcript.done', 'text': '', 'segments': [],
                    'provisional': False, 'audio_bytes': 0, 'audio_seconds': 0})
                return
            size = recording.size
            if size != last_size or (visual is not None and visual.version != last_visual_version):
                if boundary is not None:
                    boundary.snapshot_started(size)
                if stopping:
                    if translate and diarize:
                        detail = f'Мікрофон вимкнено. Узгоджую мовців та фіналізую переклад ({target_language}) у всьому записі…'
                    elif translate:
                        detail = f'Мікрофон вимкнено. Фіналізую переклад ({target_language}) у всьому записі…'
                    elif diarize:
                        detail = 'Мікрофон вимкнено. Узгоджую мовців у всьому записі…'
                    else:
                        detail = 'Мікрофон вимкнено. Фіналізую транскрипцію…'
                else:
                    if translate and diarize:
                        detail = f'Діаризація та переклад ({target_language}): аналізую запис; текст оновлюється.'
                    elif translate:
                        detail = f'Переклад ({target_language}): аналізую запис; текст оновлюється.'
                    elif diarize:
                        detail = 'Діаризація: аналізую запис; попередні позначки мовців можуть змінитися.'
                    else:
                        detail = 'Аналізую запис…'

                await websocket.send_json({'type': 'status', 'detail': detail,
                    'trigger': 'stop' if stopping else due, 'audio_seconds': size / BYTES_PER_SECOND})
                snapshot = asyncio.create_task(asyncio.to_thread(recording.snapshot, size))
                try:
                    path = await asyncio.shield(snapshot)
                except asyncio.CancelledError:
                    path = await snapshot
                    Path(path).unlink(missing_ok=True)
                    raise
                events = transcription_events(engine, path, model, DEFAULT_MAX_TOKENS, language, None,
                                              diarize=diarize, translate=translate, target_language=target_language,
                                              translation_model=translation_model, translation_context=translation_context,
                                              **({'visual': visual} if visual is not None else {}))
                text = ''
                sent_at = 0
                result = None
                try:
                    async for event in events:
                        payload = json.loads(event.removeprefix('data: '))
                        if payload['type'] == 'transcript.status':
                            await websocket.send_json({'type': 'status', 'detail': payload['message']})
                        if payload['type'] == 'transcript.visual':
                            await websocket.send_json({'type': 'visual.context', 'visual': payload['visual']})
                        if payload['type'] == 'error':
                            raise RuntimeError(payload['error']['message'])
                        if payload['type'] == 'transcript.text.delta':
                            text += payload['delta']
                            if time.monotonic() - sent_at >= 0.2:
                                await websocket.send_json({'type': 'transcript.partial', 'text': text,
                                    'provisional': True, 'audio_seconds': size / BYTES_PER_SECOND})
                                sent_at = time.monotonic()
                        elif payload['type'] == 'transcript.text.done':
                            result = payload
                finally:
                    await events.aclose()
                if result is None:
                    raise RuntimeError('Обробка завершилася без повного результату.')
                last_size = size
                if visual is not None:
                    last_visual_version = result.get('visual', {}).get('processed_version', -1)
            if (stopping and recording.size == last_size and not (visual is not None and
                    (visual.pending or visual.version != last_visual_version))):
                await websocket.send_json({'type': 'transcript.done', 'text': result['text'],
                    'segments': result.get('segments', []), 'provisional': False,
                    'source_text': result.get('source_text', ''),
                    'source_segments': result.get('source_segments', []),
                    'diarize': diarize, 'translate': translate,
                    'target_language': target_language if translate else None,
                    'translation_model': translation_model if translate else None,
                    'audio_bytes': last_size, 'audio_seconds': last_size / BYTES_PER_SECOND,
                    'speaker_scope': 'recording' if diarize else None, 'timestamps': result.get('timestamps', False),
                    **({k: result[k] for k in ('timestamp_type', 'timing_window_seconds') if k in result}),
                    **({'visual': result.get('visual')} if visual is not None else {})})
                return
            await websocket.send_json({'type': 'transcript.partial', 'text': result['text'],
                'provisional': True, 'audio_seconds': last_size / BYTES_PER_SECOND})
            if translate and diarize:
                ready_msg = f'Діаризація та переклад ({target_language}): слухаю. Остаточний результат — після Стоп.'
            elif translate:
                ready_msg = f'Переклад ({target_language}): слухаю. Остаточний результат — після Стоп.'
            elif diarize:
                ready_msg = 'Діаризація: слухаю. Позначки мовців попередні; після Стоп — остаточний результат.'
            else:
                ready_msg = 'Слухаю…'
            await websocket.send_json({'type': 'status', 'detail': ready_msg})
            if stopping or (boundary.due(recording.size) if boundary is not None else
                    recording.size - last_size >= PREVIEW_SECONDS * BYTES_PER_SECOND):
                changed.set()

    activity = engine._begin_activity('transcribing', detail='Microphone transcription')
    receiver = None
    try:
        processor = asyncio.create_task(process())
        receiver = asyncio.create_task(receive())
        await websocket.send_json({'type': 'ready', 'diarization': diarize, 'translation': translate,
            'target_language': target_language if translate else None,
            'sample_rate': SAMPLE_RATE,
            'preview_seconds': TRANSLATION_SECONDS if translate else PREVIEW_SECONDS,
            'silence_seconds': TRANSLATION_SILENCE_SECONDS if translate else None})
        try:
            await processor
        except asyncio.CancelledError:
            if not client_gone and not receiver_error:
                raise
        if receiver_error:
            raise RuntimeError(receiver_error)
    finally:
        if receiver is not None:
            receiver.cancel()
        if processor is not None:
            processor.cancel()
        await asyncio.gather(*(task for task in (receiver, processor) if task is not None), return_exceptions=True)
        recording.close()
        await engine._finish_activity(activity)
