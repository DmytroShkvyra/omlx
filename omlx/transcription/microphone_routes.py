# SPDX-License-Identifier: Apache-2.0
"""Fork modification: authenticated meeting API routes."""
import json
from pathlib import Path
import tempfile
import asyncio
import contextlib
from .microphone_diarization import run_microphone_session
from .runtime import is_nemotron_model, runtime_directory

from fastapi import WebSocket
from omlx.api import audio_routes as audio
from omlx.engine.stt import STTEngine

path = '/v1/audio/transcriptions/realtime/diarize'
@audio.realtime_router.websocket(path)
async def microphone_diarization(websocket: WebSocket):
    await websocket.accept()
    try:
        start = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        if not isinstance(start, dict) or start.get('type') != 'start':
            raise ValueError('Першим повідомленням має бути start.')
        if not audio._verify_ws_api_key(websocket, start.get('api_key')):
            raise ValueError('Invalid API key')
        model = start.get('model')
        if not isinstance(model, str) or not model:
            raise ValueError('Виберіть модель Nemotron.')
        language = start.get('language')
        if language is not None and not isinstance(language, str):
            raise ValueError('language має бути рядком.')
        diarize = bool(start.get('diarize', True))
        translate = bool(start.get('translate', False))
        target_language = str(start.get('target_language', 'Ukrainian') or 'Ukrainian')
        translation_model = start.get('translation_model')
        translation_context = start.get('translation_context', '')
        visual = None
        if start.get('visual_enabled') is True:
            from .visual_context import VisualContext
            visual = VisualContext(start.get('vision_model', ''), start.get('visual_frames', []),
                                   start.get('speaker_labels', {}), time_basis='audio_session',
                                   frame_selection='changes' if start.get('frame_selection') == 'changes' else 'uniform')
        if translate:
            from .translation import translation_options
            translation_model, translation_context = translation_options(translation_model, translation_context)
        resolved = audio._resolve_model(model)
        engine = await audio._get_engine_pool().get_engine(resolved)
        if not isinstance(engine, STTEngine) or not is_nemotron_model(engine._model):
            raise ValueError('Діаризація та переклад доступні лише для Nemotron Transcribe.')
        root = runtime_directory('microphone')
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            await run_microphone_session(websocket, engine, resolved, directory,
                audio._max_audio_upload_bytes(), language=language,
                diarize=diarize, translate=translate, target_language=target_language,
                translation_model=translation_model, translation_context=translation_context, visual=visual)
        with contextlib.suppress(Exception):
            await websocket.close(code=1000)
    except Exception as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({'type': 'error', 'detail': str(exc)})
            await websocket.close(code=1011)
