# SPDX-License-Identifier: Apache-2.0
"""Fork modification: authenticated meeting API routes."""
import json
from pathlib import Path
import tempfile
from .diarization import DEFAULT_MAX_TOKENS, transcription_events
from .runtime import is_nemotron_model, runtime_directory

# Register before omlx.server imports and mounts the audio router, so the
# normal inference authentication dependency also protects this endpoint.
from fastapi import Body, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from omlx.api import audio_routes as audio
from omlx.engine.stt import STTEngine
from omlx.exceptions import ModelNotFoundError

endpoint = '/v1/audio/transcriptions/diarize'
@audio.router.get('/v1/audio/translation/models')
async def translation_models():
    from .translation import available_translation_models
    try:
        return await available_translation_models()
    except Exception as exc:
        raise HTTPException(502, detail='Не вдалося отримати моделі перекладу з oMLX.') from exc

@audio.router.get('/v1/audio/visual/models')
async def visual_models():
    from .visual_context import vision_models
    try:
        return await vision_models()
    except Exception as exc:
        raise HTTPException(502, detail='Не вдалося отримати відеомоделі з oMLX.') from exc

@audio.router.post('/v1/audio/visual/change')
async def visual_change(payload: dict = Body(...)):
    from .scene_changes import analyze_preview_change
    try:
        return await analyze_preview_change(payload)
    except ValueError as exc:
        raise HTTPException(400,detail=str(exc)) from exc

@audio.router.post(endpoint)
async def diarize(file: UploadFile = File(...), model: str = Form(...),
                  stream: bool = Form(False), language: str = Form(None),
                  prompt: str = Form(None),
                  diarize: str = Form("true"),
                  translate: str = Form("false"),
                  target_language: str = Form("Ukrainian"),
                  translation_model: str = Form('', max_length=512),
                  translation_context: str = Form('', max_length=6000),
                  visual_enabled: bool = Form(False),
                  vision_model: str = Form('', max_length=512),
                  frame_selection: str = Form('uniform', max_length=32),
                  visual_frames: str = Form('[]', max_length=30_000_000),
                  visual_images: list[UploadFile] = File(None),
                  speaker_labels: str = Form('{}', max_length=4000),
                  max_tokens: int = Form(DEFAULT_MAX_TOKENS, gt=0, le=65536)):
    resolved = audio._resolve_model(model)
    try:
        engine = await audio._get_engine_pool().get_engine(resolved)
    except ModelNotFoundError as exc:
        raise HTTPException(404, detail=str(exc)) from exc
    if not isinstance(engine, STTEngine) or not is_nemotron_model(engine._model):
        raise HTTPException(400, detail='Діаризація доступна лише для Nemotron Transcribe.')
    suffix = Path(file.filename or 'audio.wav').suffix or '.wav'
    is_video = suffix.lower() in audio._VIDEO_CONTAINERS
    if is_video and not visual_enabled:
        suffix = '.m4a'
    uploads = runtime_directory('uploads')
    uploads.mkdir(parents=True, exist_ok=True)
    path = None
    is_diarize = str(diarize).lower() in ("true", "1", "yes")
    is_translate = str(translate).lower() in ("true", "1", "yes")
    target_lang = (target_language or "Ukrainian").strip()
    visual = None
    try:
        if visual_enabled:
            from .visual_context import VisualContext
            frames, labels = json.loads(visual_frames), json.loads(speaker_labels)
            if visual_images:
                import base64
                from .visual_context import MAX_IMAGE_BYTES
                if len(visual_images) > 24:
                    raise ValueError('Забагато кадрів.')
                images = []
                for image in visual_images:
                    data = await image.read(MAX_IMAGE_BYTES + 1)
                    if len(data) > MAX_IMAGE_BYTES:
                        raise ValueError('Завеликий JPEG-кадр.')
                    images.append('data:image/jpeg;base64,' + base64.b64encode(data).decode())
                frames = [{**frame, 'images': [images[i] for i in frame['images']
                           if type(i) is int and 0 <= i < len(images)]} for frame in frames]
            if is_video or frames:
                visual = VisualContext(vision_model, frames, labels, frame_selection=frame_selection)
        # Stream to disk rather than allocating another 2 GB upload buffer.
        with tempfile.NamedTemporaryFile(dir=uploads, suffix=suffix, delete=False) as destination:
            path = destination.name
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > audio._max_audio_upload_bytes():
                    raise HTTPException(413, detail='Файл перевищує ліміт аудіозавантаження.')
                destination.write(chunk)
        if not size:
            raise HTTPException(400, detail='Порожній аудіофайл.')
        if stream:
            events = transcription_events(engine, path, resolved, max_tokens, language, prompt,
                                          diarize=is_diarize, translate=is_translate, target_language=target_lang,
                                          translation_model=translation_model, translation_context=translation_context,
                                          visual=visual, video_path=path if visual is not None and is_video else None)
            first = await events.__anext__()
            response = StreamingResponse(audio._stream_with_prefetched_chunk(first, events),
                media_type='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})
            path = None  # The generator owns and cleans up the uploaded file.
            return response
        # Consume the same checked generator for JSON: neither API mode may
        # silently report a token-limited partial transcript as complete.
        result = None
        async for event in transcription_events(engine, path, resolved, max_tokens, language, prompt,
                                               diarize=is_diarize, translate=is_translate, target_language=target_lang,
                                               translation_model=translation_model, translation_context=translation_context,
                                               visual=visual, video_path=path if visual is not None and is_video else None):
            payload = json.loads(event.removeprefix('data: '))
            if payload['type'] == 'error':
                raise HTTPException(500, detail=payload['error']['message'])
            if payload['type'] == 'transcript.text.done':
                result = payload
        if result is None:
            raise HTTPException(500, detail='Обробка завершилася без результату.')
        result.pop('type')
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc
    finally:
        audio._cleanup_tempfile(path)
        await file.close()
        for image in visual_images or []:
            await image.close()
