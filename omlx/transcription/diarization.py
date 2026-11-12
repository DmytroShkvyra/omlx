# SPDX-License-Identifier: Apache-2.0
# Fork modification: migrated meeting transcription features from the local prototype.
"""Native whole-recording Nemotron speaker attribution and translation in the isolated server.

The perception encoder keeps its own Sortformer state across internal windows.
Do not run SAA through the independent 30-second ASR calls in upload_compat.
Speaker IDs are local to one file. This MLX conversion lacks the native CTC
timestamp head; optional visual processing estimates intervals with ASR windows.
"""
import json
from contextlib import nullcontext
from pathlib import Path
import re
import tempfile

SPEAKER = re.compile(r'<spk:([0-7])>')
DEFAULT_MAX_TOKENS = 32768
# Fork fix (hallucination loop): live mic and diarized-upload transcription
# re-decode the whole recording on every boundary tick, with no built-in
# stop condition besides max_tokens (32768). A trailing silent/noisy tail
# (very common right as the user pauses or clicks Stop) can make Nemotron
# lock onto a short phrase and repeat it until that budget is exhausted,
# flooding the transcript with garbage and wasting the whole generation.
# Detect a short unit (<=80 chars) repeated 6+ times in a row near the end
# of the text seen so far and treat it as a degenerate loop, not speech.
_REPETITION_LOOP = re.compile(r'(.{1,80}?)\1{5,}\Z', re.DOTALL)
_REPETITION_WINDOW = 4000


def _trim_repetition_loop(text):
    # Return (trimmed_text, True) if text ends in a repeated loop.
    tail = text[-_REPETITION_WINDOW:]
    match = _REPETITION_LOOP.search(tail)
    if match is None:
        return text, False
    cut = len(tail) - match.start()
    return text[:-cut].rstrip(), True
ORIGINAL_LANGUAGE_PROMPT = (
    'Transcribe the audio verbatim in the original spoken language. '
    'Keep the same language as spoken and do not translate into English or any other language. '
    'Preserve language switches, named entities, numbers and technical terms. '
    'Use punctuation and capitalization. Include all spoken words and verbal fillers.'
)
SPEAKER_ORIGINAL_LANGUAGE_PROMPT = ORIGINAL_LANGUAGE_PROMPT + (
    ' Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> <spk:4> <spk:5> <spk:6> <spk:7> '
    'on the transcription. Keep each speaker ID consistent throughout the recording.'
)


def resolve_nemotron_params(diarize: bool = False, translate: bool = False, target_language: str = "Ukrainian"):
    # Translation is a separate text pass. Never ask the audio model to change
    # language and discover speakers in the same generation.
    return (SPEAKER_ORIGINAL_LANGUAGE_PROMPT if diarize else ORIGINAL_LANGUAGE_PROMPT,
            "saa" if diarize else "asr", None)


def speaker_segments(text):
    matches = list(SPEAKER.finditer(text))
    return [dict(speaker=int(match[1]), text=text[match.end():
        matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip())
        for i, match in enumerate(matches)]


def readable_text(text):
    return SPEAKER.sub(lambda match: '\n\n**Мовець ' + str(int(match[1]) + 1) + ':** ', text).strip()


class SpeakerTextStream:
    """Convert tags split across arbitrary token boundaries without leaking HTML."""
    def __init__(self):
        self.pending = ''

    def feed(self, text, final=False):
        self.pending += text
        end = len(self.pending)
        # Hold an incomplete speaker tag until its next token arrives.
        if not final:
            start = self.pending.rfind('<')
            if start >= 0:
                tail = self.pending[start:]
                if '<spk:'.startswith(tail) or re.fullmatch(r'<spk:[0-7]?', tail):
                    end = start
        output, self.pending = self.pending[:end], self.pending[end:]
        return SPEAKER.sub(lambda match: '\n\n**Мовець ' + str(int(match[1]) + 1) + ':** ', output)


async def transcription_events(engine, path, model, max_tokens, language=None, prompt=None,
                               diarize=True, translate=False, target_language='Ukrainian',
                               translation_model=None, translation_context='', visual=None, video_path=None):
    from omlx.api import audio_routes as audio
    raw = []
    formatter = SpeakerTextStream()
    generation_tokens = prompt_tokens = 0
    source_text = ''
    source_segments = []

    nemotron_prompt, task, resolved_lang = resolve_nemotron_params(
        diarize=diarize, translate=translate, target_language=target_language
    )
    if prompt and not diarize and not translate:
        # A vocabulary hint must not replace the no-translation task instruction.
        nemotron_prompt = ORIGINAL_LANGUAGE_PROMPT + '\nContextual vocabulary (reference data): ' + json.dumps(prompt)
    effective_language = resolved_lang or language

    kw = {
        'task': task,
        'max_tokens': max_tokens,
        'language': effective_language,
    }
    iterator = None
    native_scope = nullcontext()
    native_capture = None
    # Fork fix (#session-evict): pin the engine for this call's whole span,
    # covering file-upload diarize/translate (one call = the whole file,
    # which can run for minutes) the same way run_microphone_session() pins
    # it for a live mic session.
    begin_diarize = getattr(engine, 'begin_diarize_session', None)
    end_diarize = getattr(engine, 'end_diarize_session', None)
    if callable(begin_diarize):
        begin_diarize()
    try:
        if visual is not None:
            from .native_timestamps import capture_native_timestamps
            native_scope = capture_native_timestamps(engine, path)
        native_capture = native_scope.__enter__()
        iterator = engine.transcribe_stream(path, prompt=nemotron_prompt, **kw)
        if visual is not None:
            # Emit progress before expensive frame processing so SSE starts promptly.
            yield audio._sse_event({'type': 'transcript.status', 'message': 'Аналізую відеокадри та видимі підписи…'})
            if video_path:
                try:
                    await visual.add_video(video_path)
                except Exception as exc:
                    visual.warnings = ['Кадри відео: ' + str(exc)[:240]]
            await visual.drain()
            yield audio._sse_event({'type': 'transcript.visual', 'visual': visual.metadata()})
        if translate:
            from .translation import validate_language, translation_options
            validate_language(target_language)
            translation_model, translation_context = translation_options(translation_model, translation_context)
        if translate and diarize:
            msg = 'Крок 1/2: розпізнаю оригінал і визначаю мовців…'
        elif translate:
            msg = 'Крок 1/2: розпізнаю оригінал…'
        elif diarize:
            msg = 'Аналізую весь запис і голоси. Перший текст з’явиться після кодування аудіо.'
        else:
            msg = 'Аналізую аудіо…'

        yield audio._sse_event({'type': 'transcript.status', 'message': msg})
        hallucination_truncated = False
        async for chunk in iterator:
            if native_capture is not None:
                native_capture.observe(chunk)
            raw.append(chunk.get('text') or '')
            generation_tokens = max(generation_tokens, int(chunk.get('generation_tokens') or 0))
            prompt_tokens = max(prompt_tokens, int(chunk.get('prompt_tokens') or 0))
            delta = formatter.feed(raw[-1]) if diarize else raw[-1]
            if delta and not translate:
                yield audio._sse_event({'type': 'transcript.text.delta', 'delta': delta})
            if _trim_repetition_loop(''.join(raw))[1]:
                hallucination_truncated = True
                yield audio._sse_event({'type': 'transcript.status',
                    'message': 'Виявлено зациклений фрагмент (ймовірно, тиша чи шум); зупиняю цю розшифровку раніше.'})
                break
        if diarize and not translate:
            tail = formatter.feed('', final=True)
            if tail:
                yield audio._sse_event({'type': 'transcript.text.delta', 'delta': tail})
        if not hallucination_truncated and generation_tokens >= max_tokens:
            raise RuntimeError('Досягнуто ліміт тексту; результат неповний. Повторіть з більшим max_tokens.')
        await iterator.aclose()
        text = ''.join(raw)
        if hallucination_truncated:
            text, _ = _trim_repetition_loop(text)
        source_text = readable_text(text) if diarize else text.strip()
        source_segments = speaker_segments(text) if diarize else []
        if diarize and text.strip() and (not source_segments or text[:SPEAKER.search(text).start()].strip()):
            raise RuntimeError('Nemotron не повернув повну розмітку мовців. Повторіть діаризацію.')
        segments = source_segments
        visual_revision = None
        if visual is not None:
            from .speech_timing import resolve_turn_intervals
            yield audio._sse_event({'type': 'transcript.status',
                'message': 'Використовую нативні CTC-таймінги, якщо доступні; інакше оцінюю інтервали аудіовікон…'})
            timing_turns = source_segments if diarize else ([{'text': text.strip()}] if text.strip() else [])
            if (native_capture is not None and isinstance(native_capture.payload, dict)
                    and native_capture.payload.get('error') is not None):
                visual.warnings.append('CTC-вирівнювання повернуло помилку; використовую приблизні інтервали.')
            try:
                timing_turns = await resolve_turn_intervals(engine, path, timing_turns, visual.timing_cache,
                    native_capture.payload if native_capture is not None else None)
            except Exception as exc:
                visual.warnings.append('Часові інтервали недоступні: ' + str(exc)[:180])
                timing_turns = [{**turn, 'start': None, 'end': None, 'timing_method': 'unavailable'} for turn in timing_turns]
            timing_turns = [{**turn, 'visual_context_times': [o['time'] for o in visual.observations_for_interval(
                turn.get('start'), turn.get('end')) if o['time'] is not None]} for turn in timing_turns]
            source_segments = timing_turns
            segments = source_segments
            visual.context_scoped = True
            if diarize:
                yield audio._sse_event({'type': 'transcript.status', 'message': 'Зіставляю видимих учасників із репліками…'})
                await visual.associate(engine, path, source_segments)
            yield audio._sse_event({'type': 'transcript.visual', 'visual': visual.metadata()})
            visual_revision = visual.version
        if translate:
            from .translation import translate_turns
            yield audio._sse_event({'type': 'transcript.status',
                'message': f'Крок 2/2: перекладаю на {target_language} ({translation_model}); зберігаю репліки та перевіряю мову…'})
            turns = source_segments if diarize else ([{'text': text.strip()}] if text.strip() else [])
            options = {'model': translation_model, 'context': translation_context}
            if visual is not None:
                options['turn_contexts'] = [visual.context_for_interval(t.get('start'), t.get('end')) for t in timing_turns]
                if not diarize:
                    turns = timing_turns
            translated = await translate_turns(turns, target_language, **options)
            if diarize:
                segments = translated
                text = ''.join(f"<spk:{turn['speaker']}>{turn['text']}" for turn in translated)
            else:
                text = '\n'.join(turn['text'] for turn in translated)
                if visual is not None:
                    segments = translated
            if text:
                yield audio._sse_event({'type': 'transcript.text.delta',
                    'delta': readable_text(text) if diarize else text})
        final_text = readable_text(text) if diarize else text.strip()
        if visual is not None:
            from .speech_timing import interval_text, timing_metadata
            final_text = interval_text(segments, visual.labels)
            segments = [{**s, 'speaker_label': visual.labels.get(s.get('speaker'))} for s in segments]
        yield audio._sse_event({
            'type': 'transcript.text.done',
            'text': final_text,
            'raw_text': text,
            'segments': segments,
            'task': 'translate' if translate else task,
            'source_task': task,
            'source_text': source_text,
            'source_segments': source_segments,
            'diarize': diarize,
            'translate': translate,
            'target_language': target_language if translate else None,
            'translation_model': translation_model if translate else None,
            'speaker_scope': 'recording' if diarize else None,
            'timestamps': bool(visual is not None and any(s.get('start') is not None for s in segments)),
            **(timing_metadata(segments) if visual is not None else {}),
            **({'visual': {**visual.metadata(), 'processed_version': visual_revision}} if visual is not None else {}),
            'usage': {'input_tokens': prompt_tokens, 'output_tokens': generation_tokens,
                      'total_tokens': prompt_tokens + generation_tokens}
        })
        audio._record_audio_request(model)
    except Exception as exc:
        message = str(exc)
        if translate and source_text:
            message += '\n\nОригінальна транскрипція (переклад не завершено):\n\n' + source_text
        yield audio._sse_event({'type': 'error', 'error': {'message': message}})
    finally:
        try:
            if iterator is not None:
                await iterator.aclose()
        finally:
            native_scope.__exit__(None, None, None)
            audio._cleanup_tempfile(path)
            if callable(end_diarize):
                end_diarize()
