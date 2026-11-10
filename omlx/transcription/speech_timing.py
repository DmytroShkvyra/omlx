# SPDX-License-Identifier: Apache-2.0
# Fork modification: migrated meeting transcription features from the local prototype.
"""Conservative file-time bounds for SAA turns when the MLX CTC head is absent.

Independent ASR windows anchor text only. Original full-recording speaker IDs
and words remain authoritative; window boundaries are estimates, not word times.
"""
import asyncio
import json
from difflib import SequenceMatcher
from pathlib import Path
import tempfile
import wave

from .visual_context import command_bytes, words

WINDOW_SECONDS = 15


def align_window_bounds(turns, windows, duration):
    reference = []
    ranges = []
    for turn in turns:
        start = len(reference)
        reference.extend(words(turn['text']))
        ranges.append((start, len(reference)))
    recognized, positions = [], []
    for window in windows:
        tokens = words(window['text'])
        recognized.extend(tokens)
        positions.extend([(window['start'], window['end'])] * len(tokens))
    anchors = {}
    for block in SequenceMatcher(None, reference, recognized, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            anchors[block.a + i] = positions[block.b + i]
    result = []
    for turn, (a, b) in zip(turns, ranges):
        matched = [anchors[i] for i in range(a, b) if i in anchors]
        coverage = len(matched) / max(1, b - a)
        if matched and coverage >= .5:
            start, end = min(x[0] for x in matched), max(x[1] for x in matched)
            kind = 'windowed_asr'
        else:
            # Bracket uncertain turns with neighboring anchors; never distribute
            # speech uniformly across silence based merely on character counts.
            before = [i for i in anchors if i < a]
            after = [i for i in anchors if i >= b]
            if not before and not after:
                result.append({**turn, 'start': None, 'end': None,
                               'timing_method': 'unavailable', 'timing_coverage': round(coverage, 3)})
                continue
            start = anchors[max(before)][0] if before else 0
            end = anchors[min(after)][1] if after else duration
            kind = 'neighboring_windows'
        result.append({**turn, 'start': round(max(0, start), 3),
                       'end': round(min(duration, end), 3), 'timing_method': kind,
                       'timing_coverage': round(coverage, 3), 'timestamps_approximate': True})
    return result


async def estimate_turn_intervals(engine, path, turns, cache=None):
    if not turns:
        return turns
    cache = cache if cache is not None else {}
    with tempfile.TemporaryDirectory(dir=Path(path).parent) as directory:
        pcm = Path(directory) / 'timing.wav'
        await command_bytes('ffmpeg', '-v', 'error', '-i', str(path), '-vn',
                            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(pcm), timeout=120)
        with wave.open(str(pcm), 'rb') as source:
            rate, total = source.getframerate(), source.getnframes()
            duration = total / rate
            windows = []
            for start in range(0, total, WINDOW_SECONDS * rate):
                end = min(total, start + WINDOW_SECONDS * rate)
                key = (start, end)
                if key not in cache:
                    clip = Path(directory) / 'window.wav'
                    source.setpos(start)
                    with wave.open(str(clip), 'wb') as out:
                        out.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
                        out.writeframes(source.readframes(end - start))
                    iterator = engine.transcribe_stream(str(clip), task='asr', max_tokens=1024)
                    raw, tokens = [], 0
                    try:
                        async for chunk in iterator:
                            raw.append(chunk.get('text') or '')
                            tokens = max(tokens, int(chunk.get('generation_tokens') or 0))
                    finally:
                        await iterator.aclose()
                    cache[key] = ''.join(raw) if tokens < 1024 else ''
                windows.append({'start': start / rate, 'end': end / rate, 'text': cache[key]})
                await asyncio.sleep(0)
    return align_window_bounds(turns, windows, duration)


async def resolve_turn_intervals(engine, path, turns, cache=None, native_payload=None):
    from .native_timestamps import native_turn_intervals
    native = native_turn_intervals(turns, native_payload)
    if any(t is not None for t in native):
        # Validate the native clock against the input file, without an ASR pass.
        probe = await command_bytes('ffprobe', '-v', 'error', '-show_entries',
                                    'format=duration', '-of', 'json', str(path))
        duration = float(json.loads(probe)['format']['duration'])
        native = [t if t is not None and t['end'] <= duration + .1 else None for t in native]
    if turns and all(t is not None for t in native):
        return native
    approximate = await estimate_turn_intervals(engine, path, turns, cache)
    return [n if n is not None else a for n,a in zip(native,approximate)]


def timing_metadata(turns):
    methods = {t.get('timing_method') for t in turns if t.get('start') is not None}
    if methods == {'native_ctc'}:
        kind = 'native_ctc'
    elif 'native_ctc' in methods:
        kind = 'mixed_native_and_approximate'
    elif methods:
        kind = 'approximate_window_intervals'
    else:
        kind = 'unavailable'
    return {'timestamp_type':kind, 'timing_window_seconds':15 if methods - {'native_ctc'} else None}


def interval_text(turns, labels=None):
    from .visual_context import display_label, frame_timestamp
    labels = labels or {}
    lines = []
    for turn in turns:
        speaker = turn.get('speaker')
        prefix = ''
        if speaker is not None:
            label = display_label(labels.get(speaker, ''))
            prefix = '**' + (label or f'Мовець {speaker + 1}') + ':** '
        start, end = turn.get('start'), turn.get('end')
        if start is not None and end is not None:
            mark = '' if turn.get('timing_method') == 'native_ctc' else '≈'
            prefix += f'[{mark}{frame_timestamp(start)}–{frame_timestamp(end)}] '
        lines.append(prefix + turn['text'])
    return '\n\n'.join(lines)
