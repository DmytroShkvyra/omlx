# SPDX-License-Identifier: Apache-2.0
# Fork modification: migrated meeting transcription features from the local prototype.
"""Preserve native CTC metadata that oMLX's text-stream adapter otherwise drops."""
from contextlib import contextmanager
from difflib import SequenceMatcher
from functools import wraps
import inspect
import math
from pathlib import Path

from .visual_context import words

_CAPTURES = {}


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class NativeCapture:
    def __init__(self):
        self.payload = None

    def observe(self, value):
        payload = _get(value, 'ctc_timestamps')
        if payload is not None:
            if hasattr(payload, 'model_dump'):
                payload = payload.model_dump()
            self.payload = payload


def install_capture(model_class):
    original = getattr(model_class, 'generate', None)
    if not inspect.isfunction(original) or getattr(original, '_preserve_ctc', False):
        return
    parameters = inspect.signature(original).parameters

    @wraps(original)
    def generate(self, audio, *args, **kwargs):
        key = (id(self), str(Path(audio).resolve())) if isinstance(audio, (str, Path)) else None
        capture = _CAPTURES.get(key)
        if capture is None:
            return original(self, audio, *args, **kwargs)
        # Opt in only when the runtime explicitly declares the capture option;
        # accepting arbitrary **kwargs is not evidence of timestamp support.
        if 'capture_ctc_timestamps' in parameters:
            kwargs['capture_ctc_timestamps'] = True
        result = original(self, audio, *args, **kwargs)
        if isinstance(result, (str, dict)) or not hasattr(result, '__next__'):
            capture.observe(result)
            return result

        def chunks():
            try:
                while True:
                    try:
                        chunk = next(result)
                    except StopIteration as stop:
                        capture.observe(stop.value)
                        return stop.value
                    capture.observe(chunk)
                    yield chunk
            finally:
                close = getattr(result, 'close', None)
                if close:
                    close()
        return chunks()

    generate._preserve_ctc = True
    model_class.generate = generate


@contextmanager
def capture_native_timestamps(engine, path):
    model = getattr(engine, '_model', None)
    capture = NativeCapture()
    if model is None:
        yield capture
        return
    install_capture(type(model))
    key = (id(model), str(Path(path).resolve()))
    _CAPTURES[key] = capture
    try:
        yield capture
    finally:
        _CAPTURES.pop(key, None)


def native_turn_intervals(turns, payload):
    """Use checked native word bounds; retain the original transcript and voice IDs."""
    if not isinstance(payload, dict) or 'error' not in payload or payload['error'] is not None:
        return [None] * len(turns)
    word_values = payload.get('words')
    if not isinstance(word_values, list) or not word_values:
        return [None] * len(turns)
    native_tokens, native_bounds = [], []
    for word in word_values:
        if not isinstance(word, dict):
            return [None] * len(turns)
        start, end = word.get('start'), word.get('end')
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start):
            return [None] * len(turns)
        text = word.get('word', word.get('text'))
        if not isinstance(text, str):
            return [None] * len(turns)
        speaker = word.get('speaker')
        if isinstance(speaker, str):
            tag = speaker.removeprefix('<spk:').removesuffix('>')
            speaker = int(tag) if tag.isdigit() else None
        for token in words(text):
            native_tokens.append(token)
            native_bounds.append((start, end, speaker, word))
    reference, ranges = [], []
    for turn in turns:
        a = len(reference)
        reference.extend(words(turn['text']))
        ranges.append((a, len(reference)))
    matched = {}
    for block in SequenceMatcher(None, reference, native_tokens, autojunk=False).get_matching_blocks():
        for i in range(block.size):
            matched[block.a+i] = native_bounds[block.b+i]
    result = []
    for turn, (a, b) in zip(turns, ranges):
        valid = {i:matched[i] for i in range(a,b) if i in matched and
                 (matched[i][2] is None or turn.get('speaker') is None or matched[i][2] == turn['speaker'])}
        bounds = list(valid.values())
        if not bounds or len(bounds) / max(1,b-a) < .9 or a not in valid or b-1 not in valid:
            result.append(None)
            continue
        if any(bounds[i][0] < bounds[i-1][0] for i in range(1,len(bounds))):
            result.append(None)
            continue
        result.append({**turn, 'start':min(v[0] for v in bounds), 'end':max(v[1] for v in bounds),
                       'timing_method':'native_ctc', 'timing_coverage':len(bounds)/max(1,b-a),
                       'timestamps_approximate':False,
                       'words':list({id(v[3]):v[3] for v in bounds}.values())})
    return result
