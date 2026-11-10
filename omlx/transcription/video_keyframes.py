# SPDX-License-Identifier: Apache-2.0
# Fork modification: migrated meeting transcription features from the local prototype.
"""Decode only codec keyframes; keep source-file timestamps and a bounded VLM budget."""
import base64
import json
import math
from pathlib import Path
import tempfile

from .visual_context import command_bytes, image_uri


def select_keyframes(frames, duration, start_time=0, limit=12):
    values = []
    seen = set()
    for index, frame in enumerate(frames):
        if frame.get('key_frame') != 1:
            continue
        try:
            timestamp = float(frame['best_effort_timestamp_time']) - start_time
        except (ValueError, TypeError, KeyError):
            continue
        if not math.isfinite(timestamp) or timestamp < -.001 or timestamp >= duration:
            continue
        timestamp = max(0, timestamp)
        if timestamp in seen:
            continue
        seen.add(timestamp)
        values.append({'time':timestamp, 'decode_index':index})
    # ffprobe -show_frames and the decoder both return display-order frames.
    total = len(values)
    if total > limit:
        indexes = [round(i * (total - 1) / (limit - 1)) for i in range(limit)]
        values = [values[i] for i in indexes]
    return values, total


async def extract_keyframes(path, duration, start_time=0):
    raw = await command_bytes('ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-skip_frame', 'nokey', '-show_frames', '-show_entries',
        'frame=key_frame,best_effort_timestamp_time', '-of', 'json', str(path), timeout=180)
    selected, total = select_keyframes(json.loads(raw).get('frames', []), duration, start_time)
    if not selected:
        return [], total
    # Select decoder frame numbers in a single pass. No seeking/rounding to a
    # neighboring P/B frame and no full scan repeated for every selected image.
    expression = '+'.join(f'eq(n\\,{value["decode_index"]})' for value in selected)
    with tempfile.TemporaryDirectory(dir=Path(path).parent) as directory:
        destination = str(Path(directory) / 'keyframe-%03d.jpg')
        await command_bytes('ffmpeg', '-v', 'error', '-skip_frame', 'nokey', '-i', str(path),
            '-map', '0:v:0', '-an', '-vf',
            'select=' + expression + ',scale=1280:720:force_original_aspect_ratio=decrease',
            '-fps_mode', 'vfr', '-frames:v', str(len(selected)), '-q:v', '4',
            '-start_number', '0', destination, timeout=180)
        files = sorted(Path(directory).glob('keyframe-*.jpg'))
        if len(files) != len(selected):
            raise RuntimeError('Кількість декодованих ключових кадрів не збігається з часовими мітками.')
        samples = []
        for value, file in zip(selected, files):
            uri = image_uri('data:image/jpeg;base64,' + base64.b64encode(file.read_bytes()).decode())
            samples.append({'time':value['time'], 'images':[uri]})
    return samples, total
