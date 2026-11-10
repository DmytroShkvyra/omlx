# SPDX-License-Identifier: Apache-2.0
# Fork modification: migrated meeting transcription features from the local prototype.
"""Cheap visual-change gate: scene cuts, spatial changes and residual optical flow.

OpenCV/NumPy are already bundled with the host oMLX runtime. This detector does
not identify people or understand semantics; it gates expensive VLM requests.
"""
import asyncio
import base64
from collections import deque
from io import BytesIO
import math
from pathlib import Path
import re

WIDTH, HEIGHT, FPS = 320, 180, 2
MAX_EVENTS = 12


def dependencies():
    import cv2
    import numpy as np
    return cv2, np


def preview_image(uri):
    from PIL import Image
    from .visual_context import image_uri
    if not isinstance(uri, str) or len(uri) > 120_000:
        raise ValueError('Завеликий кадр для детектора змін.')
    image_uri(uri)
    raw = base64.b64decode(uri.split(',', 1)[1], validate=True)
    with Image.open(BytesIO(raw)) as image:
        if image.width > 640 or image.height > 360:
            raise ValueError('Детектор приймає прев’ю до 640×360.')
    cv2, np = dependencies()
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError('Не вдалося прочитати прев’ю.')
    return cv2.cvtColor(cv2.resize(decoded, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)


def compare_frames(reference, current):
    cv2, np = dependencies()
    if reference.shape != current.shape or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError('Розміри кадрів детектора не збігаються.')
    height, width = current.shape[:2]
    previous = cv2.GaussianBlur(reference, (3, 3), .6)
    present = cv2.GaussianBlur(current, (3, 3), .6)
    raw_delta = np.max(np.abs(present.astype(np.int16) - previous.astype(np.int16)), axis=2)
    raw_fraction = float(np.mean(raw_delta > 24))
    raw_mean = float(raw_delta.mean())
    old_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    new_gray = cv2.cvtColor(present, cv2.COLOR_RGB2GRAY)
    old_small = cv2.resize(old_gray, (160, 90))
    new_small = cv2.resize(new_gray, (160, 90))
    textured_cells = sum(float(cell.std()) > 8 for row in np.array_split(old_small,4)
                         for cell in np.array_split(row,4,axis=1))

    # Separate a coherent camera translation from local residual changes.
    shift, response = cv2.phaseCorrelate(old_small.astype(np.float32), new_small.astype(np.float32))
    dx, dy = float(shift[0] * width / 160), float(shift[1] * height / 90)
    camera = False
    delta = raw_delta
    if (textured_cells >= 8 and .25 <= response and math.isfinite(dx) and math.isfinite(dy)
            and 1 <= math.hypot(dx, dy) <= min(width, height) * .2):
        moved = cv2.warpAffine(previous, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height),
                               borderMode=cv2.BORDER_REPLICATE)
        aligned = np.max(np.abs(present.astype(np.int16) - moved.astype(np.int16)), axis=2)
        margin = min(min(width, height)//3, int(math.ceil(max(abs(dx), abs(dy)))) + 3)
        interior = aligned[margin:-margin, margin:-margin]
        if interior.size and float(interior.mean()) < raw_mean * .65:
            camera = True
            delta = aligned
            delta[:margin] = 0; delta[-margin:] = 0
            delta[:, :margin] = 0; delta[:, -margin:] = 0
    if not camera:
        dx = dy = 0.0

    hist_old = np.histogram(old_gray, bins=16, range=(0, 256))[0] / old_gray.size
    hist_new = np.histogram(new_gray, bins=16, range=(0, 256))[0] / new_gray.size
    histogram = float(np.abs(hist_old - hist_new).sum() / 2)
    scene = not camera and ((raw_fraction > .5 and raw_mean > 35 and histogram > .35)
                            or (raw_fraction > .80 and raw_mean > 48))

    mask = (delta > 24).astype(np.uint8)
    # Ignore a tiny outer border and merge neighboring text/edge pixels. Small
    # disconnected cursor/clock components do not qualify as meaningful regions.
    mask[:2] = 0; mask[-2:] = 0; mask[:, :2] = 0; mask[:, -2:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    regions = []
    for i in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[i]]
        # Thin but wide caption/name/active-border changes remain eligible.
        substantial = (w >= 12 and h >= 10 and w*h >= 400) or (w >= 28 and h >= 3 and w*h >= 200)
        if area >= 70 and substantial:
            regions.append({'box':[round(x/width,4),round(y/height,4),round(w/width,4),round(h/height,4)],
                            'area':round(area/(width*height),5)})
    regions.sort(key=lambda r:r['area'], reverse=True)

    flow_fraction = 0.0
    if regions and not scene:
        flow = cv2.calcOpticalFlowFarneback(old_small, new_small, None, .5, 3, 15, 3, 5, 1.2, 0)
        backwards = cv2.calcOpticalFlowFarneback(new_small, old_small, None, .5, 3, 15, 3, 5, 1.2, 0)
        yy, xx = np.mgrid[:90,:160].astype(np.float32)
        reverse_at_target = cv2.remap(backwards,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        consistent = np.linalg.norm(flow+reverse_at_target,axis=2) < 1
        textured = np.hypot(cv2.Sobel(old_small,cv2.CV_32F,1,0),cv2.Sobel(old_small,cv2.CV_32F,0,1)) > 10
        flow[:, :, 0] -= dx * 160 / width
        flow[:, :, 1] -= dy * 90 / height
        flow_fraction = float(np.mean((np.linalg.norm(flow, axis=2) > 1.5) & consistent & textured))
    if scene:
        reason = 'scene_change'
    elif regions:
        dense_region = any(r['box'][3] > .08 and r['area']/max(1e-6,r['box'][2]*r['box'][3]) > .15 for r in regions)
        motion = flow_fraction > .01 and raw_fraction < .25 and histogram < .15 and dense_region
        reason = 'object_motion' if motion else 'local_change'
    elif camera and math.hypot(dx,dy) >= 6:
        reason = 'camera_motion'
    else:
        reason = None
    score = min(1., raw_fraction + histogram + sum(r['area'] for r in regions[:4])*3 + flow_fraction)
    return {'changed':reason is not None,'reason':reason,'score':round(score,4),
            'changed_fraction':round(raw_fraction,5),'histogram_distance':round(histogram,4),
            'motion_fraction':round(flow_fraction,5),'camera_shift':[round(dx,2),round(dy,2)],
            'regions':regions[:6]}


class ChangeGate:
    def __init__(self):
        self.reference = None
        self.last_time = -math.inf
        self.sampled = 0

    def consider(self, frame, timestamp):
        self.sampled += 1
        if self.reference is None:
            decision = {'changed':True,'reason':'initial','score':1.,'regions':[]}
        else:
            decision = compare_frames(self.reference, frame)
            interval = 10 if decision['reason'] in ('object_motion','camera_motion') else 2
            if not decision['changed'] or timestamp-self.last_time < interval:
                return None
        self.reference = frame.copy()
        self.last_time = timestamp
        return {'time':float(timestamp),**decision}


def choose_events(events, duration, limit=MAX_EVENTS):
    if len(events) <= limit:
        return events
    priorities = {'initial':5,'scene_change':4,'local_change':3,'object_motion':2,'camera_motion':1}
    rank = lambda event:(priorities.get(event['reason'],0),event['score'])
    chosen = [events[0]]
    for i in range(limit-1):
        lo, hi = duration*i/(limit-1), duration*(i+1)/(limit-1)
        bucket = [e for e in events[1:] if lo <= e['time'] < hi]
        if bucket:
            chosen.append(max(bucket,key=rank))
    if len(chosen) < limit:
        extra = [e for e in events if e not in chosen]
        extra.sort(key=rank,reverse=True)
        chosen.extend(extra[:limit-len(chosen)])
    return sorted(chosen,key=lambda event:event['time'])


async def scan_video_changes(path):
    cv2, np = dependencies()
    process = await asyncio.create_subprocess_exec('ffmpeg','-hide_banner','-loglevel','info','-nostats',
        '-i',str(path),'-map','0:v:0','-an','-vf',
        f'fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=rgb24,showinfo',
        '-pix_fmt','rgb24','-f','rawvideo','pipe:1',
        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    times = asyncio.Queue(maxsize=8)
    errors = deque(maxlen=8)
    async def read_metadata():
        while line := await process.stderr.readline():
            text = line.decode(errors='replace')
            match = re.search(r'\bpts_time:([+\-\d.eE]+)',text)
            if match:
                await times.put(float(match[1]))
            elif 'error' in text.lower():
                errors.append(text.strip())
        await times.put(None)
    reader = asyncio.create_task(read_metadata())
    gate, events = ChangeGate(), []
    try:
        while True:
            try:
                raw = await process.stdout.readexactly(WIDTH*HEIGHT*3)
            except asyncio.IncompleteReadError as exc:
                if exc.partial:
                    raise RuntimeError('Неповний кадр у детекторі змін.')
                break
            timestamp = await times.get()
            if timestamp is None or not math.isfinite(timestamp):
                raise RuntimeError('FFmpeg не повернув час кадру.')
            frame = np.frombuffer(raw,np.uint8).reshape(HEIGHT,WIDTH,3)
            event = await asyncio.to_thread(gate.consider,frame,timestamp)
            if event is not None:
                events.append(event)
        await process.wait()
        if process.returncode:
            raise RuntimeError('Не вдалося сканувати відео: '+' '.join(errors)[-300:])
        return events, gate.sampled
    finally:
        reader.cancel()
        if process.returncode is None:
            process.kill()
        await asyncio.gather(reader,process.stdout.read(),process.wait(),return_exceptions=True)


async def extract_change_frames(path, duration):
    from .visual_context import command_bytes, image_uri
    events, sampled = await asyncio.wait_for(scan_video_changes(path), timeout=900)
    selected = choose_events(events,duration)
    samples = []
    for event in selected:
        timestamp = max(0,min(duration-.05,event['time']))
        images = []
        for time in (timestamp,min(duration-.025,timestamp+.6)):
            jpeg = await command_bytes('ffmpeg','-v','error','-ss',str(time),'-i',str(path),
                '-frames:v','1','-vf','scale=1280:720:force_original_aspect_ratio=decrease',
                '-f','image2pipe','-vcodec','mjpeg','-q:v','4','pipe:1')
            images.append(image_uri('data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()))
        samples.append({'time':timestamp,'images':images,'detection':event})
    return samples, {'mode':'changes','sampling_fps':FPS,'sampled_frames':sampled,
                     'candidates':len(events),'selected_times':[s['time'] for s in samples],
                     'events':selected,'limit':MAX_EVENTS}


async def analyze_preview_change(value):
    if not isinstance(value,dict):
        raise ValueError('Некоректний запит детектора.')
    def compare():
        return compare_frames(preview_image(value.get('reference')),preview_image(value.get('current')))
    return await asyncio.to_thread(compare)
