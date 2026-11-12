# SPDX-License-Identifier: Apache-2.0
"""Cheap PCM16 pause detection for scheduling, not ASR or speaker attribution."""
from array import array
import sys


class SpeechBoundary:
    def __init__(self, sample_rate=16000, max_seconds=4, silence_seconds=.6, rms_threshold=.01):
        self.frame_samples = round(sample_rate * .02)
        self.max_bytes = round(sample_rate * max_seconds) * 2
        self.silence_samples = round(sample_rate * silence_seconds)
        self.energy_threshold = (32768 * rms_threshold) ** 2
        self.pending = b''
        self.quiet_samples = 0
        self.has_speech = False
        self.last_snapshot = 0

    def feed(self, pcm):
        self.pending += pcm
        frame_bytes = self.frame_samples * 2
        end = len(self.pending) // frame_bytes * frame_bytes
        for offset in range(0, end, frame_bytes):
            values = array('h', self.pending[offset:offset + frame_bytes])
            if sys.byteorder != 'little':
                values.byteswap()
            voiced = sum(value * value for value in values) >= self.energy_threshold * len(values)
            if voiced:
                self.has_speech = True
                self.quiet_samples = 0
            else:
                self.quiet_samples += self.frame_samples
        self.pending = self.pending[end:]

    def due(self, audio_bytes):
        if not self.has_speech or audio_bytes <= self.last_snapshot:
            return None
        if self.quiet_samples >= self.silence_samples:
            return 'silence'
        if audio_bytes - self.last_snapshot >= self.max_bytes:
            return 'max_interval'
        return None

    def snapshot_started(self, audio_bytes):
        self.last_snapshot = audio_bytes
        self.has_speech = False
        self.quiet_samples = 0
