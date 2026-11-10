# SPDX-License-Identifier: Apache-2.0
import unittest
from omlx.transcription.diarization import (SpeakerTextStream, readable_text, speaker_segments,
    resolve_nemotron_params, ORIGINAL_LANGUAGE_PROMPT, SPEAKER_ORIGINAL_LANGUAGE_PROMPT)


class SpeakerTests(unittest.TestCase):
    def test_every_token_split_preserves_speaker_turns(self):
        raw = '<spk:0>Hello. <spk:1>Good morning! <spk:0>Welcome back.'
        expected = '\n\n**Мовець 1:** Hello. \n\n**Мовець 2:** Good morning! \n\n**Мовець 1:** Welcome back.'
        for size in range(1, len(raw) + 1):
            formatter = SpeakerTextStream()
            parts = [formatter.feed(raw[i:i + size]) for i in range(0, len(raw), size)]
            parts.append(formatter.feed('', final=True))
            self.assertEqual(''.join(parts), expected)
        self.assertEqual(readable_text(raw), expected.strip())

    def test_speaker_ids_reused_without_invented_timestamps(self):
        self.assertEqual(speaker_segments('<spk:7>First.<spk:0>Reply.<spk:7>Again.'), [
            {'speaker': 7, 'text': 'First.'}, {'speaker': 0, 'text': 'Reply.'},
            {'speaker': 7, 'text': 'Again.'}])

    def test_plain_speech_is_preserved_if_model_emits_no_speaker(self):
        self.assertEqual(readable_text('Ordinary text'), 'Ordinary text')
        self.assertEqual(speaker_segments('Ordinary text'), [])

    def test_resolve_nemotron_params(self):
        for target in ('Ukrainian', 'Polish', 'English'):
            for translate in (False, True):
                for diarize in (False, True):
                    self.assertEqual(resolve_nemotron_params(diarize, translate, target),
                                     (SPEAKER_ORIGINAL_LANGUAGE_PROMPT if diarize else ORIGINAL_LANGUAGE_PROMPT,
                                      'saa' if diarize else 'asr', None))



if __name__ == '__main__':
    unittest.main()
