# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from omlx.transcription.speech_timing import align_window_bounds, estimate_turn_intervals, interval_text

ROOT = Path(__file__).resolve().parents[2] / '.artifacts/transcription-tests'
ROOT.mkdir(parents=True, exist_ok=True)


class BoundsTests(unittest.TestCase):
    def test_anchors_follow_file_offsets_not_relative_text_length(self):
        turns = [{'speaker':7,'text':'Welcome to our meeting today.'},
                 {'speaker':2,'text':'The database release is ready.'}]
        windows = [{'start':0,'end':15,'text':''},
                   {'start':15,'end':30,'text':turns[0]['text']},
                   {'start':30,'end':45,'text':turns[1]['text']}]
        result = align_window_bounds(turns,windows,45)
        self.assertEqual([(t['start'],t['end']) for t in result],[(15,30),(30,45)])
        self.assertEqual([t['speaker'] for t in result],[7,2])
        self.assertEqual(result[0]['timing_method'],'windowed_asr')
        self.assertIn('≈00:00:15.000–00:00:30.000',interval_text(result))

    def test_one_turn_crossing_window_boundary_gets_both_bounds(self):
        turn={'speaker':0,'text':'First part before boundary and second part after boundary.'}
        windows=[{'start':0,'end':15,'text':'First part before boundary'},
                 {'start':15,'end':22,'text':'and second part after boundary'}]
        result=align_window_bounds([turn],windows,22)[0]
        self.assertEqual((result['start'],result['end']),(0,22))

    def test_unknown_alignment_is_explicit_instead_of_fabricated_word_times(self):
        result=align_window_bounds([{'speaker':0,'text':'Hello there.'}],[],30)[0]
        self.assertIsNone(result['start']);self.assertIsNone(result['end'])
        self.assertEqual(result['timing_method'],'unavailable')

    def test_missing_turn_is_conservatively_bracketed(self):
        result=align_window_bounds([{'text':'First sentence.'},{'text':'Unrecognized speech.'},{'text':'Last sentence.'}],
            [{'start':0,'end':15,'text':'First sentence.'},{'start':30,'end':45,'text':'Last sentence.'}],45)
        self.assertEqual((result[1]['start'],result[1]['end']),(0,45))
        self.assertEqual(result[1]['timing_method'],'neighboring_windows')


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_asr_windows_keep_original_ids_without_redecoding(self):
        calls=[]
        async def decode(*args, **kwargs):
            with wave.open(args[-1],'wb') as out:
                out.setparams((1,2,16000,0,'NONE','not compressed'))
                out.writeframes(b'\x00\x00'*16000*20)
            return b''
        class Engine:
            async def transcribe_stream(self,path,**kwargs):
                calls.append(kwargs['task'])
                yield {'text':'First sentence.' if len(calls)==1 else 'Second sentence.'}
        turns=[{'speaker':7,'text':'First sentence.'},{'speaker':2,'text':'Second sentence.'}]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch('omlx.transcription.speech_timing.command_bytes',decode):
            cache={}
            first=await estimate_turn_intervals(Engine(),str(Path(directory)/'input.wav'),turns,cache)
            second=await estimate_turn_intervals(Engine(),str(Path(directory)/'input.wav'),turns,cache)
        self.assertEqual(calls,['asr','asr'])
        self.assertEqual(first,second)
        self.assertEqual([t['speaker'] for t in second],[7,2])


if __name__=='__main__':
    unittest.main()
