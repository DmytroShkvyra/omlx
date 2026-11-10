# SPDX-License-Identifier: Apache-2.0
import unittest
from omlx.transcription.video_keyframes import select_keyframes
from omlx.transcription.visual_context import VisualContext


class KeyframeTests(unittest.TestCase):
    def test_only_marked_keyframes_with_known_times_are_selected(self):
        frames=[{'key_frame':1,'best_effort_timestamp_time':'0'},
                {'key_frame':0,'best_effort_timestamp_time':'1'},
                {'key_frame':1,'best_effort_timestamp_time':'2'},
                {'key_frame':1},
                {'key_frame':1,'best_effort_timestamp_time':'NaN'}]
        result,total=select_keyframes(frames,10)
        self.assertEqual(total,2)
        self.assertEqual(result,[{'time':0,'decode_index':0},{'time':2,'decode_index':2}])

    def test_nonzero_container_origin_is_normalized_to_file_clock(self):
        frames=[{'key_frame':1,'best_effort_timestamp_time':t} for t in ['9','10','12','20']]
        result,total=select_keyframes(frames,10,start_time=10)
        self.assertEqual([f['time'] for f in result],[0,2])
        self.assertEqual(total,2)

    def test_budget_covers_the_entire_keyframe_sequence(self):
        frames=[{'key_frame':1,'best_effort_timestamp_time':str(i*2)} for i in range(100)]
        result,total=select_keyframes(frames,200)
        self.assertEqual(total,100);self.assertEqual(len(result),12)
        self.assertEqual(result[0]['time'],0);self.assertEqual(result[-1]['time'],198)
        self.assertEqual(len({f['decode_index'] for f in result}),12)

    def test_no_keyframes_are_not_silently_replaced_with_other_frames(self):
        self.assertEqual(select_keyframes([{'key_frame':0,'best_effort_timestamp_time':'1'}],10),([],0))
        with self.assertRaises(ValueError):VisualContext(frame_selection='unknown')


if __name__=='__main__':unittest.main()
