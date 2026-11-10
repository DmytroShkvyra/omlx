# SPDX-License-Identifier: Apache-2.0
import asyncio
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from omlx.transcription.scene_changes import ChangeGate, choose_events, compare_frames
from omlx.transcription.visual_context import VisualContext, unique_person_labels

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


class SelectionTests(unittest.TestCase):
    def test_budget_keeps_first_frame_and_spans_the_video(self):
        events = [{'time':i,'reason':'initial' if i==0 else 'local_change','score':.5} for i in range(100)]
        selected = choose_events(events,100)
        self.assertEqual(len(selected),12)
        self.assertEqual(selected[0]['time'],0)
        self.assertGreater(selected[-1]['time'],90)

    def test_scene_cut_is_prioritized_over_motion_in_same_time_bucket(self):
        events = [{'time':0,'reason':'initial','score':1}]
        events += [{'time':i,'reason':'object_motion','score':.9} for i in range(1,50)]
        events.append({'time':23.1,'reason':'scene_change','score':.4})
        selected = choose_events(events,50)
        self.assertIn(23.1,[e['time'] for e in selected])

    def test_only_exact_name_variants_are_merged(self):
        names = ['Bordiuhov,  Andrii [External]','bordiuhov, andrii',' Bordiuhov Andrii External ',
                 'Олена','ОЛЕНА','Daniel','Daniel Smith']
        self.assertEqual(unique_person_labels(names),['Bordiuhov, Andrii External','Олена','Daniel','Daniel Smith'])
        state=VisualContext()
        state.observations.extend([{'time':0,'context':'','people':[{'label':name} for name in names]}])
        self.assertEqual(state.metadata()['candidates'],unique_person_labels(names))


@unittest.skipIf(np is None,'Run OpenCV tests in the bundled host oMLX runtime')
class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.base=np.full((180,320,3),180,np.uint8)

    def test_static_cursor_clock_and_compression_noise_are_ignored(self):
        variants=[self.base.copy() for _ in range(4)]
        variants[1][75:81,100:106]=0
        variants[2][8:14,290:303]=0
        rng=np.random.default_rng(1)
        variants[3]=np.clip(self.base.astype(np.int16)+rng.integers(-5,6,self.base.shape),0,255).astype(np.uint8)
        for frame in variants:
            self.assertFalse(compare_frames(self.base,frame)['changed'])

    def test_scene_slide_and_name_changes_are_detected(self):
        scene=np.full_like(self.base,20)
        self.assertEqual(compare_frames(self.base,scene)['reason'],'scene_change')
        slide=self.base.copy();slide[30:135,50:270]=20
        self.assertEqual(compare_frames(self.base,slide)['reason'],'local_change')
        name=self.base.copy();name[100:108,80:155]=20
        self.assertEqual(compare_frames(self.base,name)['reason'],'local_change')

    def test_camera_translation_is_distinguished_from_a_cut(self):
        rng=np.random.default_rng(3)
        reference=cv2.GaussianBlur(rng.integers(0,256,(180,320,3),np.uint8),(3,3),.6)
        current=cv2.warpAffine(reference,np.float32([[1,0,8],[0,1,0]]),(320,180),borderMode=cv2.BORDER_REPLICATE)
        result=compare_frames(reference,current)
        self.assertEqual(result['reason'],'camera_motion',result)
        self.assertAlmostEqual(result['camera_shift'][0],8,delta=1)

    def test_object_motion_has_residual_optical_flow(self):
        rng=np.random.default_rng(9)
        patch=rng.integers(20,240,(50,50,3),np.uint8)
        before=self.base.copy();before[65:115,80:130]=patch
        after=self.base.copy();after[65:115,90:140]=patch
        result=compare_frames(before,after)
        self.assertTrue(result['changed'],result)
        self.assertGreater(result['motion_fraction'],0)

    def test_gate_uses_last_accepted_frame_and_rate_limits(self):
        gate=ChangeGate();self.assertEqual(gate.consider(self.base,0)['reason'],'initial')
        slide=self.base.copy();slide[30:135,50:270]=20
        self.assertIsNone(gate.consider(slide,1))
        self.assertIsNotNone(gate.consider(slide,2))
        self.assertIsNone(gate.consider(slide,5))
        self.assertEqual(gate.sampled,4)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_context_routes_changes_mode_and_preserves_detection_reason(self):
        from tests.transcription.test_visual_context import JPEG
        state=VisualContext(frame_selection='changes')
        info=b'{"streams":[{"codec_type":"video"}],"format":{"duration":"20"}}'
        samples=[{'time':3.5,'images':[JPEG,JPEG],'detection':{'reason':'local_change','score':.7}}]
        async def analyze(frame):
            state.observations.append({'time':frame['time'],'context':'Slide','people':[],'detection':frame['detection']})
        state.analyze=AsyncMock(side_effect=analyze)
        with patch('omlx.transcription.visual_context.command_bytes',new=AsyncMock(return_value=info)),patch(
                'omlx.transcription.scene_changes.extract_change_frames',new=AsyncMock(return_value=(samples,{'mode':'changes','selected_times':[3.5]}))):
            await state.add_video('test.mp4')
        self.assertEqual(state.metadata()['video_sampling']['mode'],'changes')
        self.assertEqual(state.metadata()['observations'][0]['detection']['reason'],'local_change')
        self.assertIn('[00:00:03.500]',state.context())


if __name__=='__main__':unittest.main()
