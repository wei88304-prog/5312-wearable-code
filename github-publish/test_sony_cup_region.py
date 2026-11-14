import math
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from sony_cup_region import cup_region
from sony_guidance import GuidanceController, GuidancePanel
from sony_haptics import SerialHaptics, ZERO, directions_for_state
from sony_targets import TargetDetector
from test_sony_guidance import item, stabilize
from test_sony_haptics import Esp32Peer


class CupRegionTests(unittest.TestCase):
    def target(self, box):
        cup = item(kind='cup')
        cup['xyxy'] = box
        cup['position'] = dict(x=(box[0]+box[2])/2-960, y=540-(box[1]+box[3])/2)
        return cup

    def test_reference_roi_and_uniform_scaling(self):
        self.assertEqual(cup_region(None, 1920, 1080)['roi_xyxy'], [710,165,1210,915])
        x1,y1,x2,y2 = cup_region(None,640,480)['roi_xyxy']
        self.assertAlmostEqual(x2-x1, 500/3)
        self.assertAlmostEqual(y2-y1, 250)
        self.assertGreater(x1,0)
        self.assertGreater(y1,0)

    def test_full_box_required_not_just_center(self):
        cup = self.target([700,500,800,580])
        r = cup_region(cup,1920,1080)
        self.assertEqual(r['status'],'outside')
        self.assertEqual(r['directions'],dict(up=False,right=False,down=False,left=True))
        self.assertFalse(r['inside'])
        self.assertTrue(cup_region(self.target([710,165,1210,915]),1920,1080)['inside'])

    def test_four_directions_and_diagonal(self):
        for box, names in (([700,500,800,580],{'left'}),([1150,500,1220,580],{'right'}),
                           ([900,150,1000,250],{'up'}),([900,850,1000,920],{'down'}),
                           ([1150,150,1220,250],{'up','right'})):
            r=cup_region(self.target(box),1920,1080)
            self.assertEqual({k for k,v in r['directions'].items() if v},names)

    def test_missing_invalid_and_oversize_are_silent(self):
        for target in (None,item(kind='person'),self.target([0,0,1800,1000]),
                       self.target([math.nan,0,100,100]),self.target([100,100,20,20])):
            r=cup_region(target,1920,1080)
            self.assertFalse(r['inside'])
            self.assertFalse(any(r['directions'].values()))

    def controller(self):
        c=GuidanceController(cup_target=True)
        stabilize(c,np.zeros((480,640,3),np.uint8))
        return c

    def test_only_cups_select_and_no_fallback_after_loss(self):
        c=self.controller()
        cup=item('base:2',kind='cup',x=150,y=0)
        c.update_detections([item('base:1'),cup])
        self.assertEqual([t['class'] for t in c.candidates],['cup'])
        self.assertTrue(c.select(c.candidates[0]['number']))
        self.assertEqual(directions_for_state(c.snapshot(640,480),640,480),ZERO)
        c.update_detections([cup])
        self.assertEqual(directions_for_state(c.snapshot(640,480),640,480),(False,True,False,False))
        c.update_detections([item('base:3',kind='cup',x=-150,y=0)])
        self.assertEqual(c.phase,'lost')
        self.assertEqual(directions_for_state(c.snapshot(640,480),640,480),ZERO)

    def test_full_region_instead_of_three_percent_center_and_render(self):
        c=self.controller()
        cup=item(kind='cup',x=40,y=0)
        c.update_detections([cup]); c.select(c.candidates[0]['number']); c.update_detections([cup])
        s=c.snapshot(640,480)
        self.assertTrue(s['cup_guidance']['inside'])
        self.assertTrue(s['guidance']['centered'])
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        panel=GuidancePanel()
        view=panel.render(np.zeros((480,640,3),np.uint8),s)
        self.assertEqual(view.shape,(760,1280,3))
        self.assertTrue(np.any(view))

    def test_edge_can_require_motion_even_when_center_is_near_zero(self):
        c=self.controller()
        cup=item(kind='cup',x=-5,y=0)
        cup['xyxy']=[235,200,395,280]
        c.update_detections([cup]);c.select(c.candidates[0]['number']);c.update_detections([cup])
        s=c.snapshot(640,480)
        self.assertFalse(s['guidance']['centered'])
        self.assertEqual(directions_for_state(s,640,480),(False,False,False,True))

    def test_all_cases_follow_roi_stop_and_stale_stop(self):
        c=self.controller()
        cup=item(kind='cup',x=150,y=0)
        c.update_detections([cup]);c.select(c.candidates[0]['number']);c.update_detections([cup])
        for case in (1,2,3):
            with Esp32Peer() as peer:
                h=SerialHaptics(peer.port,case=case)
                try:
                    out=h.update(c.snapshot(640,480),640,480,observed_at=time.monotonic())
                    self.assertTrue(out['directions']['right'])
                    out=h.update(c.snapshot(640,480),640,480,observed_at=time.monotonic()-1)
                    self.assertFalse(out['fresh'])
                    self.assertFalse(any(out['directions'].values()))
                    inside=dict(c.snapshot(640,480))
                    inside['cup_guidance']=cup_region(item(kind='cup',x=0,y=0),640,480)
                    inside['guidance']=dict(inside['guidance'],centered=True)
                    out=h.update(inside,640,480,observed_at=time.monotonic())
                    self.assertFalse(any(out['directions'].values()))
                finally:h.close()

    def test_cup_only_detector_does_not_request_person_tree(self):
        # Exercise actual tracker selection without loading model weights.
        from unittest.mock import MagicMock
        d=TargetDetector.__new__(TargetDetector)
        d.cups_only=True
        d.base_classes={'person':0,'cup':41};d.tree_classes=[1];d.cup_classes=[2]
        d.thresholds={'person':.35,'cup':.2,'tree':.2};d.device='cpu'
        d.base=MagicMock();d.trees=MagicMock()
        d.base.track.return_value=[MagicMock(boxes=[])]
        d.trees.track.return_value=[MagicMock(boxes=[])]
        d.track(np.zeros((96,96,3),np.uint8))
        self.assertEqual(d.base.track.call_args.kwargs['classes'],[41])
        self.assertEqual(d.trees.track.call_args.kwargs['classes'],[2])
