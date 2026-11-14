import contextlib
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from sony_guidance import GuidanceController
from sony_haptics import ZERO, directions_for_state
from sony_object_lock import VisualObjectLock
from test_sony_guidance import item


class ObjectLockTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((480,640,3),np.uint8)
        self.c=GuidanceController(cup_target=True,auto_target=True)
        self.fake=MagicMock()
        self.fake.init.return_value=None
        self.fake.update.return_value=(True,(440,200,60,80))
        self.lock=VisualObjectLock(factory=lambda:self.fake)
        self.detector=MagicMock()
        self.detector.detect.return_value=[('base',SimpleNamespace())]
        self.decoder=lambda result,w,h:[item(kind='cup',x=150,y=0)]

    def step(self,t):
        with patch('sony_object_lock.time.monotonic',return_value=t):
            self.lock.step(self.frame,self.c,self.detector,self.decoder,t)
        return self.c.snapshot(640,480)

    def test_detect_once_then_track_without_any_semantic_calls(self):
        self.assertEqual(self.step(0)['phase'],'arming')
        s=self.step(1)
        self.assertEqual(s['phase'],'tracking')
        self.assertEqual(s['target']['tracking_method'],'CSRT')
        self.assertEqual(self.detector.detect.call_count,1)
        self.assertEqual(self.fake.init.call_count,1)
        self.assertEqual(directions_for_state(s,640,480),(False,True,False,False))
        for t in (2,3,4):self.step(t)
        self.assertEqual(self.detector.detect.call_count,1)
        self.detector.track.assert_not_called()

    def test_failure_is_silent_and_never_restarts_detector(self):
        s=self.step(0); identity=s['selected_track_id']
        self.step(1)
        self.fake.update.return_value=(False,(0,0,0,0))
        for t in (2,3,20):
            s=self.step(t)
            self.assertEqual(s['selected_track_id'],identity)
            self.assertEqual(s['phase'],'lost')
            self.assertEqual(directions_for_state(s,640,480),ZERO)
        self.assertEqual(self.detector.detect.call_count,1)
        self.fake.update.return_value=(True,(440,200,60,80))
        self.assertEqual(self.step(21)['phase'],'tracking')

    def test_reset_is_the_only_way_to_scan_again(self):
        self.step(0);self.step(1)
        self.c.reset()
        self.assertEqual(self.step(2)['phase'],'arming')
        self.assertEqual(self.detector.detect.call_count,2)
        self.assertEqual(self.fake.init.call_count,2)

    def test_scan_retries_until_initial_cup(self):
        self.detector.detect.return_value=[]
        for t in (0,1,2):self.assertEqual(self.step(t)['phase'],'empty')
        self.detector.detect.return_value=[('base',SimpleNamespace())]
        self.assertEqual(self.step(3)['phase'],'arming')
        self.assertEqual(self.detector.detect.call_count,4)

    def test_invalid_or_out_of_frame_box_never_supplies_old_coordinates(self):
        self.step(0)
        for box in ((float('nan'),0,50,50),(-200,0,60,80),(20,20,0,0)):
            self.fake.update.return_value=(True,box)
            self.assertIsNone(self.lock.update(self.frame))
        self.assertIsNone(self.lock.update(np.zeros((240,320,3),np.uint8)))

    def test_real_csrt_tracks_generated_object_translation(self):
        rng=np.random.default_rng(5)
        patch_image=rng.integers(30,240,(90,70,3),dtype=np.uint8)
        def frame(x,y):
            f=np.full((360,480,3),40,np.uint8)
            f[y:y+90,x:x+70]=patch_image
            return f
        lock=VisualObjectLock()
        target=item(kind='cup')
        target['xyxy']=[180,120,250,210]
        lock.start(frame(180,120),target)
        for x,y in ((185,122),(191,125),(198,128),(205,131)):
            found=lock.update(frame(x,y))
            self.assertIsNotNone(found)
            self.assertLess(abs(found['xyxy'][0]-x),12)
            self.assertLess(abs(found['xyxy'][1]-y),12)

    def test_actual_sony_main_confirms_three_frames_without_serial(self):
        import sony_wifi_yolo as app
        camera=MagicMock(); camera.start.return_value='unused'
        reader=MagicMock()
        reader.get.side_effect=[(1,self.frame),(2,self.frame),(3,self.frame)]
        fake_result=SimpleNamespace(boxes=[],names={})
        self.detector.detect.return_value=[('base',fake_result)]
        with tempfile.NamedTemporaryFile(suffix='.pt') as model:
            output=io.StringIO()
            with patch('sony_targets.TargetDetector',return_value=self.detector), \
                 patch('sony_object_lock.cv2.TrackerCSRT_create',return_value=self.fake), \
                 patch.object(app,'detection_items',self.decoder), \
                 patch.object(app,'discover',return_value=[('fake','http://127.0.0.1/fake')]), \
                 patch.object(app,'SonyCamera',return_value=camera), \
                 patch.object(app,'LatestFrame',return_value=reader), \
                 patch('serial.Serial',side_effect=AssertionError('No physical serial')), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(app.main(['--cup-target','--headless','--frames','3','--model',model.name]),0)
        self.assertEqual(self.detector.detect.call_count,3)
        self.detector.track.assert_not_called()
        self.assertIn('CSRT',output.getvalue())
        reader.close.assert_called_once()
        camera.close.assert_called_once()
