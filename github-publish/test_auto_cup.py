import unittest
import numpy as np
from sony_guidance import GuidanceController, GuidancePanel
from sony_haptics import ZERO, directions_for_state
from test_sony_guidance import item


class AutoCupTests(unittest.TestCase):
    def setUp(self):
        self.c = GuidanceController(cup_target=True, auto_target=True)
        self.frame = np.zeros((480,640,3),np.uint8)
        self.cup = item('base:1', kind='cup', x=150, y=0)

    def update(self, now, cups=None):
        self.assertTrue(self.c.observe(self.frame, now))
        self.c.update_detections([self.cup] if cups is None else cups, completed_at=now)
        return self.c.snapshot(640,480)

    def test_immediate_detection_but_full_one_second_wait(self):
        for t in (0,.3,.999):
            s=self.update(t)
            self.assertEqual(s['phase'],'arming')
            self.assertEqual(directions_for_state(s,640,480),ZERO)
        s=self.update(1)
        self.assertEqual(s['phase'],'tracking')
        self.assertEqual(directions_for_state(s,640,480),(False,True,False,False))

    def test_snapshot_alone_cannot_release_wait(self):
        self.update(0)
        self.assertEqual(self.c.snapshot(640,480)['phase'],'arming')

    def test_nearest_cup_then_fixed_identity(self):
        other=item('base:2',kind='cup',x=200,y=0)
        s=self.update(0,[other,self.cup,item('base:3',kind='person',x=0,y=0)])
        self.assertEqual(s['selected_track_id'],'base:1')
        other['position']['x']=0
        s=self.update(1,[other,self.cup])
        self.assertEqual(s['selected_track_id'],'base:1')

    def test_loss_stops_and_same_cup_resumes_without_rearming(self):
        self.update(0);self.update(1)
        other=item('base:2',kind='cup',x=0,y=0)
        s=self.update(1.1,[other])
        self.assertEqual(s['phase'],'lost')
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        s=self.update(2)
        self.assertEqual(s['phase'],'tracking')
        self.assertEqual(directions_for_state(s,640,480),(False,True,False,False))
        self.assertEqual(self.update(3)['phase'],'tracking')

    def test_inside_target_does_not_vibrate_after_wait(self):
        self.cup=item('base:1',kind='cup',x=0,y=0)
        self.update(0)
        s=self.update(1)
        self.assertTrue(s['cup_guidance']['inside'])
        self.assertEqual(directions_for_state(s,640,480),ZERO)

    def test_reset_restarts_auto_lock_and_wait(self):
        self.update(0);self.update(1)
        self.c.reset()
        s=self.update(5)
        self.assertEqual(s['phase'],'arming')
        self.assertEqual(self.update(6)['phase'],'tracking')

    def test_empty_scene_and_waiting_ui_render(self):
        s=self.update(0,[])
        self.assertEqual(s['phase'],'empty')
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        s=self.update(1)
        self.assertEqual(s['phase'],'arming')
        self.assertEqual(GuidancePanel().render(self.frame,s).shape,(760,1280,3))

    def test_scan_until_late_cup_without_reset(self):
        for t in (0,1,2,5,10):
            self.assertEqual(self.update(t,[])['phase'],'empty')
        self.assertEqual(self.update(11)['phase'],'arming')
        self.assertEqual(self.update(12)['phase'],'tracking')

    def test_old_id_never_expires_or_switches_to_another_cup(self):
        self.update(0);self.update(1)
        other=item('base:9',kind='cup',x=-150,y=0)
        s=self.update(2,[other])
        self.assertEqual(s['phase'],'lost')
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        s=self.update(3,[other])
        self.assertEqual(s['selected_track_id'],'base:1')
        self.assertEqual(s['phase'],'lost')
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        s=self.update(4,[other])
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        s=self.update(100,[other])
        self.assertEqual(s['selected_track_id'],'base:1')
        self.assertEqual(s['phase'],'lost')

    def test_empty_after_loss_preserves_identity_until_explicit_reset(self):
        self.update(0);self.update(1)
        self.update(2,[])
        s=self.update(3,[])
        self.assertEqual(s['phase'],'lost')
        self.assertEqual(s['selected_track_id'],'base:1')
        other=item('base:9',kind='cup',x=-150,y=0)
        self.assertEqual(self.update(5,[other])['phase'],'lost')
        self.c.reset()
        self.assertEqual(self.update(5,[other])['phase'],'arming')
        self.assertEqual(self.update(6,[other])['phase'],'tracking')

    def test_loss_during_initial_delay_restarts_delay_but_not_identity(self):
        self.update(0)
        self.update(.5,[])
        self.assertEqual(self.update(2)['phase'],'arming')
        self.assertEqual(self.update(3)['phase'],'tracking')

    def test_visual_feedback_without_opening_any_serial_device(self):
        from unittest.mock import patch
        import time
        from sony_haptics import preview_feedback
        with patch('serial.Serial',side_effect=AssertionError('Must not open serial')):
            waiting=preview_feedback(self.update(0),640,480,time.monotonic())
            self.assertFalse(any(waiting['directions'].values()))
            active=preview_feedback(self.update(1),640,480,time.monotonic())
            self.assertTrue(active['directions']['right'])
            self.assertTrue(active['simulated'])
            stale=preview_feedback(self.c.snapshot(640,480),640,480,time.monotonic()-1)
            self.assertFalse(any(stale['directions'].values()))
