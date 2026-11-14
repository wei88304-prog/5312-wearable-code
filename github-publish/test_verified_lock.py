import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from sony_guidance import GuidanceController
from sony_verified_lock import VerifiedObjectLock
from sony_haptics import ZERO,directions_for_state
from test_sony_guidance import item


class VerifiedLockTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((480,640,3),np.uint8)
        self.c=GuidanceController(cup_target=True,auto_target=True)
        self.fake=MagicMock()
        self.fake.init.return_value=None
        self.fake.update.return_value=(True,(440,200,60,80))
        self.lock=VerifiedObjectLock(factory=lambda:self.fake)
        self.detector=MagicMock()
        self.candidates=[item(kind='cup',x=150,y=0)]
        self.detector.detect.side_effect=self.scan
        self.decoder=lambda result,w,h:result.items

    def scan(self,view):
        ox,oy=0,0
        if view.shape[:2]!=(480,640):ox,oy,_,_=self.lock.region()
        result=[]
        for original in self.candidates:
            x1,y1,x2,y2=original['xyxy']
            if x2<=ox or y2<=oy or x1>=ox+view.shape[1] or y1>=oy+view.shape[0]:continue
            entry=dict(original)
            entry['xyxy']=[x1-ox,y1-oy,x2-ox,y2-oy]
            result.append(entry)
        return [('base',SimpleNamespace(items=result))]

    def step(self,t):
        with patch('sony_verified_lock.time.monotonic',return_value=t):
            self.lock.step(self.frame,self.c,self.detector,self.decoder,t)
        return self.c.snapshot(640,480)

    def lock_target(self):
        self.step(0);self.step(.1)
        return self.step(.2)

    def test_requires_three_frames_and_one_second_after_lock(self):
        for t in (0,.1):
            s=self.step(t)
            self.assertIsNone(s['selected_track_id'])
            self.assertEqual(directions_for_state(s,640,480),ZERO)
        self.assertEqual(self.step(.2)['phase'],'arming')
        self.assertEqual(self.step(1.1)['phase'],'arming')
        self.assertEqual(self.step(1.3)['phase'],'tracking')

    def test_low_confidence_single_frame_or_interruption_cannot_lock(self):
        self.candidates[0]['confidence']=.1
        for t in (0,.1,.2,.3):self.assertIsNone(self.step(t)['selected_track_id'])
        self.candidates[0]['confidence']=.9
        self.step(.4);self.step(.5)
        self.candidates=[]
        self.step(.6)
        self.candidates=[item(kind='cup',x=150,y=0)]
        self.assertIsNone(self.step(.7)['selected_track_id'])
        self.assertIsNone(self.step(.8)['selected_track_id'])
        self.assertIsNotNone(self.step(.9)['selected_track_id'])

    def test_inconsistent_locations_reset_confirmation(self):
        self.step(0)
        self.candidates=[item(kind='cup',x=-150,y=0)]
        self.step(.1)
        self.assertEqual(self.lock.pending_count,1)
        self.assertIsNone(self.c.selected_id)

    def test_duplicate_model_boxes_do_not_count_as_multiple_frames(self):
        self.candidates*=2
        self.step(0)
        self.assertEqual(self.lock.pending_count,1)
        self.assertIsNone(self.c.selected_id)

    def test_verification_is_local_and_low_frequency(self):
        self.lock_target()
        self.step(.3);self.step(.4)
        self.assertEqual(self.detector.detect.call_count,3)
        self.step(.8)
        self.assertEqual(self.detector.detect.call_count,4)
        view=self.detector.detect.call_args.args[0]
        self.assertLess(view.shape[0],480)
        self.assertLess(view.shape[1],640)
        self.step(.9)
        self.assertEqual(self.detector.detect.call_count,4)

    def test_local_detection_corrects_tracker_drift_without_new_identity(self):
        identity=self.lock_target()['selected_track_id']
        self.fake.update.return_value=(True,(530,200,60,80))
        s=self.step(.8)
        self.assertEqual(s['selected_track_id'],identity)
        self.assertEqual(s['target']['xyxy'],[440,200,500,280])
        self.assertEqual(self.fake.init.call_count,2)

    def test_semantic_misses_never_expire_reliable_current_image_tracking(self):
        identity=self.lock_target()['selected_track_id']
        self.candidates=[]
        for t in (.8,.9,1.3,2.4,5,30,60):
            s=self.step(t)
            self.assertIn(s['phase'],('arming','tracking'))
            self.assertEqual(s['selected_track_id'],identity)
            self.assertIsNotNone(s['target'])
        self.assertEqual(s['phase'],'tracking')
        self.assertNotEqual(directions_for_state(s,640,480),ZERO)
        self.assertIn('当前视觉跟踪有效',self.lock.status)

    def test_visual_failure_still_stops_after_long_semantic_dropout(self):
        self.lock_target()
        self.candidates=[]
        for t in (.8,1.3,2.4,5):self.step(t)
        self.fake.update.return_value=(False,(0,0,0,0))
        with patch('sony_verified_lock.flow_box',return_value=None):
            s=self.step(5.1)
        self.assertEqual(s['phase'],'lost')
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        self.assertIn('无法可靠定位',self.lock.status)

    def test_far_cup_never_replaces_lost_original(self):
        identity=self.lock_target()['selected_track_id']
        self.fake.update.return_value=(False,(0,0,0,0))
        self.candidates=[item('other',kind='cup',x=-150,y=0)]
        for t in (.8,1.5,5):
            s=self.step(t)
            self.assertEqual(s['selected_track_id'],identity)
            self.assertEqual(s['phase'],'lost')
        self.assertEqual(self.fake.init.call_count,1)

    def test_recovers_matching_nearby_cup_after_csrt_failure(self):
        identity=self.lock_target()['selected_track_id']
        self.fake.update.return_value=(False,(0,0,0,0))
        s=self.step(.8)
        self.assertEqual(s['selected_track_id'],identity)
        self.assertIsNotNone(s['target'])
        self.assertEqual(self.fake.init.call_count,2)

    def test_different_appearance_is_not_accepted_as_original(self):
        self.lock_target()
        self.frame[:]=(0,0,255)
        self.assertEqual(self.step(.8)['phase'],'lost')

    def test_two_similar_neighbours_are_ambiguous(self):
        self.lock_target()
        self.candidates=[item('a',kind='cup',x=120,y=0),item('b',kind='cup',x=180,y=0)]
        self.assertEqual(self.step(.8)['phase'],'lost')

    def test_frame_resize_and_reset_do_not_reuse_old_coordinates(self):
        self.lock_target()
        self.frame=np.zeros((240,320,3),np.uint8)
        s=self.step(.8)
        self.assertEqual(s['phase'],'lost')
        self.assertIsNone(s['target'])
        self.frame=np.zeros((480,640,3),np.uint8)
        self.c.reset()
        self.assertIsNone(self.step(1)['selected_track_id'])
        self.assertEqual(self.lock.pending_count,1)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({'lock_confidence':-1},{'lock_confidence':2},{'confirm_frames':1}):
            with self.assertRaises(ValueError):VerifiedObjectLock(**kwargs)

    def test_reset_during_confirmation_discards_partial_count(self):
        self.step(0);self.step(.1)
        self.assertEqual(self.lock.pending_count,2)
        self.c.reset()
        self.step(.2)
        self.assertEqual(self.lock.pending_count,1)
        self.assertIsNone(self.c.selected_id)

    def test_search_and_correction_follow_current_position_not_old_anchor(self):
        self.lock_target()
        for t,x in ((.3,460),(.4,480),(.5,500),(.8,520)):
            self.fake.update.return_value=(True,(x,200,60,80))
            self.candidates=[item(kind='cup',x=x+30-320,y=0)]
            s=self.step(t)
            self.assertIsNotNone(s['target'])
        self.assertEqual(s['target']['xyxy'],[520,200,580,280])
        self.assertEqual(self.lock.anchor,[520,200,580,280])

    def test_failed_current_image_tracking_never_outputs_motion_prediction(self):
        self.lock_target()
        self.fake.update.return_value=(True,(460,200,60,80))
        self.step(.3)
        self.fake.update.return_value=(False,(0,0,0,0))
        self.candidates=[]
        with patch('sony_verified_lock.flow_box',return_value=None):
            s=self.step(.4)
        self.assertEqual(s['phase'],'lost')
        self.assertEqual(directions_for_state(s,640,480),ZERO)
        self.assertIsNotNone(self.lock.search_box)

    def test_optical_flow_current_image_evidence_can_replace_failed_csrt(self):
        self.lock_target()
        self.fake.update.return_value=(False,(0,0,0,0))
        with patch('sony_verified_lock.flow_box',return_value=[445,200,505,280]):
            s=self.step(.3)
        self.assertIsNotNone(s['target'])
        self.assertEqual(self.lock.tracking_source,'optical flow')
        self.assertEqual(s['target']['xyxy'],[445,200,505,280])

    def test_ambiguous_identity_stays_paused_until_resolved(self):
        self.lock_target()
        self.candidates=[item('a',kind='cup',x=120,y=0),item('b',kind='cup',x=180,y=0)]
        self.assertEqual(self.step(.8)['phase'],'lost')
        self.assertEqual(self.step(.9)['phase'],'lost')
        self.candidates=[item(kind='cup',x=150,y=0)]
        self.assertIsNotNone(self.step(1.4)['target'])

    def test_failed_verification_reports_rejection_reason(self):
        self.lock_target()
        self.candidates=[]
        self.step(.8)
        self.assertEqual(self.lock.reject_reason,'局部未检出杯子或分数不足')

    def test_motion_auxiliary_tracks_generated_translation(self):
        from sony_motion import flow_box
        rng=np.random.default_rng(9)
        texture=rng.integers(0,255,(90,70,3),np.uint8)
        import cv2
        a=np.full((480,640,3),30,np.uint8);b=a.copy()
        a[180:270,200:270]=texture
        b[190:280,220:290]=texture
        result=flow_box(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),[200,180,270,270])
        self.assertIsNotNone(result)
        self.assertLess(abs(result[0]-220),3)
        self.assertLess(abs(result[1]-190),3)

    def test_motion_auxiliary_refuses_blank_frame_or_insufficient_features(self):
        from sony_motion import flow_box
        a=np.zeros((480,640),np.uint8)
        self.assertIsNone(flow_box(a,a,[200,180,270,270]))
        self.assertIsNone(flow_box(a,np.zeros((240,320),np.uint8),[200,180,270,270]))

    def test_real_continuous_pan_survives_semantic_dropout_beyond_two_seconds(self):
        # Real CSRT + optical flow on generated moving imagery, not mocked tracking.
        rng=np.random.default_rng(12)
        texture=rng.integers(30,235,(90,70,3),np.uint8)
        self.lock=VerifiedObjectLock()
        for i in range(30):
            x=170+max(0,i-2)*6
            self.frame=np.full((480,640,3),40,np.uint8)
            self.frame[190:280,x:x+70]=texture
            target=item(kind='cup')
            target['xyxy']=[x,190,x+70,280]
            self.candidates=[] if i>=6 else [target]
            s=self.step(i*.1)
            if i>=2:
                self.assertIsNotNone(s['target'],f'lost on moving frame {i}: {self.lock.status}')
                self.assertLess(abs(s['target']['xyxy'][0]-x),15)
            if i>=13:self.assertEqual(s['phase'],'tracking')
