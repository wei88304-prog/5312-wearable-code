"""Conservative multi-frame lock and local semantic correction of a CSRT track."""
import math
import time
import cv2
import numpy as np
from sony_object_lock import VisualObjectLock
from sony_targets import target_class
from sony_motion import flow_box


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0


def nearby(a,b,w,h):
    aw,ah,bw,bh=a[2]-a[0],a[3]-a[1],b[2]-b[0],b[3]-b[1]
    return (min(aw,ah,bw,bh)>=4 and .6<=bw/aw<=1.67 and .6<=bh/ah<=1.67
            and abs((a[0]+a[2]-b[0]-b[2])/2)<=min(w*.12,max(aw*.9,12))
            and abs((a[1]+a[3]-b[1]-b[3])/2)<=min(h*.12,max(ah*.9,12)))


class VerifiedObjectLock(VisualObjectLock):
    VERIFY_INTERVAL=.5
    MOTION_GRACE=.75

    def __init__(self,factory=None,lock_confidence=.30,confirm_frames=3):
        if not 0<=lock_confidence<=1 or confirm_frames<2:
            raise ValueError('锁定阈值为0至1，确认帧数至少2')
        self.lock_confidence=lock_confidence
        self.confirm_frames=confirm_frames
        super().__init__(factory)

    def clear(self):
        super().clear()
        self.pending=None
        self.pending_count=0
        self.pending_at=None
        self.anchor=None
        self.appearance=None
        self.last_check=float('-inf')
        self.verified=False
        self.motion_frame=None
        self.motion_box=None
        self.motion_at=None
        self.velocity=np.zeros(2)
        self.search_box=None
        self.reject_reason='尚未锁定'
        self.tracking_source='none'
        self.ambiguous=False
        self.status='持续扫描水杯，等待连续多帧确认。'

    @staticmethod
    def patch(frame,box):
        h,w=frame.shape[:2]
        x1,y1,x2,y2=box
        image=frame[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]
        return cv2.resize(image,(32,32)) if image.size else None

    def same_appearance(self,frame,box):
        candidate=self.patch(frame,box)
        if candidate is None or self.appearance is None:return False
        a=cv2.cvtColor(self.appearance,cv2.COLOR_BGR2HSV)
        b=cv2.cvtColor(candidate,cv2.COLOR_BGR2HSV)
        ha=cv2.calcHist([a],[0,1],None,[16,8],[0,180,0,256])
        hb=cv2.calcHist([b],[0,1],None,[16,8],[0,180,0,256])
        cv2.normalize(ha,ha);cv2.normalize(hb,hb)
        if cv2.compareHist(ha,hb,cv2.HISTCMP_BHATTACHARYYA)>.55:return False
        ga=cv2.cvtColor(self.appearance,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb=cv2.cvtColor(candidate,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if min(ga.std(),gb.std())<5:return float(np.abs(ga-gb).mean())<=45
        return float(np.corrcoef(ga.ravel(),gb.ravel())[0,1])>=.15

    def visual_appearance(self,frame,box):
        """Looser current-image check tolerates blur, but rejects major colour changes."""
        candidate=self.patch(frame,box)
        if candidate is None or self.appearance is None:return False
        a=cv2.cvtColor(self.appearance,cv2.COLOR_BGR2HSV)
        b=cv2.cvtColor(candidate,cv2.COLOR_BGR2HSV)
        ha=cv2.calcHist([a],[0,1],None,[16,8],[0,180,0,256])
        hb=cv2.calcHist([b],[0,1],None,[16,8],[0,180,0,256])
        cv2.normalize(ha,ha);cv2.normalize(hb,hb)
        if cv2.compareHist(ha,hb,cv2.HISTCMP_BHATTACHARYYA)>.65:return False
        return abs(float(self.appearance.mean())-float(candidate.mean()))<65

    def remember_motion(self,frame,box,now):
        if self.motion_box is not None and self.motion_at is not None and now>self.motion_at:
            old=np.array([(self.motion_box[0]+self.motion_box[2])/2,(self.motion_box[1]+self.motion_box[3])/2])
            new=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
            self.velocity=.5*self.velocity+.5*(new-old)/(now-self.motion_at)
        self.motion_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.motion_box=list(box)
        self.motion_at=now

    def motion_prediction(self,now):
        if self.motion_box is None:return self.anchor
        dt=min(self.MOTION_GRACE,max(0,now-self.motion_at))
        dx,dy=self.velocity*dt
        w,h=self.frame_size
        a=self.motion_box
        dx=float(np.clip(dx,-a[0],w-a[2]));dy=float(np.clip(dy,-a[1],h-a[3]))
        return [a[0]+dx,a[1]+dy,a[2]+dx,a[3]+dy]

    @staticmethod
    def scan(frame,detector,decode,region=None):
        h,w=frame.shape[:2]
        ox,oy=0,0
        view=frame
        if region is not None:
            ox,oy,x2,y2=region
            view=frame[oy:y2,ox:x2]
        items=[]
        for source,result in detector.detect(view):
            for item in decode(result,view.shape[1],view.shape[0]):
                item=dict(item)
                if source=='scene':item['class']=target_class(item['class'])
                box=item.get('xyxy',[])
                if (item['class']!='cup' or len(box)!=4 or not all(math.isfinite(v) for v in box)
                        or not math.isfinite(item.get('confidence',0)) or box[2]-box[0]<4 or box[3]-box[1]<4):continue
                item['xyxy']=[box[0]+ox,box[1]+oy,box[2]+ox,box[3]+oy]
                x1,y1,x2,y2=item['xyxy']
                item['position']=dict(x=(x1+x2)/2-w/2,y=h/2-(y1+y2)/2)
                items.append(item)
        unique=[]
        for item in sorted(items,key=lambda i:-i['confidence']):
            if not any(iou(item['xyxy'],other['xyxy'])>=.6 for other in unique):unique.append(item)
        return unique

    def region(self):
        w,h=self.frame_size
        x1,y1,x2,y2=self.search_box if self.search_box is not None else self.anchor
        px=max((x2-x1)*1.2,w*.12);py=max((y2-y1)*1.2,h*.12)
        return max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),min(w,math.ceil(x2+px)),min(h,math.ceil(y2+py))

    def correction(self,frame,candidates,predicted):
        w,h=self.frame_size
        reference=predicted['xyxy'] if predicted is not None else self.search_box
        if reference is None:reference=self.anchor
        recent=self.motion_box if self.motion_box is not None else self.anchor
        eligible=[c for c in candidates if c['confidence']>=self.lock_confidence]
        if not eligible:self.reject_reason='局部未检出杯子或分数不足';return None
        located=[c for c in eligible if nearby(reference,c['xyxy'],w,h) or nearby(recent,c['xyxy'],w,h)]
        if not located:self.reject_reason='候选位置或大小超限';return None
        matches=[(iou(reference,c['xyxy'])+.5*iou(recent,c['xyxy']),c) for c in located
                 if self.visual_appearance(frame,c['xyxy'])]
        matches.sort(key=lambda pair:-pair[0])
        if not matches:self.reject_reason='候选外观与原目标不符';return None
        if len(matches)>1 and matches[0][0]-matches[1][0]<.2:
            self.reject_reason='多个相似杯子，身份有歧义';return None
        self.reject_reason='通过'
        return matches[0][1]

    def step(self,frame,controller,detector,decode,observed_at):
        new_trial=controller.frame_size is None
        controller.observe(frame,observed_at)
        h,w=frame.shape[:2]
        if controller.selected_id is None:
            if self.tracker is not None or new_trial:self.clear()
            candidates=[i for i in self.scan(frame,detector,decode) if i['confidence']>=self.lock_confidence]
            now=time.monotonic()
            matches=[] if self.pending is None else [i for i in candidates if nearby(self.pending['xyxy'],i['xyxy'],w,h)
                                                     and iou(self.pending['xyxy'],i['xyxy'])>=.35]
            if matches and self.pending_at is not None and now-self.pending_at<=2:
                chosen=max(matches,key=lambda i:iou(self.pending['xyxy'],i['xyxy']))
                self.pending_count+=1
            elif candidates:
                chosen=max(candidates,key=lambda i:(i['confidence'],-((i['position']['x']/w)**2+(i['position']['y']/h)**2)))
                self.pending_count=1
            else:chosen=None;self.pending_count=0
            self.pending=chosen;self.pending_at=now
            self.status=f'水杯确认 {self.pending_count}/{self.confirm_frames} 帧，锁定置信度 ≥{self.lock_confidence:.2f}。'
            if chosen is None or self.pending_count<self.confirm_frames:
                controller.update_detections([],completed_at=now)
                return
            chosen=dict(chosen,track_id='object:confirmed:1')
            controller.update_detections([chosen],completed_at=now)
            if controller.target is not None:
                self.start(frame,controller.target)
                self.anchor=list(chosen['xyxy'])
                self.appearance=self.patch(frame,self.anchor)
                self.last_check=now;self.verified=True
                self.remember_motion(frame,self.anchor,now)
                self.tracking_source='CSRT'
                self.status='已连续确认水杯，等待1秒后启用反馈。'
            return
        if (w,h)!=self.frame_size:
            self.status='画面尺寸改变，暂停反馈；请按R重新锁定。'
            self.verified=False
            controller.update_detections([],completed_at=time.monotonic())
            return
        predicted=self.update(frame)
        now=time.monotonic()
        # Validate against the last current-image location, not a stale semantic anchor.
        previous=self.motion_box if self.motion_box is not None else self.anchor
        if predicted is not None:
            box=predicted['xyxy']
            bw,bh=previous[2]-previous[0],previous[3]-previous[1]
            dx=abs((box[0]+box[2]-previous[0]-previous[2])/2)
            dy=abs((box[1]+box[3]-previous[1]-previous[3])/2)
            if (dx>min(w*.18,bw*1.6) or dy>min(h*.18,bh*1.6)
                    or not .6<=(box[2]-box[0])/bw<=1.67 or not .6<=(box[3]-box[1])/bh<=1.67
                    or not self.visual_appearance(frame,box)):
                predicted=None
        self.tracking_source='CSRT' if predicted is not None else 'none'
        if self.motion_at is not None and now-self.motion_at<=self.MOTION_GRACE:
            flow=flow_box(self.motion_frame,cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),previous)
            if flow is not None and self.visual_appearance(frame,flow):
                if predicted is None or iou(flow,predicted['xyxy'])<.35:
                    predicted=dict(self.template,xyxy=flow,
                        position=dict(x=(flow[0]+flow[2])/2-w/2,y=h/2-(flow[1]+flow[3])/2))
                    self.tracking_source='optical flow'
                    # Keep the next CSRT frame aligned with the independent motion evidence.
                    old=dict(self.template);reference=self.appearance
                    self.start(frame,predicted);self.template=old;self.appearance=reference
        self.search_box=list(predicted['xyxy']) if predicted is not None else self.motion_prediction(now)
        if now-self.last_check>=self.VERIFY_INTERVAL:
            candidates=self.scan(frame,detector,decode,self.region())
            self.last_check=time.monotonic()
            matched=self.correction(frame,candidates,predicted)
            if matched is None:
                # Semantic misses alone never expire reliable current-image tracking.
                self.status=f'校正未通过：{self.reject_reason}。'
                if self.reject_reason=='多个相似杯子，身份有歧义':self.ambiguous=True
            else:
                # Keep the ORIGINAL identity and appearance, update only geometry.
                original=dict(self.template)
                original.update(xyxy=matched['xyxy'],position=matched['position'])
                reference=self.appearance
                if predicted is None or iou(predicted['xyxy'],matched['xyxy'])<.65:
                    self.start(frame,original)
                self.appearance=reference
                self.anchor=list(matched['xyxy'])
                predicted=dict(original)
                self.verified=True
                self.ambiguous=False
                self.tracking_source='local correction'
                self.status='视觉跟踪＋局部校正，保持原水杯锁定。'
        if self.ambiguous:predicted=None
        if predicted is None:
            self.status=f'当前图像无法可靠定位，反馈暂停；校正原因：{self.reject_reason}。'
            controller.update_detections([],completed_at=time.monotonic())
        else:
            self.remember_motion(frame,predicted['xyxy'],now)
            self.status=(f'{self.tracking_source}持续定位原物体；杯子校正未通过但当前视觉跟踪有效：{self.reject_reason}。'
                         if self.reject_reason!='通过' and self.reject_reason!='尚未锁定' else
                         f'{self.tracking_source}持续跟随原目标。')
            controller.update_detections([predicted],completed_at=time.monotonic())
