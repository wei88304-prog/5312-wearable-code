"""Detect cups until the first lock; thereafter use CSRT, never YOLO re-detection."""
import math
import time

import cv2

from sony_targets import target_class


class VisualObjectLock:
    def __init__(self, factory=None):
        self.factory = factory or getattr(cv2, "TrackerCSRT_create", None)
        if self.factory is None:
            raise RuntimeError("缺少CSRT物体跟踪扩展，请安装opencv-contrib-python。")
        self.clear()

    def clear(self):
        self.tracker = None
        self.template = None
        self.frame_size = None
        self.last_box = None

    def start(self, frame, target):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = target["xyxy"]
        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        if x2-x1 < 4 or y2-y1 < 4:
            raise RuntimeError("检测框过小，无法初始化物体跟踪；请靠近目标后按R。")
        tracker = self.factory()
        result = tracker.init(frame, (x1, y1, x2-x1, y2-y1))
        if result is False:
            raise RuntimeError("物体跟踪初始化失败，请调整画面后按R。")
        self.tracker = tracker
        self.template = dict(target)
        self.template["initial_confidence"] = target["confidence"]
        self.template["tracking_method"] = "CSRT"
        self.frame_size = (width, height)
        self.last_box = [x1,y1,x2,y2]

    def update(self, frame):
        height, width = frame.shape[:2]
        if self.tracker is None or (width, height) != self.frame_size:
            return None
        ok, box = self.tracker.update(frame)
        if not ok or len(box) != 4 or not all(math.isfinite(v) for v in box):
            return None
        x,y,w,h = box
        if w < 4 or h < 4:
            return None
        left,top,right,bottom = max(0,x),max(0,y),min(width,x+w),min(height,y+h)
        if max(0,right-left)*max(0,bottom-top) < w*h*.8:
            return None
        result = dict(self.template)
        result["xyxy"] = [float(left),float(top),float(right),float(bottom)]
        result["position"] = dict(x=(left+right)/2-width/2, y=height/2-(top+bottom)/2)
        self.last_box = result["xyxy"]
        return result

    def step(self, frame, controller, detector, decode, observed_at):
        """Only the pre-lock branch is allowed to call semantic detection."""
        controller.observe(frame, observed_at)
        height,width = frame.shape[:2]
        if controller.selected_id is None:
            self.clear()
            items = []
            for source,result in detector.detect(frame):
                for index,item in enumerate(decode(result,width,height)):
                    if source == "scene":
                        item["source_class"] = item["class"]
                        item["class"] = target_class(item["class"])
                    if item["class"] != "cup":
                        continue
                    item["track_id"] = f"object:{source}:{index}"
                    items.append(item)
            controller.update_detections(items, completed_at=time.monotonic())
            if controller.target is not None:
                self.start(frame,controller.target)
                controller.target.update(tracking_method="CSRT",
                                         initial_confidence=controller.target["confidence"])
        else:
            # Tracker failure preserves the selected identity. Do NOT scan again.
            item = self.update(frame)
            controller.update_detections([] if item is None else [item], completed_at=time.monotonic())
