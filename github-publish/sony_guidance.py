"""Stable-scene selection and text guidance in native image coordinates."""

import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TARGET_NAMES = {"person": "人", "tree": "树", "cup": "水杯"}


class StabilityGate:
    # Limits apply against the fixed start-of-window frame, not the previous
    # frame: handheld wobble can settle, but a slow pan still restarts the timer.
    MAX_SHIFT_FRACTION = 0.08
    MAX_MEAN_DIFFERENCE = 22.0
    CHANGED_PIXEL_DIFFERENCE = 44
    MAX_CHANGED_FRACTION = 0.28

    def __init__(self, seconds=1.5):
        self.seconds = seconds
        self.window = cv2.createHanningWindow((160, 96), cv2.CV_32F)
        self.reset()

    def reset(self):
        self.anchor = None
        self.since = None
        self.last_time = None
        self.elapsed = 0.0

    def _scene_changed(self, gray):
        height, width = gray.shape
        # phaseCorrelate applies its window in place; copies keep the reference
        # intact throughout the countdown and while candidates are displayed.
        (dx, dy), response = cv2.phaseCorrelate(self.anchor.copy(), gray.copy(), self.window)
        reference = self.anchor
        if response >= 0.5 and math.isfinite(dx) and math.isfinite(dy):
            if abs(dx) > width * self.MAX_SHIFT_FRACTION or abs(dy) > height * self.MAX_SHIFT_FRACTION:
                return True
            gray = cv2.warpAffine(gray, np.float32([[1, 0, -dx], [0, 1, -dy]]), (width, height))
            # Ignore only the thin border exposed by this bounded alignment.
            margin_x = math.ceil(width * self.MAX_SHIFT_FRACTION) + 1
            margin_y = math.ceil(height * self.MAX_SHIFT_FRACTION) + 1
            gray = gray[margin_y:-margin_y, margin_x:-margin_x]
            reference = reference[margin_y:-margin_y, margin_x:-margin_x]
        # Low-texture scenes cannot provide a reliable motion estimate. In that
        # case compare the unaligned image using the same relaxed noise limits.
        delta = gray - reference
        difference = np.abs(delta - np.clip(np.median(delta), -6, 6))
        return (float(difference.mean()) > self.MAX_MEAN_DIFFERENCE
                or float((difference > self.CHANGED_PIXEL_DIFFERENCE).mean()) > self.MAX_CHANGED_FRACTION)

    def observe(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(cv2.resize(gray, (160, 96)), (5, 5), 0).astype(np.float32)
        restart = self.anchor is None or self.last_time is None or now - self.last_time > 0.75
        if not restart:
            restart = self._scene_changed(gray)
        if restart:
            self.anchor = gray
            self.since = now
        self.last_time = now
        self.elapsed = max(0.0, now - self.since)
        return self.elapsed + 1e-9 >= self.seconds


def movement_guidance(position, width, height, tolerance=0.03, was_centered=False):
    x, y = position["x"], position["y"]
    limit_x, limit_y = max(8, width * tolerance), max(8, height * tolerance)
    factor = 1.3 if was_centered else 1.0
    centered = abs(x) <= limit_x * factor and abs(y) <= limit_y * factor
    distance = round(math.hypot(x, y), 1)
    if centered:
        return {"centered": True, "distance_px": distance,
                "text": "目标已居中，请保持相机稳定，可以拍照。", "horizontal": "水平已对齐", "vertical": "垂直已对齐"}

    def hint(offset, dimension, limit, positive, negative, axis):
        if abs(offset) <= limit:
            return f"{axis}已对齐"
        fraction = abs(offset) / dimension
        degree = "较大幅度" if fraction >= 0.25 else "适度" if fraction >= 0.10 else "小幅"
        direction = positive if offset > 0 else negative
        return f"{degree}{direction}（偏移 {abs(offset):.1f} px，占画面{'宽' if axis == '水平' else '高'} {fraction:.1%}）"

    horizontal = hint(x, width, limit_x, "向右转动相机", "向左转动相机", "水平")
    vertical = hint(y, height, limit_y, "向上抬起镜头", "向下压低镜头", "垂直")
    return {"centered": False, "distance_px": distance,
            "text": "请缓慢调整取景，让已锁定目标靠近中心。", "horizontal": horizontal, "vertical": vertical}


class GuidanceController:
    INVENTORY_INTERVAL = 0.75

    def __init__(self, stable_seconds=1.5, tolerance=0.03, cup_target=False, auto_target=False):
        self.cup_target = cup_target
        self.auto_target = auto_target
        self.auto_delay = 1.0
        self.gate = StabilityGate(stable_seconds)
        self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.gate.reset()
        self.phase = "detecting" if self.auto_target else "stabilizing"
        self.lock_started = None
        self.feedback_enabled = False
        self.lost_since = None
        self.last_detection_at = None
        self.observed_at = None
        self.frame_size = None
        self.candidates = []
        self.target = None
        self.selected_id = None
        self.selected_class = None
        self.selected_number = None
        self.centered = False
        self.last_scan_at = None
        self.numbers = {}
        self.next_number = 1

    def observe(self, frame, now):
        self.observed_at = now
        self.frame_size = (frame.shape[1], frame.shape[0])
        if self.selected_id is not None:
            return True  # Selection removes the stability wait entirely.
        if self.auto_target:
            return True  # Detect now; the 1s delay comes AFTER a cup is locked.
        stable = self.gate.observe(frame, now)
        if not stable:
            self.phase = "stabilizing"
            self.candidates = []
            self.last_scan_at = None
            self.numbers = {}
            self.next_number = 1
            return False
        if self.phase == "stabilizing":
            self.phase = "detecting"
            return True
        return (self.phase in ("choose", "empty") and self.last_scan_at is not None
                and now - self.last_scan_at >= self.INVENTORY_INTERVAL)

    def _distinct_items(self, items):
        # The two models can detect the same cup. Prefer an already numbered
        # track (especially the selected one) so refreshing cannot replace it.
        items.sort(key=lambda item: (item["track_id"] != self.selected_id,
                                     item["track_id"] not in self.numbers, -item["confidence"]))
        distinct = []
        for item in items:
            duplicate = False
            if item["class"] == "cup":
                x1, y1, x2, y2 = item["xyxy"]
                for other in distinct:
                    if (other["class"] != "cup" or other["track_id"].partition(":")[0]
                            == item["track_id"].partition(":")[0]):
                        continue
                    a1, b1, a2, b2 = other["xyxy"]
                    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
                    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
                    if union > 0 and intersection / union >= 0.5:
                        duplicate = True
                        break
            if not duplicate:
                distinct.append(item)
        return distinct

    def update_detections(self, detections, completed_at=None):
        now = completed_at if completed_at is not None else self.observed_at
        if now is None:
            now = time.monotonic()
        self.last_detection_at = now
        items = [dict(item) for item in detections
                 if item["class"] in TARGET_NAMES and item.get("track_id") is not None
                 and (not self.cup_target or item["class"] == "cup")]
        items = self._distinct_items(items)
        if self.selected_id is not None:
            self.target = next((item for item in items if item["track_id"] == self.selected_id
                                and item["class"] == self.selected_class), None)
            self.phase = "tracking" if self.target else "lost"
            if self.target:
                self.lost_since = None
                self.target["number"] = self.selected_number
                if self.auto_target:
                    if self.lock_started is None and not self.feedback_enabled:
                        self.lock_started = now
                    if not self.feedback_enabled and now - self.lock_started >= self.auto_delay:
                        self.feedback_enabled = True
                    self.phase = "tracking" if self.feedback_enabled else "arming"
            else:
                self.centered = False
                if not self.feedback_enabled:
                    self.lock_started = None
                if self.lost_since is None:
                    self.lost_since = now
                # A lock is released only by an explicit reset, never a timeout.
                # Missing observations cannot drive motors using old coordinates.
            return
        order = {name: index for index, name in enumerate(TARGET_NAMES)}
        self.candidates = sorted(items, key=lambda item: (order[item["class"]], item["xyxy"][0]))
        for item in self.candidates:
            track_id = item["track_id"]
            if track_id not in self.numbers:
                self.numbers[track_id] = self.next_number
                self.next_number += 1
            item["number"] = self.numbers[track_id]
        self.candidates.sort(key=lambda item: item["number"])
        self.phase = "choose" if self.candidates else "empty"
        if self.auto_target and self.candidates:
            width, height = self.frame_size
            selected = min(self.candidates, key=lambda item: (
                (item["position"]["x"] / width) ** 2 + (item["position"]["y"] / height) ** 2,
                -item["confidence"], item["number"]))
            self.select(selected["number"])
            self.target = selected
            self.lock_started = now
            self.phase = "arming"
        self.last_scan_at = self.gate.last_time if completed_at is None else completed_at
        if completed_at is not None:
            # The initial stable window is already satisfied. Model computation
            # is not a fresh stability window; the next image still has to match
            # the fixed anchor before these choices remain valid.
            self.gate.last_time = completed_at

    def select(self, number):
        if self.phase != "choose":
            return False
        item = next((item for item in self.candidates if item["number"] == number), None)
        if item is None:
            return False
        self.selected_id = item["track_id"]
        self.selected_class = item["class"]
        self.selected_number = number
        # The choice frame is not a fresh tracking observation. Wait for the
        # next inference instead of issuing directions from an old snapshot.
        self.target = None
        self.phase = "tracking"
        return True

    def snapshot(self, width, height):
        counts = {name: sum(item["class"] == name for item in self.candidates) for name in TARGET_NAMES}
        messages = {
            "stabilizing": f"请保持画面稳定：{min(self.gate.elapsed, self.gate.seconds):.1f} / {self.gate.seconds:.1f} 秒",
            "detecting": "画面已稳定，正在识别人、树、水杯……",
            "choose": "请选择想拍摄的具体对象：点击画面中的框或右侧条目。",
            "empty": "暂未检测到人、树或水杯，保持取景会自动重试；也可调整取景或按 R。",
            "tracking": "已锁定目标，正在读取它的最新位置……",
            "lost": "已锁定目标暂时丢失。请让原目标回到画面，或按 R 重新选择。",
            "arming": f"已自动锁定水杯，等待 {max(0, self.auto_delay - ((self.last_detection_at or 0) - (self.lock_started or 0))):.1f} 秒后启用震动。",
        }
        if self.auto_target:
            messages["lost"] = "原水杯暂时丢失，保留锁定并停止反馈；只找回原目标，不换杯子。需要换目标才按R。"
        guidance = None
        if self.target is not None:
            guidance = movement_guidance(self.target["position"], width, height, self.tolerance, self.centered)
            self.centered = guidance["centered"]
            messages["tracking"] = "正在跟踪已锁定对象，请按下方文字提示调整相机。"
        region = None
        if self.cup_target:
            from sony_cup_region import cup_region
            region = cup_region(self.target, width, height)
            messages["detecting"] = "画面已稳定，正在识别水杯……"
            if self.auto_target:
                messages["detecting"] = "正在识别水杯，找到后自动锁定，无需点击。"
            messages["choose"] = "点击选择一只水杯；整个水杯检测框入中央区才算完成。"
            messages["empty"] = "未检测到水杯；持续扫描中，找到后自动锁定，无需按R。" if self.auto_target else "未检测到水杯；保持取景自动重试，不用其他物体替代。"
            if self.target is not None:
                self.centered = region["inside"]
                guidance.update(centered=self.centered, text=region["instruction"],
                                horizontal="", vertical="")
                messages["tracking"] = "跟踪已锁定的同一只水杯；中央框是目标区域，并非人工标注真值。"
        return {"phase": self.phase, "message": messages[self.phase],
                "stable_seconds": round(self.gate.elapsed, 2), "required_seconds": self.gate.seconds,
                "counts": counts, "candidates": self.candidates if self.phase == "choose" else [],
                "selected_track_id": self.selected_id, "selected_class": self.selected_class,
                "target": self.target, "guidance": guidance, "cup_guidance": region,
                "auto_target": self.auto_target}

    def display_items(self):
        if self.selected_id is not None:
            return [self.target] if self.target else []
        return self.candidates if self.phase == "choose" else []


class GuidancePanel:
    PAGE_SIZE = 7

    def __init__(self, font_path=None):
        paths = [font_path] if font_path else [
            "/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
        path = next((path for path in paths if path and Path(path).is_file()), None)
        if path is None:
            raise RuntimeError("没有找到中文字体，请用 --font 指定本地中文字体。")
        self.title_font = ImageFont.truetype(path, 26)
        self.text_font = ImageFont.truetype(path, 20)
        self.small_font = ImageFont.truetype(path, 16)
        self.hit_regions = []
        self.page = 0
        self.pending_selection = None

    def mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        matches = [(area, number) for left, top, right, bottom, number, area in self.hit_regions
                   if left <= x <= right and top <= y <= bottom]
        if matches:
            self.pending_selection = min(matches)[1]

    def take_selection(self):
        number, self.pending_selection = self.pending_selection, None
        return number

    def render(self, annotated_frame, state, fps=0):
        canvas = Image.new("RGB", (1280, 760), "#111b28")
        draw = ImageDraw.Draw(canvas)
        title = "Sony 水杯对准 · 中央目标区＋三个Case" if state.get("cup_guidance") else "Sony 拍摄引导 · 人 / 树 / 水杯"
        draw.text((20, 16), title, font=self.title_font, fill="#edf3fa")
        selection_hint = "自动锁定 · 等待1秒" if state.get("auto_target") else "点击选择"
        draw.text((20, 57), f"{fps:.1f} FPS    {selection_hint}    R 重新选择    N/P 翻页    Q/Esc 退出",
                  font=self.small_font, fill="#b1c0d3")
        height, width = annotated_frame.shape[:2]
        scale = min(790 / width, 640 / height)
        display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        left, top = 20 + (790 - display_size[0]) // 2, 94 + (640 - display_size[1]) // 2
        video = Image.fromarray(cv2.cvtColor(cv2.resize(annotated_frame, display_size), cv2.COLOR_BGR2RGB))
        canvas.paste(video, (left, top))
        region = state.get("cup_guidance")
        if region:
            x1, y1, x2, y2 = region["roi_xyxy"]
            color = "#64f064" if region["inside"] else "#ffd166"
            draw.rectangle((left + x1 * scale, top + y1 * scale,
                            left + x2 * scale, top + y2 * scale), outline=color, width=3)
            draw.text((left + x1 * scale + 4, top + y1 * scale + 4), "TARGET ROI",
                      font=self.small_font, fill=color, stroke_width=1, stroke_fill="#111b28")
        draw.rounded_rectangle((830, 16, 1260, 744), radius=14, fill="#1c2b40")
        self.hit_regions = []

        def paragraph(text, y, color="#edf3fa", font=None):
            font = font or self.text_font
            line = ""
            for character in text:
                if draw.textlength(line + character, font=font) > 380:
                    draw.text((850, y), line, font=font, fill=color)
                    y += 29
                    line = ""
                line += character
            if line:
                draw.text((850, y), line, font=font, fill=color)
                y += 29
            return y

        titles = {"stabilizing": "① 保持画面稳定", "detecting": "② 识别候选对象", "choose": "③ 选择拍摄对象",
                  "empty": "② 未找到候选对象", "tracking": "④ 实时调整取景", "lost": "④ 请找回原目标",
                  "arming": "③ 已锁定 · 等待1秒"}
        draw.text((850, 36), titles[state["phase"]], font=self.title_font, fill="#7ee4c4")
        y = paragraph(state["message"], 87)
        if state.get("haptics"):
            y = paragraph(f"振动：{state['haptics']['text']}", y + 6, "#7ee4c4", self.small_font)
        if state["phase"] == "stabilizing":
            fraction = min(1, state["stable_seconds"] / state["required_seconds"])
            draw.rounded_rectangle((850, y + 12, 1230, y + 24), radius=5, fill="#34465e")
            if fraction > 0:
                draw.rounded_rectangle((850, y + 12, 850 + round(380 * fraction), y + 24), radius=5, fill="#7ee4c4")
        if state["phase"] in ("choose", "empty"):
            counts = state["counts"]
            labels = ("cup",) if region else TARGET_NAMES
            y = paragraph("  ".join(f"{TARGET_NAMES[name]}：{counts[name]}" for name in labels), y + 16)
        candidates = state["candidates"]
        pages = max(1, math.ceil(len(candidates) / self.PAGE_SIZE))
        self.page = min(max(self.page, 0), pages - 1)
        if candidates:
            for row, item in enumerate(candidates[self.page * self.PAGE_SIZE:(self.page + 1) * self.PAGE_SIZE]):
                row_y = y + 16 + row * 45
                text = f"#{item['number']}  {TARGET_NAMES[item['class']]}   ({item['position']['x']:+.0f}, {item['position']['y']:+.0f})"
                draw.rounded_rectangle((846, row_y, 1240, row_y + 38), radius=5, fill="#2b405a")
                draw.text((858, row_y + 5), text, font=self.text_font, fill="#edf3fa")
                self.hit_regions.append((846, row_y, 1240, row_y + 38, item["number"], -1))
            paragraph(f"第 {self.page + 1}/{pages} 页 · N/P 翻页", y + 16 + min(len(candidates), self.PAGE_SIZE) * 45,
                      font=self.small_font)
            for item in candidates:
                x1, y1, x2, y2 = item["xyxy"]
                self.hit_regions.append((left + x1 * scale, top + y1 * scale, left + x2 * scale,
                                         top + y2 * scale, item["number"], (x2 - x1) * (y2 - y1)))
                draw.text((left + x1 * scale + 3, top + y1 * scale + 3), f"#{item['number']}",
                          font=self.text_font, fill="#7ee4c4", stroke_width=1, stroke_fill="#111b28")
        if state["selected_class"] is not None:
            y = paragraph(f"已锁定：{TARGET_NAMES[state['selected_class']]}", y + 20, "#7ee4c4")
        if state["target"] is not None:
            position = state["target"]["position"]
            guidance = state["guidance"]
            y = paragraph(f"X = {position['x']:+.1f} px    Y = {position['y']:+.1f} px", y + 10)
            y = paragraph(f"距中心：{guidance['distance_px']:.1f} px", y)
            y = paragraph(guidance["text"], y + 18, "#7ee4c4" if guidance["centered"] else "#ffd18a")
            if not guidance["centered"] and not region:
                y = paragraph(guidance["horizontal"], y + 14)
                paragraph(guidance["vertical"], y + 10)
        draw.line((850, 650, 1240, 650), fill="#41536a")
        draw.text((850, 666), "原点：取景框中心 (0, 0)", font=self.small_font, fill="#b1c0d3")
        draw.text((850, 691), "X 向右，Y 向上；偏移单位为像素。", font=self.small_font, fill="#b1c0d3")
        draw.text((850, 716), "慢慢转动相机；靠近中心时缩小幅度。", font=self.small_font, fill="#b1c0d3")
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
