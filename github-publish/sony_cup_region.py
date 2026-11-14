"""Full-box cup containment, adapted from upstream iphone_camera/cup_target.py.

Sony keeps explicit track selection instead of switching to the nearest cup.
The upstream 500x750 portrait region is uniformly scaled from 1920x1080.
This is an experimental target region, NOT an annotated ground-truth box.
"""
import math


def cup_region(target, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions")
    scale = min(width / 1920, height / 1080)
    half_width, half_height = 250 * scale, 375 * scale
    bounds = [width / 2 - half_width, height / 2 - half_height,
              width / 2 + half_width, height / 2 + half_height]
    result = dict(status="no_cup", inside=False, target=target,
                  roi_xyxy=bounds, criterion="entire_cup_box_inside_roi",
                  directions=dict(up=False, right=False, down=False, left=False),
                  instruction="等待锁定的水杯；停止震动")
    if target is None or target.get("class") != "cup":
        return result
    box = target.get("xyxy", [])
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return result
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return result
    left, top, right, bottom = bounds
    inside = left <= x1 and top <= y1 and x2 <= right and y2 <= bottom
    result.update(status="inside" if inside else "outside", inside=inside)
    if inside:
        result["instruction"] = "整个水杯框已入区，停止震动，可以拍照。"
    elif x2 - x1 > right - left or y2 - y1 > bottom - top:
        result.update(status="too_large", instruction="水杯框大于目标区，请后退或缩小画面；停止方向震动。")
    else:
        directions = dict(up=y1 < top, right=x2 > right, down=y2 > bottom, left=x1 < left)
        result["directions"] = directions
        names = dict(up="上", right="右", down="下", left="左")
        result["instruction"] = "相机向" + "、".join(names[k] for k, v in directions.items() if v) + "调整，让整个杯子框入区。"
    return result
