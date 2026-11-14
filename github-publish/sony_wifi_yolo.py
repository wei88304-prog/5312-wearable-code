"""Read Sony Smart Remote Liveview over Wi-Fi and detect objects locally."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
(ROOT / ".yolo-config" / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".yolo-config")

import cv2
import numpy as np
import requests


def camera_from_description(xml):
    root = ET.fromstring(xml)
    name = root.findtext(".//{*}friendlyName") or "Sony camera"
    for service in root.findall(".//{*}X_ScalarWebAPI_Service"):
        if service.findtext("{*}X_ScalarWebAPI_ServiceType") == "camera":
            base = service.findtext("{*}X_ScalarWebAPI_ActionList_URL")
            if base:
                return name, base.rstrip("/") + "/camera"
    raise RuntimeError("设备描述没有 Sony camera API。")


def discover(timeout=4):
    request = "\r\n".join([
        "M-SEARCH * HTTP/1.1", "HOST: 239.255.255.250:1900",
        'MAN: "ssdp:discover"', "MX: 1",
        "ST: urn:schemas-sony-com:service:ScalarWebAPI:1", "", "",
    ]).encode()
    locations = set()
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(0.5)
        sock.sendto(request, ("239.255.255.250", 1900))
        while time.monotonic() < deadline:
            try:
                response, _ = sock.recvfrom(8192)
            except socket.timeout:
                continue
            for line in response.decode(errors="replace").splitlines():
                key, separator, value = line.partition(":")
                if separator and key.lower() == "location":
                    locations.add(value.strip())
    cameras = {}
    with requests.Session() as session:
        session.trust_env = False  # Camera LAN traffic should bypass proxy variables.
        for location in sorted(locations):
            try:
                response = session.get(location, timeout=3)
                response.raise_for_status()
                name, endpoint = camera_from_description(response.content)
                cameras[endpoint] = name
            except (requests.RequestException, ET.ParseError, RuntimeError) as error:
                print(f"跳过无法读取的设备描述：{error}", file=sys.stderr)
    return [(name, endpoint) for endpoint, name in cameras.items()]


class SonyCamera:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.trust_env = False
        self.request_id = 0
        self.started_mode = False
        self.started_liveview = False

    def call(self, method):
        self.request_id += 1
        response = self.session.post(self.endpoint, json={
            "method": method, "params": [], "id": self.request_id, "version": "1.0",
        }, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"相机拒绝 {method}：{data['error']}")
        if "result" not in data:
            raise RuntimeError(f"相机返回了无效的 {method} 响应。")
        return data["result"]

    def start(self):
        available = self.call("getAvailableApiList")[0]
        if "startRecMode" in available:
            self.call("startRecMode")
            self.started_mode = True
        deadline = time.monotonic() + 15
        while "startLiveview" not in available:
            if time.monotonic() >= deadline:
                raise RuntimeError("相机未开放 startLiveview，请检查智能遥控状态。")
            time.sleep(0.5)
            available = self.call("getAvailableApiList")[0]
        stream_url = self.call("startLiveview")[0]
        self.started_liveview = True
        return stream_url

    def close(self):
        for should_stop, method in [
            (self.started_liveview, "stopLiveview"),
            (self.started_mode, "stopRecMode"),
        ]:
            if should_stop:
                try:
                    self.call(method)
                except (requests.RequestException, RuntimeError, ValueError) as error:
                    print(f"退出时 {method} 未成功：{error}", file=sys.stderr)
        self.session.close()


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("相机画面流已结束或连接断开。")
        data.extend(chunk)
    return bytes(data)


def jpeg_frames(stream):
    """Sony framing: 16-byte common header, 120-byte metadata, payload, padding."""
    while True:
        header = read_exact(stream, 16)
        if header[0] != 0xFF or header[8:12] != b"\x24\x35\x68\x79":
            raise RuntimeError("收到的不是有效 Sony Liveview 帧。")
        size = int.from_bytes(header[12:15], "big")
        if size > 16 * 1024 * 1024:
            raise RuntimeError("Sony 帧大小异常。")
        read_exact(stream, 120)
        payload = read_exact(stream, size)
        read_exact(stream, header[15])
        if header[1] in (0x01, 0x11):
            yield payload


class LatestFrame:
    """Drain the camera stream continuously; retain only the newest decoded frame."""
    def __init__(self, url):
        self.url = url
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def _read(self):
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(self.url, stream=True, timeout=(5, 8)) as response:
                    response.raise_for_status()
                    for jpeg in jpeg_frames(response.raw):
                        if self.stopped.is_set():
                            break
                        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.condition:
                                self.frame = frame
                                self.sequence += 1
                                self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def get(self, previous_sequence):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > previous_sequence or self.error is not None,
                timeout=12,
            )
            if self.sequence > previous_sequence:
                return self.sequence, self.frame
            if self.error is not None:
                raise RuntimeError(f"读取相机画面失败：{self.error}") from self.error
            if not ready:
                raise TimeoutError("12 秒内没有收到新画面。")

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=9)


def frame_position(xyxy, width, height):
    """Bounding-box center in native-frame pixels, with +X right and +Y up."""
    x1, y1, x2, y2 = xyxy
    return {
        "x": round((x1 + x2) / 2 - width / 2, 1),
        "y": round(height / 2 - (y1 + y2) / 2, 1),
    }


def detection_items(result, width, height):
    items = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        items.append({
            "class": result.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 3),
            "xyxy": xyxy,
            "position": frame_position(xyxy, width, height),
        })
    return items


def draw_label(frame, text, x, y, color=(255, 255, 255), scale=0.45, occupied=None):
    """Draw readable text and keep the label inside the image."""
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(int(x), frame.shape[1] - width - 8))
    y = max(0, min(int(y), frame.shape[0] - height - baseline - 8))
    label_height = height + baseline + 8
    if occupied is not None:
        candidates = [y, *range(y - label_height, -1, -label_height),
                      *range(y + label_height, frame.shape[0] - label_height + 1, label_height)]
        for candidate in candidates:
            if all(x + width + 8 <= left or x >= right or
                   candidate + label_height <= top or candidate >= bottom
                   for left, top, right, bottom in occupied):
                y = candidate
                break
        occupied.append((x, y, x + width + 8, y + label_height))
    cv2.rectangle(frame, (x, y), (x + width + 8, y + height + baseline + 8), (15, 15, 15), -1)
    cv2.putText(frame, text, (x + 4, y + height + 4), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def annotate_frame(frame, detections):
    """Overlay a 100 px grid, centered axes, and each detection's coordinates."""
    height, width = frame.shape[:2]
    origin_x, origin_y = width / 2, height / 2
    center_x, center_y = round(origin_x), round(origin_y)
    grid = frame.copy()
    ticks = []
    for offset in range(100, max(width, height), 100):
        for sign in (-1, 1):
            x = round(origin_x + sign * offset)
            y = round(origin_y - sign * offset)
            if 0 < x < width - 1:
                cv2.line(grid, (x, 0), (x, height - 1), (180, 180, 180), 1)
                ticks.append((f"{sign * offset:+d}", x - 16, center_y + 6))
            if 0 < y < height - 1:
                cv2.line(grid, (0, y), (width - 1, y), (180, 180, 180), 1)
                ticks.append((f"{sign * offset:+d}", center_x + 6, y - 10))
    output = cv2.addWeighted(frame, 0.75, grid, 0.25, 0)
    axis_color = (255, 210, 0)
    cv2.arrowedLine(output, (0, center_y), (width - 1, center_y), axis_color, 1, tipLength=0.02)
    cv2.arrowedLine(output, (center_x, height - 1), (center_x, 0), axis_color, 1, tipLength=0.02)
    for text, x, y in ticks:
        draw_label(output, text, x, y, axis_color, scale=0.35)
    cv2.circle(output, (center_x, center_y), 3, axis_color, -1)
    draw_label(output, "(0,0)", center_x + 6, center_y + 6, axis_color)
    draw_label(output, "+X", width - 35, center_y - 25, axis_color)
    draw_label(output, "+Y", center_x + 6, 4, axis_color)
    draw_label(output, "Box center | X: right, Y: up | unit: px", 6, 6)

    labels = []
    for item in detections:
        x1, y1, x2, y2 = item["xyxy"]
        center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
        color = (80, 240, 100)
        cv2.rectangle(output, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 12, 2)
        position = item["position"]
        number = f"#{item['number']} " if "number" in item else ""
        score = f"CSRT init={item['initial_confidence']:.2f}" if item.get("tracking_method") == "CSRT" else f"{item['confidence']:.2f}"
        text = (f"{number}{item['class']} {score} "
                f"({position['x']:+.1f}, {position['y']:+.1f}) px")
        labels.append((text, x1, y1 - 25, color))
    occupied = []
    for text, x, y, color in labels:
        draw_label(output, text, x, y, color, occupied=occupied)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="α6000：水杯测试自动锁定并等待1秒；旧模式支持稳定后手动选择。")
    parser.add_argument("--discover", action="store_true", help="只列出相机，不启动取景")
    parser.add_argument("--api-url", help="手动指定相机 API 的完整 URL")
    parser.add_argument("--preview", action="store_true", help="只显示画面，不运行 YOLO")
    parser.add_argument("--model", default=str(ROOT / "yolo26n.pt"))
    parser.add_argument("--tree-model", default=str(ROOT / "yoloe-26n-seg-pf.pt"))
    parser.add_argument("--coordinates-only", action="store_true", help="使用原版逐帧全类别坐标模式")
    parser.add_argument("--stable-seconds", type=float, default=1.5, help="选择对象前需要连续稳定的秒数")
    parser.add_argument("--center-tolerance", type=float, default=0.03, help="居中容差占画面宽、高的比例")
    parser.add_argument("--target-number", type=int, help="无窗口模式下，稳定识别后选择指定候选编号")
    parser.add_argument("--font", help="中文提示使用的本地字体文件")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--confidence", type=float, default=0.35, help="人或旧版全类别模式的检测阈值")
    parser.add_argument("--cup-confidence", type=float, default=0.20, help="水杯检测阈值")
    parser.add_argument("--tree-confidence", type=float, default=0.20, help="树木检测阈值")
    parser.add_argument("--headless", action="store_true", help="不打开画面窗口")
    parser.add_argument("--frames", type=int, default=0, help="处理指定帧数后退出，0 表示持续运行")
    parser.add_argument("--haptics-port", help="启用 ESP32 四向振动：指定 USB 串口或 auto")
    parser.add_argument("--testing", action="store_true", help="默认Case2、gap0.3秒启动，并允许终端切换三个Case")
    parser.add_argument("--choose-case", action="store_true", help="启动时询问Case及间隔，同时启用--testing")
    parser.add_argument("--cup-target", action="store_true", help="只识别水杯，整个检测框进入中央目标区才完成；testing默认启用")
    parser.add_argument("--manual-target", action="store_true", help="水杯模式恢复原来的稳定等待与手动点击；默认自动锁定后等待1秒")
    parser.add_argument("--lock-confidence", type=float, default=0.30, help="连续确认及局部校正的水杯置信度阈值，默认0.30")
    parser.add_argument("--lock-frames", type=int, default=3, help="首次锁定需连续确认帧数，默认3，至少2")
    parser.add_argument("--haptics-case", type=int, choices=(1, 2, 3), help="Case1强度、Case2次数、Case3持续")
    parser.add_argument("--pulse-gap", type=float, choices=(0.1, 0.3, 0.5), default=0.3)
    parser.add_argument("--list-haptics-ports", action="store_true", help="只枚举串口，不打开相机或控制板")
    args = parser.parse_args(argv)
    if not 0 <= args.lock_confidence <= 1 or args.lock_frames < 2:
        parser.error("lock-confidence须在0至1，lock-frames至少2")
    if args.choose_case:
        args.testing = True
    if args.testing:
        args.cup_target = True
        if not sys.stdin.isatty() or args.headless:
            parser.error("--testing需要交互终端和图形界面")
        args.haptics_port = args.haptics_port or "auto"
    if args.choose_case:
        while True:
            choice = input("选择Case [1=强度 / 2=次数 / 3=持续，默认2]: ").strip() or "2"
            if choice in ("1", "2", "3"):
                args.haptics_case = int(choice)
                break
        if args.haptics_case == 2:
            while True:
                gap = input("脉冲间隔 [0.1 / 0.3 / 0.5秒，默认0.3]: ").strip() or "0.3"
                if gap in ("0.1", "0.3", "0.5"):
                    args.pulse_gap = float(gap)
                    break
    if args.haptics_port and args.haptics_case is None:
        args.haptics_case = 2
    if args.haptics_port and not (args.preview or args.discover or args.coordinates_only or args.list_haptics_ports):
        args.cup_target = True
    if args.haptics_case is not None and not args.haptics_port:
        parser.error("--haptics-case需要--haptics-port")
    if args.cup_target and (args.preview or args.discover or args.coordinates_only or args.list_haptics_ports):
        parser.error("--cup-target只适用于对象选择与引导模式")
    if (not all(0 <= value <= 1 for value in (args.confidence, args.cup_confidence, args.tree_confidence))
            or args.frames < 0 or args.stable_seconds <= 0
            or not 0 < args.center_tolerance <= 0.25):
        parser.error("检测阈值必须在 0 到 1 之间，frames 非负，stable-seconds 为正，center-tolerance 在 0 到 0.25 之间。")
    if args.target_number is not None and (not args.headless or args.target_number < 1):
        parser.error("target-number 需要 --headless，并且编号大于等于 1。")
    if args.target_number is not None and (args.preview or args.discover or args.coordinates_only):
        parser.error("target-number 只能用于默认拍摄引导模式。")
    if args.haptics_port and (args.preview or args.discover or args.coordinates_only or args.list_haptics_ports):
        parser.error("振动引导只能用于默认的对象选择与实时引导模式。")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_haptics_ports:
        from sony_haptics import list_ports
        print(json.dumps(list_ports(), ensure_ascii=False, indent=2))
        return 0

    model = None
    detector = controller = panel = None
    object_lock = None
    device = args.device
    if not args.discover and not args.preview:
        if not Path(args.model).is_file():
            raise RuntimeError(f"本地模型不存在：{args.model}。请先联网下载。")
        import torch
        from ultralytics import YOLO
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("当前环境无法使用 MPS，请使用 --device cpu。")
        if args.coordinates_only:
            model = YOLO(args.model)
        else:
            from sony_targets import TargetDetector, target_class
            from sony_guidance import GuidanceController, GuidancePanel
            use_object_lock = args.cup_target and not args.manual_target and args.target_number is None
            detector = TargetDetector(args.model, args.tree_model, device, args.confidence,
                                      args.cup_confidence, args.tree_confidence, cups_only=args.cup_target,
                                      scan_only=use_object_lock)
            controller = GuidanceController(args.stable_seconds, args.center_tolerance,
                                            cup_target=args.cup_target,
                                            auto_target=args.cup_target and not args.manual_target and args.target_number is None)
            if use_object_lock:
                from sony_verified_lock import VerifiedObjectLock
                object_lock = VerifiedObjectLock(lock_confidence=args.lock_confidence,
                                                 confirm_frames=args.lock_frames)
            if not args.headless:
                panel = GuidancePanel(args.font)

    cameras = [("Sony camera", args.api_url)] if args.api_url else discover()
    if args.discover:
        print(json.dumps([{"name": name, "api_url": url} for name, url in cameras], ensure_ascii=False, indent=2))
        return 0 if cameras else 1
    if not cameras:
        raise RuntimeError("未发现相机。请启动相机智能遥控，并让 Mac 连接其 DIRECT-… Wi-Fi。")
    if len(cameras) > 1:
        raise RuntimeError("发现多台相机，请先用 --discover 查看，再用 --api-url 指定。")

    name, endpoint = cameras[0]
    print(f"连接：{name}；推理设备：{device if model or detector else '仅预览'}", flush=True)
    camera = SonyCamera(endpoint)
    reader = None
    haptics = None
    sequence = count = 0
    last_display_time = None
    automatic_choice_used = False
    try:
        if args.haptics_port:
            from sony_haptics import SerialHaptics
            haptics = SerialHaptics(args.haptics_port, case=args.haptics_case,
                                    gap_ms=round(args.pulse_gap * 1000))
            print(f"四向振动已连接：{haptics.device}；上/右/下/左 = TCA 0/4/2/3。", flush=True)
            print(f"当前反馈：Case {haptics.case}；脉冲间隔 {haptics.gap_ms / 1000:.1f}秒。", flush=True)
        reader = LatestFrame(camera.start())
        reader.thread.start()
        if controller is not None:
            print("自动识别并锁定水杯，等待1秒后启用震动；R重新识别；Q/Esc退出。" if controller.auto_target else
                  "画面稳定后点击对象锁定；R 重新选择；Q、Esc 或 Ctrl+C 退出。", flush=True)
        else:
            print("按 Q、Esc 或 Ctrl+C 退出。", flush=True)
        if panel is not None:
            cv2.namedWindow("Sony Wi-Fi YOLO", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("Sony Wi-Fi YOLO", panel.mouse)
        commands = None
        paused = False
        last_preview_status = None
        if controller is not None and controller.auto_target and haptics is None:
            print(f"无硬件测试：杯子置信度≥{args.lock_confidence:.2f}且连续{args.lock_frames}帧确认→锁定等待1秒→CSRT＋局部校正。无法确认时暂停，不选远处杯子。不会打开串口。", flush=True)
        if args.testing:
            import queue
            commands = queue.Queue()
            def terminal_commands():
                for line in sys.stdin:
                    commands.put(line.strip().lower())
            threading.Thread(target=terminal_commands, daemon=True).start()
            print("终端输入：1/2/3切换Case；gap 0.1/0.3/0.5；0暂停；start继续；r重选；q退出。切换后会重新选目标：自动模式自动锁定，手动模式需点击。", flush=True)
        while True:
            if commands is not None:
                quit_requested = False
                while not commands.empty():
                    command = commands.get_nowait()
                    if command == "q":
                        quit_requested = True
                        break
                    if command in ("1", "2", "3"):
                        haptics.configure(int(command), haptics.gap_ms)
                        controller.reset()
                        panel.pending_selection = None
                        panel.page = 0
                        print(f"已选择Case {command}；已重置目标识别。", flush=True)
                    elif command.startswith("gap ") and command[4:] in ("0.1", "0.3", "0.5"):
                        haptics.configure(haptics.case, round(float(command[4:]) * 1000))
                        controller.reset()
                        panel.pending_selection = None
                        print(f"间隔改为{command[4:]}秒；已重置目标识别。", flush=True)
                    elif command == "0":
                        paused = True
                        haptics.send_directions((False, False, False, False), force=True)
                        print("已暂停震动；start继续。", flush=True)
                    elif command == "start":
                        paused = False
                    elif command == "r":
                        haptics.send_directions((False, False, False, False), force=True)
                        controller.reset()
                        panel.pending_selection = None
                        panel.page = 0
                    elif command:
                        print("指令：1/2/3 | gap 0.1/0.3/0.5 | 0 | start | r | q", flush=True)
                if quit_requested:
                    break
            sequence, frame = reader.get(sequence)
            height, width = frame.shape[:2]
            detections = []
            state = None
            now = time.monotonic()
            if model is not None:
                result = model.predict(frame, device=device, conf=args.confidence, imgsz=640, verbose=False)[0]
                detections = detection_items(result, width, height)
            elif detector is not None:
                if object_lock is not None:
                    object_lock.step(frame,controller,detector,detection_items,now)
                elif controller.observe(frame, now):
                    if controller.phase == "detecting":
                        detector.reset()
                    items = []
                    for source, result in detector.track(frame, controller.selected_class):
                        for box, item in zip(result.boxes, detection_items(result, width, height)):
                            if box.id is None:
                                continue
                            item["track_id"] = f"{source}:{int(box.id.item())}"
                            if source == "scene":
                                item["source_class"] = item["class"]
                                item["class"] = target_class(item["class"])
                            items.append(item)
                    controller.update_detections(items, completed_at=time.monotonic())
                if (args.target_number is not None and not automatic_choice_used
                        and controller.phase == "choose"):
                    if not controller.select(args.target_number):
                        raise RuntimeError("候选编号不存在；请先查看 headless 输出中的候选对象。")
                    automatic_choice_used = True
                state = controller.snapshot(width, height)
                if object_lock is not None:
                    state["tracking_method"] = "CSRT+local verification" if controller.selected_id is not None else "YOLO confirmation"
                    state["tracking_diagnostics"] = dict(source=object_lock.tracking_source,
                        verification=object_lock.reject_reason, status=object_lock.status)
                    if controller.phase in ("lost", "empty"):
                        state["message"] = object_lock.status
                    elif controller.phase == "tracking":
                        state["message"] = object_lock.status
                if haptics is not None:
                    state["haptics"] = haptics.update(state if not paused else None, width, height,
                                                       controller.tolerance, observed_at=now)
                elif controller.auto_target:
                    from sony_haptics import preview_feedback
                    state["haptics"] = preview_feedback(state, width, height, now)
                    status = (state["phase"], state["selected_track_id"],
                              state["haptics"]["text"],
                              state.get("tracking_diagnostics", {}).get("verification"))
                    if status != last_preview_status:
                        print(f"[无硬件] {state['phase']} | {state['message']} | {state['haptics']['text']}", flush=True)
                        last_preview_status = status
                detections = controller.display_items()
                if object_lock is not None and controller.selected_id is None and object_lock.pending is not None:
                    detections = [dict(object_lock.pending, tracking_method="candidate")]
            if args.headless and (model is not None or detector is not None):
                output = {
                    "frame": count + 1,
                    "image_size": {"width": width, "height": height},
                    "coordinate_system": {
                        "unit": "pixel", "origin": [width / 2, height / 2],
                        "x_positive": "right", "y_positive": "up",
                        "position_reference": "bounding_box_center",
                    },
                    "detections": detections,
                }
                if state is not None:
                    output["shooting"] = state
                print(json.dumps(output, ensure_ascii=False), flush=True)
            if not args.headless:
                view = annotate_frame(frame, detections)
                if panel is not None:
                    fps = 0 if last_display_time is None else 1 / max(now - last_display_time, 1e-6)
                    view = panel.render(view, state, fps)
                cv2.imshow("Sony Wi-Fi YOLO", view)
                last_display_time = now
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if controller is not None:
                    if key in (ord("r"), ord("R")):
                        controller.reset()
                        if haptics is not None:
                            haptics.send_directions((False, False, False, False), force=True)
                        panel.page = 0
                        panel.pending_selection = None
                    elif key in (ord("n"), ord("N")):
                        panel.page += 1
                    elif key in (ord("p"), ord("P")):
                        panel.page = max(0, panel.page - 1)
                    selection = panel.take_selection()
                    if selection is not None:
                        controller.select(selection)
                if cv2.getWindowProperty("Sony Wi-Fi YOLO", cv2.WND_PROP_VISIBLE) < 1:
                    break
            count += 1
            if args.frames and count >= args.frames:
                break
    finally:
        if haptics is not None and not haptics.close():
            print("未收到停止振动确认，请检查 ESP32 和模块供电。", file=sys.stderr)
        if reader is not None:
            reader.close()
        camera.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已停止。")
    except (RuntimeError, requests.RequestException, TimeoutError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        sys.exit(1)
