from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageGrab
from PyQt5 import QtCore, QtGui, QtWidgets


# ============================================================
#  保底计数
# ============================================================

PITY_LIMIT = 80


@dataclass
class CaptureRecord:
    id: str
    image_path: str
    captured_at: str
    locked_roi: tuple[int, int, int, int] | None = None
    gif_path: str = ""

    @property
    def captured_dt(self) -> datetime:
        return datetime.fromisoformat(self.captured_at)


@dataclass
class AccountState:
    account_id: str
    name: str
    pity_progress: int = 0
    pity_cycles: int = 0
    captures: list[CaptureRecord] = field(default_factory=list)

    @property
    def remaining_to_pity(self) -> int:
        return PITY_LIMIT - self.pity_progress

    def register_hit(self, record: CaptureRecord) -> bool:
        self.pity_progress += 1
        self.captures.insert(0, record)
        if self.pity_progress >= PITY_LIMIT:
            self.pity_progress = 0
            self.pity_cycles += 1
            return True
        return False


# ============================================================
#  工具函数 & RegionSelector（已内置，不再依赖 box_detector_client）
# ============================================================

def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", app_root()))


def normalize_roi(
    roi: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> dict[str, float]:
    frame_w, frame_h = frame_size
    x1, y1, x2, y2 = roi
    return {
        "x1": x1 / frame_w,
        "y1": y1 / frame_h,
        "x2": x2 / frame_w,
        "y2": y2 / frame_h,
    }


def denormalize_roi(
    roi_pct: dict[str, float],
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    frame_w, frame_h = frame_size
    if frame_w <= 0 or frame_h <= 0:
        return None
    x1 = int(round(float(roi_pct["x1"]) * frame_w))
    y1 = int(round(float(roi_pct["y1"]) * frame_h))
    x2 = int(round(float(roi_pct["x2"]) * frame_w))
    y2 = int(round(float(roi_pct["y2"]) * frame_h))
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return (x1, y1, x2, y2)


class RegionSelector(QtWidgets.QDialog):
    def __init__(self, screenshot: np.ndarray, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setCursor(QtCore.Qt.CrossCursor)
        h, w = screenshot.shape[:2]
        rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(qimg)
        desk = QtWidgets.QApplication.desktop()
        virtual_rect = desk.screenGeometry(0)
        for i in range(1, desk.screenCount()):
            virtual_rect = virtual_rect.united(desk.screenGeometry(i))
        self.setGeometry(virtual_rect)
        self.origin: QtCore.QPoint | None = None
        self.sel_rect: QtCore.QRect | None = None
        self.selected_roi: tuple[int, int, int, int] | None = None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)
        if self.sel_rect is not None:
            painter.setPen(QtGui.QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))
            painter.setBrush(QtGui.QColor(255, 0, 0, 40))
            painter.drawRect(self.sel_rect)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.origin = event.pos()
            self.sel_rect = QtCore.QRect(self.origin, self.origin)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.origin is not None:
            self.sel_rect = QtCore.QRect(self.origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.sel_rect is not None and self.sel_rect.width() > 10 and self.sel_rect.height() > 10:
            r = self.sel_rect
            self.selected_roi = (r.left(), r.top(), r.right() + 1, r.bottom() + 1)
        self.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.reject()


# ============================================================
#  常量
# ============================================================

GIF_FPS = 10
_ABSENT_FRAMES = 3  # 内部滞后，防止闪烁误触发（不暴露给用户）


# ============================================================
#  路径工具
# ============================================================

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _user_data_root() -> Path:
    """回退目录：当项目根目录不可写时（如打包后的 Program Files），使用用户目录下的隐藏文件夹。"""
    return Path.home() / ".RocoLuckyBoxDetector"


def _try_preferred_dir(preferred: Path, fallback_name: str) -> Path:
    """尝试在项目根目录下创建文件夹，不可写则回退到用户目录。"""
    try:
        _ensure_dir(preferred)
        (preferred / ".test").write_text("")
        (preferred / ".test").unlink()
        return preferred
    except Exception:
        fb = _user_data_root() / fallback_name
        _ensure_dir(fb)
        print(f"[路径] 项目根目录不可写，已回退到 {fb}")
        return fb


def resolve_storage_dir(base_dir: Path) -> Path:
    return _try_preferred_dir(base_dir / "captures", "captures")


def resolve_config_path(base_dir: Path) -> Path:
    preferred = base_dir / "config" / "lucky_box_config.json"
    try:
        if not preferred.exists():
            preferred.parent.mkdir(parents=True, exist_ok=True)
            preferred.write_text("{}", encoding="utf-8")
        return preferred
    except Exception:
        fb = _user_data_root() / "config" / "lucky_box_config.json"
        fb.parent.mkdir(parents=True, exist_ok=True)
        if not fb.exists():
            fb.write_text("{}", encoding="utf-8")
        print(f"[路径] 项目根目录不可写，配置已回退到 {fb}")
        return fb


def resolve_state_path(base_dir: Path) -> Path:
    preferred = base_dir / "config" / "lucky_box_state.json"
    try:
        if not preferred.exists():
            preferred.parent.mkdir(parents=True, exist_ok=True)
            preferred.write_text("{}", encoding="utf-8")
        return preferred
    except Exception:
        fb = _user_data_root() / "config" / "lucky_box_state.json"
        fb.parent.mkdir(parents=True, exist_ok=True)
        if not fb.exists():
            fb.write_text("{}", encoding="utf-8")
        print(f"[路径] 项目根目录不可写，状态已回退到 {fb}")
        return fb


def verify_dir() -> Path:
    """验证图存放于项目根目录下的 verify 文件夹，不可写时回退到用户目录。"""
    return _try_preferred_dir(app_root() / "verify", "verify")


def _format_file_ts(stem: str) -> str:
    """从文件名提取时间戳，例如 '20260527_143025_a1b2c3d4' -> '2026-05-27 14:30:25'"""
    parts = stem.split("_")
    if len(parts) >= 2:
        ds = parts[0]
        ts = parts[1]
        try:
            dt = datetime.strptime(f"{ds}_{ts}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return ""


# ============================================================
#  模板匹配检测器
# ============================================================

class TemplateMatchDetector:
    """用 cv2.matchTemplate 做像素级模板匹配，适合 UI 文字等固定元素。"""

    def __init__(self, template_path: Path, threshold: float = 0.70):
        image = cv2.imdecode(np.fromfile(str(template_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"找不到模板图: {template_path}")
        self.template_gray = image
        self.th = threshold
        self.th_h, self.th_w = image.shape[:2]

    def detect(self, frame_bgr: np.ndarray) -> tuple[bool, float, tuple | None]:
        """返回 (found, score, location_xywh)。"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < self.th_h or gray.shape[1] < self.th_w:
            return False, 0.0, None
        result = cv2.matchTemplate(gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= self.th:
            return True, float(max_val), (max_loc[0], max_loc[1], self.th_w, self.th_h)
        return False, float(max_val), None


# ============================================================
#  环形缓冲区
# ============================================================

class RingBuffer:
    def __init__(self, maxlen: int):
        self._buf: deque[PILImage.Image] = deque(maxlen=maxlen)

    def push(self, img: PILImage.Image) -> None:
        self._buf.append(img.copy())

    def snapshot(self) -> list[PILImage.Image]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def maxlen(self) -> int:
        return self._buf.maxlen


# ============================================================
#  后台检测线程
# ============================================================

class LuckyBoxWorker(threading.Thread):
    def __init__(
        self,
        detector: TemplateMatchDetector,
        name_roi: tuple[int, int, int, int] | None,
        box_roi: tuple[int, int, int, int] | None,
        storage_dir: Path,
        event_queue: queue.Queue,
        gif_duration: float = 1.0,
        offset_after: float = 0.5,
        interval_sec: float = 0.10,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.name_roi = name_roi
        self.box_roi = box_roi
        self.storage_dir = storage_dir
        self.event_queue = event_queue
        self.gif_duration = gif_duration
        self.offset_after = offset_after
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()

    @property
    def _ring_maxlen(self) -> int:
        return int((self.offset_after + self.gif_duration + 0.3) * GIF_FPS)

    def run(self) -> None:
        ring_max = self._ring_maxlen
        ring = RingBuffer(maxlen=ring_max)
        name_present = False
        absent_count = 0
        last_cap = 0.0
        post_trigger_frames = 0
        post_trigger_target = 0

        print(f"[worker] 启动, ring_maxlen={ring_max}, offset={self.offset_after}s, duration={self.gif_duration}s")

        while not self.stop_event.is_set():
            if not self.enabled.is_set():
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            try:
                screenshot = ImageGrab.grab(all_screens=True)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                name_bgr = self._crop(frame, self.name_roi) if self.name_roi else frame
                box_bgr = self._crop(frame, self.box_roi) if self.box_roi else frame

                # 盒子区按 24fps 塞环形缓冲区（post-trigger 阶段也不跳过）
                now = time.perf_counter()
                if now - last_cap >= 1.0 / GIF_FPS:
                    box_rgb = box_bgr[:, :, ::-1]
                    ring.push(PILImage.fromarray(box_rgb.copy()))
                    last_cap = now

                # 消失后继续采集阶段（仍然塞帧到 ring，只跳过检测）
                if post_trigger_target > 0:
                    post_trigger_frames += 1
                    if post_trigger_frames >= post_trigger_target:
                        take = min(int(self.gif_duration * GIF_FPS), len(ring))
                        print(f"[worker] post-trigger 结束, ring_len={len(ring)}, take={take}, saving GIF...")
                        path = self._save_gif(ring, take)
                        if path:
                            record = CaptureRecord(
                                id=uuid.uuid4().hex,
                                image_path=path,
                                gif_path=path,
                                captured_at=datetime.now().isoformat(timespec="seconds"),
                            )
                            self.event_queue.put(("capture", record))
                            print(f"[worker] GIF 已保存: {path}")
                        else:
                            print(f"[worker] GIF 保存失败 (ring_len={len(ring)})")
                        post_trigger_target = 0
                        ring.clear()
                        ring = RingBuffer(maxlen=ring_max)
                        last_cap = time.perf_counter()
                    # 跳过检测，但要等帧间隔
                    left = self.interval_sec - (time.perf_counter() - t0)
                    if left > 0:
                        self._sleep(left)
                    continue

                # 模板匹配检测名字
                found, score, _loc = self.detector.detect(name_bgr)

                if found:
                    absent_count = 0
                    if not name_present:
                        name_present = True
                        msg = f"检测到幸运惊喜盒 (score={score:.2f})"
                        print(f"[worker] {msg}")
                        self.event_queue.put(("status", msg))
                else:
                    if name_present:
                        absent_count += 1
                        print(f"[worker] 未匹配 absent_count={absent_count}/{_ABSENT_FRAMES} score={score:.2f}")
                        if absent_count >= _ABSENT_FRAMES:
                            post_trigger_target = int(self.offset_after * GIF_FPS)
                            post_trigger_frames = 0
                            name_present = False
                            absent_count = 0
                            msg = f"消失！延迟{self.offset_after:.1f}s后录制{self.gif_duration:.1f}s GIF..."
                            print(f"[worker] {msg}")
                            self.event_queue.put(("status", msg))
                            continue
                    elif not name_present:
                        self.event_queue.put(("status", f"未检测到 (score={score:.2f})"))

            except Exception as exc:
                print(f"[worker] 异常: {exc}")
                self.event_queue.put(("status", f"异常: {exc}"))
                self._sleep(1.0)
                continue

            left = self.interval_sec - (time.perf_counter() - t0)
            if left > 0:
                self._sleep(left)

    @staticmethod
    def _crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = roi
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2]
        return frame

    def _save_gif(self, ring: RingBuffer, take: int = 0) -> str | None:
        frames = ring.snapshot()
        if take > 0 and take < len(frames):
            frames = frames[-take:]
        if len(frames) < 2:
            print(f"[worker] _save_gif: 帧数不足 ({len(frames)})")
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        path = self.storage_dir / f"{ts}_{uid}.gif"
        try:
            # RingBuffer 存的是纯 PIL Image，直接就是帧列表
            frames[0].save(
                str(path),
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / GIF_FPS),
                loop=0,
            )
            self.event_queue.put(("status", f"GIF已保存: {path.name}"))
            return str(path)
        except Exception as exc:
            self.event_queue.put(("status", f"GIF保存失败: {exc}"))
            return None

    def _sleep(self, sec: float) -> None:
        dead = time.perf_counter() + sec
        while time.perf_counter() < dead:
            if self.stop_event.is_set():
                return
            time.sleep(min(0.05, max(0.001, dead - time.perf_counter())))

    def set_enabled(self, v: bool) -> None:
        self.enabled.set() if v else self.enabled.clear()

    def stop(self) -> None:
        self.stop_event.set()


# ============================================================
#  GUI
# ============================================================

class LuckyBoxWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        template_path: Path,
        storage_dir: Path,
        config_path: Path,
        state_path: Path,
        match_threshold: float = 0.70,
    ):
        super().__init__()
        self.template_path = template_path
        self.storage_dir = storage_dir
        self.config_path = config_path
        self.state_path = state_path
        self.match_threshold = match_threshold

        self.name_roi: tuple | None = None
        self.box_roi: tuple | None = None
        self.detector = TemplateMatchDetector(template_path, threshold=match_threshold)

        self.event_queue: queue.Queue = queue.Queue()
        self.monitoring = False

        # 保底计数状态
        self._account = self._load_state()

        self.setWindowTitle("幸运惊喜盒检测器")
        self.resize(860, 520)
        self._build_ui()
        self._load_config()

        # worker 中的参数：先从配置加载，否则用默认值
        duration = getattr(self, "_cfg_duration", 1.0)
        offset = getattr(self, "_cfg_offset", 0.5)

        self.worker = LuckyBoxWorker(
            detector=self.detector,
            name_roi=self.name_roi,
            box_roi=self.box_roi,
            storage_dir=self.storage_dir,
            event_queue=self.event_queue,
            gif_duration=duration,
            offset_after=offset,
        )

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._drain_events)
        self._timer.start(200)
        self.worker.start()

        # 如果已配置 ROI，自动进入监控模式
        if self.name_roi is not None and self.box_roi is not None:
            self.monitoring = True
            self.btn_toggle.setText("停止监控")
            self.lbl_status.setText("监控中 — 等待幸运惊喜盒出现...")
        else:
            self.worker.set_enabled(False)

        self._refresh_pity_display()

    # ---------- UI 构建 ----------
    def _build_ui(self) -> None:
        c = QtWidgets.QWidget()
        self.setCentralWidget(c)
        root = QtWidgets.QVBoxLayout(c)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 模板名
        h1 = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"模板: {self.template_path.name}")
        f = lbl.font(); f.setBold(True); lbl.setFont(f)
        h1.addWidget(lbl)
        h1.addStretch()
        root.addLayout(h1)

        # 按钮行
        h2 = QtWidgets.QHBoxLayout()
        self.btn_name = QtWidgets.QPushButton("框选名字区域")
        self.btn_name.clicked.connect(lambda: self._select_roi("name"))
        h2.addWidget(self.btn_name)

        self.btn_box = QtWidgets.QPushButton("框选盒子区域")
        self.btn_box.clicked.connect(lambda: self._select_roi("box"))
        h2.addWidget(self.btn_box)

        self.btn_v_name = QtWidgets.QPushButton("查看名字验证图")
        self.btn_v_name.clicked.connect(lambda: self._open_verify("name"))
        h2.addWidget(self.btn_v_name)

        self.btn_v_box = QtWidgets.QPushButton("查看盒子验证图")
        self.btn_v_box.clicked.connect(lambda: self._open_verify("box"))
        h2.addWidget(self.btn_v_box)

        self.btn_toggle = QtWidgets.QPushButton("开始监控")
        self.btn_toggle.clicked.connect(self._toggle)
        h2.addWidget(self.btn_toggle)
        root.addLayout(h2)

        # ROI 信息
        h3 = QtWidgets.QHBoxLayout()
        self.lbl_name = QtWidgets.QLabel("名字范围: 未设置")
        self.lbl_name.setStyleSheet("color: #e74c3c; font-weight: bold;")
        h3.addWidget(self.lbl_name)
        self.lbl_box = QtWidgets.QLabel("  盒子范围: 未设置")
        self.lbl_box.setStyleSheet("color: #2980b9; font-weight: bold;")
        h3.addWidget(self.lbl_box)
        h3.addStretch()
        root.addLayout(h3)

        # 状态 + 匹配分
        h4 = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("就绪 — 请框选两个区域后点击开始")
        self.lbl_status.setStyleSheet("font-size: 11pt; padding: 4px;")
        h4.addWidget(self.lbl_status, 1)
        self.lbl_score = QtWidgets.QLabel("")
        self.lbl_score.setStyleSheet("color: #666;")
        h4.addWidget(self.lbl_score)
        root.addLayout(h4)

        # ---- 可折叠参数面板 ----
        param_group = QtWidgets.QGroupBox("检测参数")
        param_group.setCheckable(True)
        param_group.toggled.connect(lambda on: self._toggle_param_group(param_group, on))
        pg = QtWidgets.QGridLayout(param_group)
        pg.setSpacing(6)

        pg.addWidget(QtWidgets.QLabel("匹配分阈值"), 0, 0)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.10, 0.99)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(self.match_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        pg.addWidget(self.threshold_spin, 0, 1)

        pg.addWidget(QtWidgets.QLabel("消失后偏移(秒)"), 0, 2)
        pg_tip = QtWidgets.QLabel("检测到名字消失后，等多久开始录GIF")
        pg_tip.setStyleSheet("font-size: 9px; color: #999;")
        pg.addWidget(pg_tip, 1, 2)
        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(0.0, 5.0)
        self.offset_spin.setSingleStep(0.1)
        self.offset_spin.setDecimals(1)
        self.offset_spin.setValue(2.5)
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        pg.addWidget(self.offset_spin, 0, 3)

        pg.addWidget(QtWidgets.QLabel("GIF时长(秒)"), 2, 0)
        self.duration_spin = QtWidgets.QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 10.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(2.5)
        self.duration_spin.valueChanged.connect(self._on_duration_changed)
        pg.addWidget(self.duration_spin, 2, 1)

        reset_btn = QtWidgets.QPushButton("重置为默认")
        reset_btn.clicked.connect(self._reset_params)
        pg.addWidget(reset_btn, 2, 2)

        self._toggle_param_group(param_group, False)
        root.addWidget(param_group)

        # ---- 保底计数面板 ----
        pity_group = QtWidgets.QGroupBox("保底计数")
        pity_layout = QtWidgets.QVBoxLayout(pity_group)
        pity_layout.setSpacing(6)

        pity_info_row = QtWidgets.QHBoxLayout()
        self.lbl_pity_summary = QtWidgets.QLabel()
        self.lbl_pity_summary.setStyleSheet("font-size: 11pt; font-weight: bold;")
        pity_info_row.addWidget(self.lbl_pity_summary, 1)

        self.btn_manual_hit = QtWidgets.QPushButton("手动命中 +1")
        self.btn_manual_hit.clicked.connect(self._manual_hit)
        pity_info_row.addWidget(self.btn_manual_hit)

        self.btn_trigger_pity = QtWidgets.QPushButton("触发保底")
        self.btn_trigger_pity.clicked.connect(self._trigger_pity)
        pity_info_row.addWidget(self.btn_trigger_pity)

        self.btn_edit_pity = QtWidgets.QPushButton("编辑保底计数")
        self.btn_edit_pity.clicked.connect(self._edit_pity)
        pity_info_row.addWidget(self.btn_edit_pity)

        self.btn_clear_pity = QtWidgets.QPushButton("清除全部")
        self.btn_clear_pity.setStyleSheet("QPushButton { color: #d32f2f; }")
        self.btn_clear_pity.clicked.connect(self._clear_pity)
        pity_info_row.addWidget(self.btn_clear_pity)

        self.btn_clear_captures = QtWidgets.QPushButton("清空截图文件夹")
        self.btn_clear_captures.setStyleSheet("QPushButton { color: #d32f2f; }")
        self.btn_clear_captures.clicked.connect(self._clear_captures)
        pity_info_row.addWidget(self.btn_clear_captures)

        pity_layout.addLayout(pity_info_row)
        root.addWidget(pity_group)

        # 最近 GIF
        self.lbl_last = QtWidgets.QLabel("")
        self.lbl_last.setOpenExternalLinks(True)
        self.lbl_last.setTextFormat(QtCore.Qt.RichText)
        root.addWidget(self.lbl_last)

        # 历史缩略图
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        self._history_grid = QtWidgets.QGridLayout(container)
        self._history_grid.setContentsMargins(0, 0, 0, 0)
        self._history_grid.setSpacing(8)
        self._history_grid.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        for col in range(6):
            self._history_grid.setColumnStretch(col, 0)
        scroll.setWidget(container)
        self._redraw_history()

    # ---------- ROI 选择 ----------
    def _select_roi(self, which: str) -> None:
        self.showMinimized()
        QtCore.QTimer.singleShot(300, lambda: self._show_selector(which))

    def _show_selector(self, which: str) -> None:
        ss = ImageGrab.grab(all_screens=True)
        frame = cv2.cvtColor(np.array(ss), cv2.COLOR_RGB2BGR)
        sel = RegionSelector(frame)
        if sel.exec_() == QtWidgets.QDialog.Accepted and sel.selected_roi is not None:
            roi = sel.selected_roi
            label = "名字" if which == "name" else "盒子"
            if which == "name":
                self.name_roi = roi
                self.worker.name_roi = roi
                self.lbl_name.setText(f"名字范围: {roi}")
            else:
                self.box_roi = roi
                self.worker.box_roi = roi
                self.lbl_box.setText(f"盒子范围: {roi}")
            self._save_verify(frame, roi, which)
            self._save_config()
            self.lbl_status.setText(f"{label}范围已更新 — 验证图已保存")
        else:
            self.lbl_status.setText("已取消框选")
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _save_verify(self, frame: np.ndarray, roi: tuple, which: str) -> None:
        x1, y1, x2, y2 = roi
        ann = frame.copy()
        cv2.rectangle(ann, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 3)
        vd = verify_dir()
        # 删除旧的同类验证图，只保留最新一张
        prefix = f"verify_{which}_"
        for old in vd.glob(f"{prefix}*.png"):
            try:
                old.unlink()
            except OSError:
                pass
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = vd / f"verify_{which}_{ts}.png"
        cv2.imencode('.png', ann)[1].tofile(str(p))
        self.lbl_status.setText(
            f"{'名字' if which == 'name' else '盒子'}范围已更新 — 验证图: {p.name}"
        )

    def _open_verify(self, which: str) -> None:
        """打开验证图。"""
        vd = verify_dir()
        prefix = f"verify_{which}_"
        candidates = sorted(
            [p for p in vd.glob(f"{prefix}*.png")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            QtWidgets.QMessageBox.information(self, "提示", f"尚未保存过{'名字' if which == 'name' else '盒子'}验证图。")
            return
        os.startfile(str(candidates[0]))

    # ---------- 启停 ----------
    def _toggle(self) -> None:
        if self.name_roi is None or self.box_roi is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先框选名字区域和盒子区域。")
            return
        self.monitoring = not self.monitoring
        self.worker.set_enabled(self.monitoring)
        if self.monitoring:
            self.btn_toggle.setText("停止监控")
            self.lbl_status.setText("监控中 — 等待幸运惊喜盒出现...")
        else:
            self.btn_toggle.setText("开始监控")
            self.lbl_status.setText("监控已停止")

    # ---------- 事件消费 ----------
    def _drain_events(self) -> None:
        while True:
            try:
                ev = self.event_queue.get_nowait()
            except queue.Empty:
                return
            if ev[0] == "status":
                self.lbl_status.setText(ev[1])
                if "score=" in ev[1]:
                    try:
                        s = ev[1].split("score=")[1].split(")")[0]
                        self.lbl_score.setText(f"匹配分: {s}")
                    except Exception:
                        pass
            elif ev[0] == "capture":
                record: CaptureRecord = ev[1]
                self._account.register_hit(record)
                self._refresh_pity_display()
                self._save_state()
                self._redraw_history()
                self.lbl_last.setText(
                    f'<a href="file:///{record.gif_path}" style="color:#1a73e8;">最近: {Path(record.gif_path).name}</a>'
                )

    # ---------- 保底计数 ----------
    def _refresh_pity_display(self) -> None:
        a = self._account
        summary = (
            f"保底进度 {a.pity_progress}/{PITY_LIMIT} | "
            f"距保底 {a.remaining_to_pity} 次 | "
            f"已触发保底 {a.pity_cycles} 次"
        )
        self.lbl_pity_summary.setText(summary)
        if a.remaining_to_pity <= 10:
            self.lbl_pity_summary.setStyleSheet("font-size: 11pt; font-weight: bold; color: #e74c3c;")
        else:
            self.lbl_pity_summary.setStyleSheet("font-size: 11pt; font-weight: bold;")

    def _edit_pity(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("编辑保底计数")
        dialog.resize(300, 150)
        layout = QtWidgets.QFormLayout(dialog)
        layout.setSpacing(10)

        spin_progress = QtWidgets.QSpinBox()
        spin_progress.setRange(0, PITY_LIMIT - 1)
        spin_progress.setValue(self._account.pity_progress)
        layout.addRow("保底进度:", spin_progress)

        spin_cycles = QtWidgets.QSpinBox()
        spin_cycles.setRange(0, 9999)
        spin_cycles.setValue(self._account.pity_cycles)
        layout.addRow("保底次数:", spin_cycles)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addRow(btn_box)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._account.pity_progress = spin_progress.value()
        self._account.pity_cycles = spin_cycles.value()
        self._refresh_pity_display()
        self._save_state()
        self.lbl_status.setText("保底计数已更新")

    def _manual_hit(self) -> None:
        record = CaptureRecord(
            id=uuid.uuid4().hex,
            image_path="",
            gif_path="",
            captured_at=datetime.now().isoformat(timespec="seconds"),
        )
        triggered = self._account.register_hit(record)
        self._refresh_pity_display()
        self._save_state()
        self._redraw_history()
        if triggered:
            QtWidgets.QMessageBox.information(self, "保底触发", f"本轮第 {PITY_LIMIT} 次！保底已触发！")
            self.lbl_status.setText(f"保底触发！累计 {self._account.pity_cycles} 次保底")
        else:
            self.lbl_status.setText("手动命中 +1")

    def _trigger_pity(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "确认触发保底",
            f"确定要手动触发一次保底吗？\n当前: 保底进度 {self._account.pity_progress}/{PITY_LIMIT}，{self._account.pity_cycles} 次保底",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self._account.pity_cycles += 1
        self._refresh_pity_display()
        self._save_state()
        self.lbl_status.setText(f"手动触发保底，当前累计 {self._account.pity_cycles} 次保底")

    def _clear_pity(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "确认清除",
            f"确定要清除全部保底计数吗？\n当前: 保底进度 {self._account.pity_progress}/{PITY_LIMIT}，保底 {self._account.pity_cycles} 次",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._account = AccountState(account_id="default", name="默认")
            self._refresh_pity_display()
            self._save_state()
            self._redraw_history()
            self.lbl_status.setText("保底计数已清除")

    def _clear_captures(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "确认清空",
            f"确定要删除 captures 文件夹中的所有文件吗？\n此操作不可恢复。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        import shutil
        deleted = 0
        if self.storage_dir.exists():
            for f in self.storage_dir.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                        deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        deleted += 1
                except Exception:
                    pass
        self._account.captures.clear()
        self._save_state()
        self._redraw_history()
        self.lbl_last.setText("")
        self.lbl_status.setText(f"已清空 captures，删除 {deleted} 个文件/文件夹")

    def _load_state(self) -> AccountState:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                acc = AccountState(
                    account_id=data.get("account_id", "default"),
                    name=data.get("name", "默认"),
                    pity_progress=data.get("pity_progress", data.get("total_hits", 0)),
                    pity_cycles=data.get("pity_cycles", 0),
                )
                for c in data.get("captures", []):
                    acc.captures.append(CaptureRecord(
                        id=c.get("id", ""),
                        image_path=c.get("image_path", ""),
                        gif_path=c.get("gif_path", ""),
                        captured_at=c.get("captured_at", ""),
                    ))
                return acc
            except Exception:
                pass
        return AccountState(account_id="default", name="默认")

    def _save_state(self) -> None:
        a = self._account
        data = {
            "account_id": a.account_id,
            "name": a.name,
            "pity_progress": a.pity_progress,
            "pity_cycles": a.pity_cycles,
            "captures": [
                {
                    "id": c.id,
                    "image_path": c.image_path,
                    "gif_path": c.gif_path,
                    "captured_at": c.captured_at,
                }
                for c in a.captures[:200]
            ],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---------- 历史缩略图 ----------
    def _redraw_history(self) -> None:
        while self._history_grid.count():
            it = self._history_grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        captures = [c for c in self._account.captures if c.gif_path or c.image_path]
        if not captures:
            empty = QtWidgets.QLabel("暂无录制记录")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            empty.setStyleSheet("padding: 36px; color: #666;")
            self._history_grid.addWidget(empty, 0, 0)
            return
        for i, c in enumerate(captures[:30]):
            self._history_grid.addWidget(self._card(c), i // 6, i % 6)

    def _card(self, record: CaptureRecord) -> QtWidgets.QFrame:
        path_str = record.gif_path or record.image_path
        path = Path(path_str) if path_str else None
        has_file = path is not None and path.is_file()
        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        card.setStyleSheet("QFrame { border: 1px solid #d9d9d9; border-radius: 8px; background: white; }")
        card.setFixedWidth(190)
        lo = QtWidgets.QVBoxLayout(card)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setSpacing(10)

        thumb = QtWidgets.QLabel()
        thumb.setFixedSize(166, 112)
        thumb.setAlignment(QtCore.Qt.AlignCenter)
        thumb.setStyleSheet("background: #f5f5f5; border: none;")
        if has_file:
            if path.suffix == ".gif":
                mv = QtGui.QMovie(str(path))
                mv.setScaledSize(QtCore.QSize(166, 112))
                thumb.setMovie(mv)
                mv.start()
            else:
                px = QtGui.QPixmap(str(path))
                if not px.isNull():
                    thumb.setPixmap(px.scaled(166, 112, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                else:
                    thumb.setText("(图片无效)")
        else:
            thumb.setText("(手动记录)")
        lo.addWidget(thumb, 0, QtCore.Qt.AlignHCenter)

        nm = QtWidgets.QLabel(path.name if has_file else "(手动记录)")
        nm.setWordWrap(True)
        nm.setFixedWidth(166)
        nm.setAlignment(QtCore.Qt.AlignCenter)
        nm.setStyleSheet("font-size: 9pt; color: #555; border: none;")
        lo.addWidget(nm)

        ts_label = QtWidgets.QLabel(_format_file_ts(path.stem) if has_file else "")
        ts_label.setFixedWidth(166)
        ts_label.setAlignment(QtCore.Qt.AlignCenter)
        ts_label.setStyleSheet("font-size: 8pt; color: #999; border: none;")
        lo.addWidget(ts_label)

        btn = QtWidgets.QPushButton("打开")
        if has_file:
            btn.clicked.connect(lambda checked, p=str(path): os.startfile(p))
        else:
            btn.setEnabled(False)
        lo.addWidget(btn)
        return card

    # ---------- 配置持久化 ----------
    _DEFAULTS = {"threshold": 0.70, "offset": 2.5, "duration": 2.5}

    def _load_config(self) -> None:
        raw = {}
        if self.config_path.exists():
            try:
                raw = json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        try:
            ss = ImageGrab.grab(all_screens=True)
            sz = ss.size
        except Exception:
            sz = None
        # ROI
        if sz:
            for key, attr in [("name_roi_pct", "name_roi"), ("box_roi_pct", "box_roi")]:
                pct = raw.get(key)
                if isinstance(pct, dict) and {"x1", "y1", "x2", "y2"}.issubset(pct):
                    roi = denormalize_roi(pct, sz)
                    if roi:
                        setattr(self, attr, roi)
        if self.name_roi:
            self.lbl_name.setText(f"名字范围: {self.name_roi}")
        if self.box_roi:
            self.lbl_box.setText(f"盒子范围: {self.box_roi}")
        # 检测参数
        params = raw.get("params", {})
        self.match_threshold = float(params.get("threshold", self._DEFAULTS["threshold"]))
        self._cfg_offset = float(params.get("offset", self._DEFAULTS["offset"]))
        self._cfg_duration = float(params.get("duration", self._DEFAULTS["duration"]))
        self.detector.th = self.match_threshold
        self.threshold_spin.setValue(self.match_threshold)
        self.offset_spin.setValue(self._cfg_offset)
        self.duration_spin.setValue(self._cfg_duration)

    def _save_config(self) -> None:
        try:
            ss = ImageGrab.grab(all_screens=True)
            sz = ss.size
        except Exception:
            sz = None
        data: dict = {}
        if sz and self.name_roi:
            data["name_roi_pct"] = normalize_roi(self.name_roi, sz)
        if sz and self.box_roi:
            data["box_roi_pct"] = normalize_roi(self.box_roi, sz)
        data["params"] = {
            "threshold": self.match_threshold,
            "offset": self.offset_spin.value(),
            "duration": self.duration_spin.value(),
        }
        data["saved_at"] = datetime.now().isoformat(timespec="seconds")
        if sz:
            data["source_size"] = {"width": sz[0], "height": sz[1]}
        try:
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            fb = _user_data_root() / "config" / "lucky_box_config.json"
            fb.parent.mkdir(parents=True, exist_ok=True)
            fb.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.config_path = fb
            print(f"[路径] 项目根目录不可写，配置已回退到 {fb}")

    def _reset_params(self) -> None:
        d = self._DEFAULTS
        self.threshold_spin.setValue(d["threshold"])
        self.offset_spin.setValue(d["offset"])
        self.duration_spin.setValue(d["duration"])
        self._save_config()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.worker.stop()
        self.worker.join(timeout=1.5)
        super().closeEvent(event)

    # ---------- 参数面板回调 ----------
    @staticmethod
    def _toggle_param_group(group: QtWidgets.QGroupBox, visible: bool) -> None:
        for child in group.findChildren(QtWidgets.QWidget):
            if child is not group:
                child.setVisible(visible)

    def _on_threshold_changed(self, value: float) -> None:
        self.match_threshold = value
        self.detector.th = value
        self._save_config()
        self.lbl_status.setText(f"匹配分阈值已更新: {value:.2f}")

    def _on_offset_changed(self, value: float) -> None:
        self.worker.offset_after = value
        self._save_config()
        self.lbl_status.setText(f"消失后偏移已更新: {value:.1f}s")

    def _on_duration_changed(self, value: float) -> None:
        self.worker.gif_duration = value
        self._save_config()
        self.lbl_status.setText(f"GIF时长已更新: {value:.1f}s")


# ============================================================
#  入口
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="幸运惊喜盒检测器 — 模板匹配名字，消失时录盒子GIF")
    p.add_argument("--template", type=Path, default=Path("name.png"), help="名字模板图")
    p.add_argument("--threshold", type=float, default=0.70, help="模板匹配阈值 (0~1)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = app_root()
    res = bundled_root()
    tpl = args.template if args.template.is_absolute() else res / args.template
    if not tpl.exists():
        print(f"错误: 找不到模板图 {tpl}")
        return 1

    if sys.platform.startswith("win"):
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QtWidgets.QApplication(sys.argv)
    win = LuckyBoxWindow(
        template_path=tpl,
        storage_dir=resolve_storage_dir(base),
        config_path=resolve_config_path(base),
        state_path=resolve_state_path(base),
        match_threshold=args.threshold,
    )
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
