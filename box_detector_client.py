from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab
from PyQt5 import QtCore, QtGui, QtWidgets


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
    total_hits: int = 0
    pity_cycles: int = 0
    captures: list[CaptureRecord] = field(default_factory=list)

    @property
    def pity_progress(self) -> int:
        return max(0, self.total_hits - self.pity_cycles * PITY_LIMIT)

    @property
    def remaining_to_pity(self) -> int:
        return PITY_LIMIT - self.pity_progress

    def register_hit(self, record: CaptureRecord) -> bool:
        self.total_hits += 1
        pity_triggered = self.total_hits >= (self.pity_cycles + 1) * PITY_LIMIT
        if pity_triggered:
            self.pity_cycles += 1
        self.captures.insert(0, record)
        return pity_triggered


@dataclass
class DetectionResult:
    found: bool
    good_matches: int = 0
    inliers: int = 0
    template_score: float = 0.0
    debug_text: str = ""
    polygon: np.ndarray | None = None
    locked_roi: tuple[int, int, int, int] | None = None
    roi_locked_now: bool = False


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", app_root()))


def dir_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def resolve_storage_dir(base_dir: Path) -> Path:
    preferred = base_dir / "client_data" / "captures"
    if dir_is_writable(preferred):
        return preferred
    fallback = Path.home() / ".box_detector_client" / "captures"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def resolve_state_path(base_dir: Path) -> Path:
    preferred = base_dir / "client_data" / "state.json"
    if dir_is_writable(preferred.parent):
        return preferred
    fallback = Path.home() / ".box_detector_client" / "state.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


def resolve_config_path(base_dir: Path) -> Path:
    preferred = base_dir / "config.json"
    if preferred.exists() or dir_is_writable(preferred.parent):
        return preferred
    fallback = Path.home() / ".box_detector_client" / "config.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


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


ALGORITHM_LABELS = {"sift": "SIFT", "orb": "ORB", "brisk": "BRISK"}


class TemplateDetector:
    def __init__(
        self,
        template_path: Path,
        algorithm: str = "sift",
        ratio: float | None = None,
        min_matches: int = 5,
        min_inliers: int = 3,
        screen_scale: float = 0.75,
        roi_margin_ratio: float = 0.18,
        min_template_score: float = 0.20,
    ) -> None:
        self.template_path = template_path
        self.algorithm = algorithm
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.screen_scale = screen_scale
        self.roi_margin_ratio = roi_margin_ratio
        self.min_template_score = min_template_score
        self.locked_roi: tuple[int, int, int, int] | None = None

        template_max_height = 360
        if algorithm == "sift":
            self.feature_detector = cv2.SIFT_create(nfeatures=1800)
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
            self.ratio = ratio if ratio is not None else 0.80
            template_max_height = 260
        elif algorithm == "orb":
            self.feature_detector = cv2.ORB_create(
                nfeatures=4000,
                fastThreshold=10,
                scoreType=cv2.ORB_FAST_SCORE,
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self.ratio = ratio if ratio is not None else 0.80
        elif algorithm == "brisk":
            self.feature_detector = cv2.BRISK_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self.ratio = ratio if ratio is not None else 0.80
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        self.template_gray, self.template_mask, self.template_size, self.template_bgr, self.template_hsv_hist = self._load_template(template_path, template_max_height)
        self.template_area = float(self.template_size[0] * self.template_size[1])
        self.template_aspect_ratio = self.template_size[0] / self.template_size[1]
        self.template_keypoints, self.template_descriptors = self.feature_detector.detectAndCompute(
            self.template_gray,
            self.template_mask,
        )
        if self.template_descriptors is None or len(self.template_keypoints) < self.min_matches:
            raise RuntimeError("模板特征不足，无法进行检测。")

    def _verify(self, frame_bgr: np.ndarray, polygon: np.ndarray) -> tuple[float, float]:
        w, h = self.template_size
        src = polygon.reshape(4, 2).astype(np.float32)
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        transform = cv2.getPerspectiveTransform(src, dst)
        warped_bgr = cv2.warpPerspective(frame_bgr, transform, (w, h))
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        gray_score = float(cv2.matchTemplate(warped_gray, self.template_gray, cv2.TM_CCOEFF_NORMED, mask=self.template_mask)[0, 0])
        if self.template_hsv_hist is None:
            return gray_score, 1.0
        hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        color_score = float(cv2.compareHist(self.template_hsv_hist, hist, cv2.HISTCMP_CORREL))
        return gray_score, max(color_score, 0.0)

    def _load_template(self, template_path: Path, max_height: int = 360) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        def resize_rgb_mask(rgb: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
            if rgb.shape[0] <= max_height:
                return rgb, mask
            scale = max_height / rgb.shape[0]
            width = max(1, int(round(rgb.shape[1] * scale)))
            rgb = cv2.resize(rgb, (width, max_height), interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, (width, max_height), interpolation=cv2.INTER_NEAREST)
                mask = np.where(mask > 0, 255, 0).astype("uint8")
            return rgb, mask

        def hsv_hist(bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            return hist

        image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"找不到模板图: {template_path}")
        if image.ndim == 2:
            h, w = image.shape[:2]
            return image, None, (w, h), None, None
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = np.where(alpha > 10, 255, 0).astype("uint8")
            bbox = cv2.boundingRect(mask)
            x, y, w, h = bbox
            gray = gray[y:y+h, x:x+w]
            bgr = bgr[y:y+h, x:x+w]
            mask = mask[y:y+h, x:x+w]
            gray, mask_i = resize_rgb_mask(gray, mask)
            bgr, _ = resize_rgb_mask(bgr, mask)
            h, w = gray.shape[:2]
            return gray, mask_i, (w, h), bgr, hsv_hist(bgr, mask_i)
        bgr = image[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        non_white_mask = np.where(gray < 245, 255, 0).astype("uint8")
        if np.count_nonzero(non_white_mask) > 0:
            x, y, w, h = cv2.boundingRect(non_white_mask)
            gray = gray[y:y+h, x:x+w]
            bgr = bgr[y:y+h, x:x+w]
            non_white_mask = non_white_mask[y:y+h, x:x+w]
            gray, non_white_mask = resize_rgb_mask(gray, non_white_mask)
            bgr, _ = resize_rgb_mask(bgr, non_white_mask)
            h, w = gray.shape[:2]
            return gray, non_white_mask, (w, h), bgr, hsv_hist(bgr, non_white_mask)
        h, w = gray.shape[:2]
        return gray, None, (w, h), bgr, hsv_hist(bgr, None)

    def reset_locked_roi(self) -> None:
        self.locked_roi = None

    def has_locked_roi(self) -> bool:
        return self.locked_roi is not None

    def _scaled_locked_roi(self) -> tuple[int, int, int, int] | None:
        if self.locked_roi is None:
            return None
        x1, y1, x2, y2 = self.locked_roi
        return (
            max(int(x1 * self.screen_scale), 0),
            max(int(y1 * self.screen_scale), 0),
            max(int(np.ceil(x2 * self.screen_scale)), 1),
            max(int(np.ceil(y2 * self.screen_scale)), 1),
        )

    def _expand_polygon_to_roi(self, polygon: np.ndarray, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        pts = polygon.reshape(-1, 2)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        pad_x = max(16.0, width * self.roi_margin_ratio)
        pad_y = max(16.0, height * self.roi_margin_ratio)
        frame_h, frame_w = frame_shape[:2]
        x1 = max(int(np.floor(min_x - pad_x)), 0)
        y1 = max(int(np.floor(min_y - pad_y)), 0)
        x2 = min(int(np.ceil(max_x + pad_x)), frame_w)
        y2 = min(int(np.ceil(max_y + pad_y)), frame_h)
        return (x1, y1, x2, y2)

    def _polygon_is_reasonable(self, polygon: np.ndarray, frame_shape: tuple) -> bool:
        pts = polygon.reshape(4, 2).astype(np.float32)
        area = float(cv2.contourArea(pts))
        if area <= 0:
            return False
        area_ratio = area / self.template_area
        if area_ratio < 0.02 or area_ratio > 8.0:
            return False
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        if width <= 0 or height <= 0:
            return False
        fh, fw = frame_shape[:2]
        overshoot = max(abs(min_x), abs(min_y), abs(max_x - fw), abs(max_y - fh))
        if overshoot > max(fw, fh) * 3:
            return False
        aspect_ratio = width / height
        if aspect_ratio < self.template_aspect_ratio * 0.35 or aspect_ratio > self.template_aspect_ratio * 3.0:
            return False
        return True

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        max_dim = 1400
        effective_scale = self.screen_scale
        if max(frame_h, frame_w) * effective_scale > max_dim:
            effective_scale = max_dim / max(frame_h, frame_w)
        if effective_scale != 1.0:
            scaled = cv2.resize(
                frame_bgr,
                None,
                fx=effective_scale,
                fy=effective_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            scaled = frame_bgr
            effective_scale = 1.0

        offset_x = 0
        offset_y = 0
        search_bgr = scaled
        scaled_roi = self._scaled_locked_roi()
        if scaled_roi is not None:
            x1, y1, x2, y2 = scaled_roi
            x2 = min(x2, scaled.shape[1])
            y2 = min(y2, scaled.shape[0])
            if x2 > x1 and y2 > y1:
                search_bgr = scaled[y1:y2, x1:x2]
                offset_x = x1
                offset_y = y1

        frame_gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self.feature_detector.detectAndCompute(frame_gray, None)
        if frame_descriptors is None or len(frame_keypoints) < self.min_matches:
            return DetectionResult(found=False, locked_roi=self.locked_roi)

        pairs = self.matcher.knnMatch(self.template_descriptors, frame_descriptors, k=2)
        good_matches = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_matches:
            return DetectionResult(found=False, good_matches=len(good_matches), locked_roi=self.locked_roi)

        src_pts = np.float32(
            [self.template_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
        if homography is None or mask is None:
            return DetectionResult(found=False, good_matches=len(good_matches), locked_roi=self.locked_roi)

        inliers = int(mask.ravel().sum())
        if inliers < self.min_inliers:
            return DetectionResult(found=False, good_matches=len(good_matches), inliers=inliers, locked_roi=self.locked_roi)

        w, h = self.template_size
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        polygon = cv2.perspectiveTransform(corners, homography)
        polygon[:, 0, 0] += offset_x
        polygon[:, 0, 1] += offset_y
        if effective_scale != 1.0:
            polygon = polygon / effective_scale

        if not self._polygon_is_reasonable(polygon, frame_bgr.shape):
            return DetectionResult(
                found=False,
                good_matches=len(good_matches),
                inliers=inliers,
                debug_text=f"good={len(good_matches)} inliers={inliers} shape=fail",
                locked_roi=self.locked_roi,
            )

        template_score, color_score = self._verify(frame_bgr, polygon)
        combined = template_score + color_score * 0.15
        if template_score < self.min_template_score:
            return DetectionResult(
                found=False,
                good_matches=len(good_matches),
                inliers=inliers,
                template_score=combined,
                debug_text=f"good={len(good_matches)} inliers={inliers} gray={template_score:.2f} color={color_score:.2f}",
                locked_roi=self.locked_roi,
            )

        roi_locked_now = False
        if self.locked_roi is None:
            self.locked_roi = self._expand_polygon_to_roi(polygon, frame_bgr.shape)
            roi_locked_now = True

        return DetectionResult(
            found=True,
            good_matches=len(good_matches),
            inliers=inliers,
            template_score=combined,
            debug_text=f"good={len(good_matches)} inliers={inliers} gray={template_score:.2f} color={color_score:.2f}",
            polygon=polygon,
            locked_roi=self.locked_roi,
            roi_locked_now=roi_locked_now,
        )


def create_detector(algorithm: str, template_path: Path):
    return TemplateDetector(template_path=template_path, algorithm=algorithm)


class DetectorWorker(threading.Thread):
    def __init__(
        self,
        detectors: dict[str, object],
        active_algorithms: set[str],
        storage_dir: Path,
        get_active_account_id,
        should_detect,
        event_queue: queue.Queue,
        interval_sec: float = 0.25,
        cooldown_sec: float = 5.0,
    ) -> None:
        super().__init__(daemon=True)
        self.detectors = detectors
        self.active_algorithms = active_algorithms
        self.storage_dir = storage_dir
        self.get_active_account_id = get_active_account_id
        self.should_detect = should_detect
        self.event_queue = event_queue
        self.interval_sec = interval_sec
        self.cooldown_sec = cooldown_sec
        self.stop_event = threading.Event()
        self.enabled = threading.Event()
        self.enabled.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            if not self.enabled.is_set():
                time.sleep(0.1)
                continue
            if not self.should_detect():
                self.event_queue.put(("status", "程序在前台，检测已自动暂停"))
                time.sleep(0.15)
                continue

            started = time.perf_counter()
            try:
                screenshot = ImageGrab.grab(all_screens=True)
                frame_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

                found_result = None
                found_algo = ""
                for algo in sorted(self.active_algorithms):
                    r = self.detectors[algo].detect(frame_bgr)
                    if r.found:
                        found_result = r
                        found_algo = algo
                        break

                lock_state = "已锁定" if any(
                    d.has_locked_roi() for d in self.detectors.values()
                ) else "整屏检测中"
                algo_names = "/".join(ALGORITHM_LABELS[a] for a in sorted(self.active_algorithms))
                if found_result is not None:
                    debug_text = found_result.debug_text
                else:
                    debug_text = "未检测到"
                self.event_queue.put(
                    ("status", f"[{algo_names}] 检测中 每0.25秒一次 | {lock_state} | {debug_text}")
                )
                if found_result is not None:
                    account_id = self.get_active_account_id()
                    locked_roi = found_result.locked_roi

                    self.event_queue.put(("status", f"命中 {account_id} [{found_algo.upper()}]，录制2.5秒GIF"))
                    gif_path, thumb_path = self._record_gif(account_id, locked_roi)
                    record = CaptureRecord(
                        id=uuid.uuid4().hex,
                        image_path=thumb_path,
                        gif_path=gif_path,
                        captured_at=datetime.now().isoformat(timespec="seconds"),
                        locked_roi=locked_roi,
                    )
                    self.event_queue.put(("capture", account_id, record, found_result.good_matches, found_result.inliers))
                    self.event_queue.put(("status", f"已命中 {account_id} [{found_algo.upper()}] | GIF已保存 | {self.cooldown_sec}s冷却中"))
                    self._sleep_interruptible(self.cooldown_sec)
                    continue
            except Exception as exc:
                self.event_queue.put(("status", f"检测异常: {exc}"))
                self._sleep_interruptible(1.0)
                continue

            elapsed = time.perf_counter() - started
            remaining = self.interval_sec - elapsed
            if remaining > 0:
                self._sleep_interruptible(remaining)

    def _record_gif(self, account_id: str, locked_roi: tuple[int, int, int, int]) -> tuple[str, str]:
        from PIL import Image as PILImage

        account_dir = self.storage_dir / account_id
        account_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        base = now.strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        x1, y1, x2, y2 = locked_roi

        frames: list[PILImage.Image] = []
        interval = 1.0 / 24
        deadline = time.perf_counter() + 2.5
        while time.perf_counter() < deadline:
            if self.stop_event.is_set():
                break
            try:
                ss = ImageGrab.grab(all_screens=True)
                f = np.array(ss)
                roi_bgr = f[y1:y2, x1:x2, ::-1]
                roi_rgb = roi_bgr[:, :, ::-1]
                frames.append(PILImage.fromarray(roi_rgb))
            except Exception:
                pass
            self._sleep_interruptible(interval)

        gif_path = account_dir / f"{base}.gif"
        thumb_path = account_dir / f"{base}.png"
        if frames:
            frames[0].save(str(thumb_path))
            if len(frames) > 1:
                frames[0].save(
                    str(gif_path), save_all=True, append_images=frames[1:],
                    duration=int(interval * 1000), loop=0,
                )
            else:
                frames[0].save(str(gif_path))
        return str(gif_path), str(thumb_path)

    def _sleep_interruptible(self, duration: float) -> None:
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            if self.stop_event.is_set():
                return
            time.sleep(0.05)

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self.enabled.set()
        else:
            self.enabled.clear()

    def stop(self) -> None:
        self.stop_event.set()


class CaptureCard(QtWidgets.QFrame):
    def __init__(
        self,
        image_path: str,
        relative_text: str,
        exact_text: str,
        locked_roi: tuple[int, int, int, int] | None,
        gif_path: str,
        on_open_clicked,
        on_delete_clicked,
    ) -> None:
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #d9d9d9; border-radius: 8px; background: white; }")
        self.setFixedWidth(190)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        preview = QtWidgets.QLabel()
        preview.setFixedSize(166, 112)
        preview.setAlignment(QtCore.Qt.AlignCenter)
        pixmap = QtGui.QPixmap(image_path)
        if not pixmap.isNull():
            preview.setPixmap(
                pixmap.scaled(
                    preview.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
        else:
            preview.setText("图片加载失败")
        layout.addWidget(preview, 0, QtCore.Qt.AlignHCenter)

        meta = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(relative_text)
        title_font = title.font()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(QtCore.Qt.AlignCenter)
        meta.addWidget(title)

        exact = QtWidgets.QLabel(exact_text)
        exact.setStyleSheet("color: #666666;")
        exact.setAlignment(QtCore.Qt.AlignCenter)
        meta.addWidget(exact)

        btn_row = QtWidgets.QHBoxLayout()
        open_button = QtWidgets.QPushButton("放大")
        open_button.clicked.connect(lambda: on_open_clicked(gif_path, locked_roi))
        btn_row.addWidget(open_button)
        delete_button = QtWidgets.QPushButton("删除")
        delete_button.setStyleSheet("QPushButton { color: #d32f2f; }")
        delete_button.clicked.connect(on_delete_clicked)
        btn_row.addWidget(delete_button)
        meta.addLayout(btn_row)
        layout.addLayout(meta)


class ImagePreviewDialog(QtWidgets.QDialog):
    def __init__(
        self,
        image_path: str,
        locked_roi: tuple[int, int, int, int] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.locked_roi = locked_roi
        self.setWindowTitle("截图预览")
        self.resize(800, 600)

        root = QtWidgets.QVBoxLayout(self)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(self.image_label, 1)

        self.movie: QtGui.QMovie | None = None
        if image_path.endswith(".gif"):
            self.movie = QtGui.QMovie(image_path)
            self._scale_movie()
            self.image_label.setMovie(self.movie)
            self.movie.start()
        else:
            self.image_label.setPixmap(QtGui.QPixmap(image_path).scaled(
                self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
            ))

    def _scale_movie(self) -> None:
        if self.movie is None:
            return
        src = self.movie.currentPixmap()
        if src.isNull():
            return
        label_size = self.image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            label_size = self.size()
        src_w, src_h = src.size().width(), src.size().height()
        ratio = min(label_size.width() / src_w, label_size.height() / src_h)
        self.movie.setScaledSize(QtCore.QSize(
            int(src_w * ratio), int(src_h * ratio),
        ))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.movie is not None:
            self._scale_movie()
        elif self.image_label.pixmap() is not None and not self.image_label.pixmap().isNull():
            self.image_label.setPixmap(
                QtGui.QPixmap(self.image_path).scaled(
                    self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
                )
            )


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
        self.setGeometry(desk.screenGeometry(desk.primaryScreen()))
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


class AccountPage(QtWidgets.QWidget):
    def __init__(self, on_hits_changed, on_pity_triggered, on_clear_clicked) -> None:
        super().__init__()
        self.columns = 5
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        self.summary_label = QtWidgets.QLabel()
        header_font = self.summary_label.font()
        header_font.setPointSize(10)
        header_font.setBold(True)
        self.summary_label.setFont(header_font)
        header.addWidget(self.summary_label, 1)

        header.addWidget(QtWidgets.QLabel("命中次数:"))
        self.hits_spin = QtWidgets.QSpinBox()
        self.hits_spin.setRange(0, 99999)
        self.hits_spin.setFixedWidth(70)
        self.hits_spin.valueChanged.connect(on_hits_changed)
        header.addWidget(self.hits_spin)

        pity_btn = QtWidgets.QPushButton("触发保底")
        pity_btn.clicked.connect(on_pity_triggered)
        header.addWidget(pity_btn)

        clear_btn = QtWidgets.QPushButton("清除全部")
        clear_btn.clicked.connect(on_clear_clicked)
        header.addWidget(clear_btn)
        root.addLayout(header)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(scroll, 1)

        container = QtWidgets.QWidget()
        self.capture_layout = QtWidgets.QGridLayout(container)
        self.capture_layout.setContentsMargins(0, 0, 0, 0)
        self.capture_layout.setSpacing(12)
        self.capture_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        for column in range(self.columns):
            self.capture_layout.setColumnStretch(column, 0)
        scroll.setWidget(container)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def set_hits(self, value: int) -> None:
        self.hits_spin.blockSignals(True)
        self.hits_spin.setValue(value)
        self.hits_spin.blockSignals(False)

    def set_captures(self, capture_widgets: list[QtWidgets.QWidget]) -> None:
        for row in range(self.capture_layout.rowCount() + 2):
            self.capture_layout.setRowStretch(row, 0)
        while self.capture_layout.count():
            item = self.capture_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not capture_widgets:
            empty = QtWidgets.QLabel("当前账号还没有命中记录。")
            empty.setAlignment(QtCore.Qt.AlignCenter)
            empty.setStyleSheet("padding: 36px; color: #666666;")
            self.capture_layout.addWidget(empty, 0, 0, 1, self.columns)
        else:
            for index, widget in enumerate(capture_widgets):
                row = index // self.columns
                column = index % self.columns
                self.capture_layout.addWidget(widget, row, column)
            final_row = (len(capture_widgets) - 1) // self.columns + 1
            self.capture_layout.setRowStretch(final_row, 1)


class BoxDetectorWindow(QtWidgets.QMainWindow):
    def __init__(self, template_path: Path, resource_dir: Path, storage_dir: Path, state_path: Path, config_path: Path, algorithm: str = "orb") -> None:
        super().__init__()
        self.template_path = template_path
        self.storage_dir = storage_dir
        self.state_path = state_path
        self.config_path = config_path
        self.state_lock = threading.Lock()
        self.preview_dialogs: list[ImagePreviewDialog] = []
        self.accounts = self.load_state()
        if not self.accounts:
            self.accounts = [AccountState(account_id="account_1", name="账号1")]
            self.save_state()
        self.active_account_id = self.accounts[0].account_id
        self.foreground_pause = False
        self.active_algorithms: set[str] = {algorithm}

        self.detectors: dict[str, TemplateDetector] = {}
        for algo in ALGORITHM_LABELS:
            self.detectors[algo] = create_detector(algo, template_path)
        self.event_queue: queue.Queue = queue.Queue()
        self.worker = DetectorWorker(
            detectors=self.detectors,
            active_algorithms=self.active_algorithms,
            storage_dir=self.storage_dir,
            get_active_account_id=self.get_active_account_id,
            should_detect=self.should_run_detection,
            event_queue=self.event_queue,
        )

        self.account_pages: dict[str, AccountPage] = {}
        self.monitoring_enabled = True
        self.status_label = QtWidgets.QLabel("准备开始检测")
        self.monitor_label = QtWidgets.QLabel("运行中")

        self.setWindowTitle("箱子保底检测客户端")
        self.resize(1120, 760)
        self.build_ui()
        self.confirm_startup_roi_choice()

        self.event_timer = QtCore.QTimer(self)
        self.event_timer.timeout.connect(self.process_events)
        self.event_timer.start(200)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_current_page)
        self.refresh_timer.start(1000)

        self.foreground_timer = QtCore.QTimer(self)
        self.foreground_timer.timeout.connect(self.update_foreground_pause_state)
        self.foreground_timer.start(150)

        self.worker.start()

    def load_roi_config(self) -> dict | None:
        if not self.config_path.exists():
            return None
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        roi_pct = raw.get("locked_roi_pct")
        if not isinstance(roi_pct, dict):
            return None
        required = {"x1", "y1", "x2", "y2"}
        if not required.issubset(roi_pct):
            return None
        return raw

    def save_roi_config(self, roi: tuple[int, int, int, int]) -> None:
        try:
            screenshot = ImageGrab.grab(all_screens=True)
            frame_size = screenshot.size
        except Exception:
            return

        data = {
            "locked_roi_pct": normalize_roi(roi, frame_size),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source_size": {"width": frame_size[0], "height": frame_size[1]},
        }
        try:
            self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            fallback = Path.home() / ".box_detector_client" / "config.json"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.config_path = fallback

    def clear_roi_config(self) -> None:
        try:
            self.config_path.unlink(missing_ok=True)
        except Exception:
            pass

    def get_current_capture_size(self) -> tuple[int, int] | None:
        try:
            screenshot = ImageGrab.grab(all_screens=True)
        except Exception:
            return None
        return screenshot.size

    def confirm_startup_roi_choice(self) -> None:
        raw = self.load_roi_config()
        if raw is None:
            return

        frame_size = self.get_current_capture_size()
        if frame_size is None:
            self.status_label.setText("读取已保存检测框失败，启动后将重新整屏找坐标")
            return

        locked_roi = denormalize_roi(raw["locked_roi_pct"], frame_size)
        if locked_roi is None:
            self.status_label.setText("已保存检测框无效，启动后将重新整屏找坐标")
            return

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("检测框配置")
        msg.setText("检测到 config.json，是否使用当前保存的检测框坐标？")
        msg.setInformativeText("选择“使用当前坐标”会直接按保存的位置检测；选择“重新找坐标”会清除旧坐标并重新整屏定位。")
        use_button = msg.addButton("使用当前坐标", QtWidgets.QMessageBox.AcceptRole)
        refind_button = msg.addButton("重新找坐标", QtWidgets.QMessageBox.DestructiveRole)
        msg.setDefaultButton(use_button)
        msg.exec_()

        if msg.clickedButton() is use_button:
            self._set_all_roi(locked_roi)
            self.status_label.setText("已加载 config.json 中的检测框，启动后直接按当前坐标检测")
            return

        self._reset_all_roi()
        self.clear_roi_config()
        self.status_label.setText("已清除保存的检测框，启动后将重新整屏找坐标")

    def build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("当前检测账号")
        font = label.font()
        font.setPointSize(10)
        font.setBold(True)
        label.setFont(font)
        top.addWidget(label)

        select_roi_button = QtWidgets.QPushButton("框选区域")
        select_roi_button.clicked.connect(self.start_region_selection)
        top.addWidget(select_roi_button)

        reset_roi_button = QtWidgets.QPushButton("重置检测框")
        reset_roi_button.clicked.connect(self.reset_detection_roi)
        top.addWidget(reset_roi_button)

        self.toggle_button = QtWidgets.QPushButton("暂停检测")
        self.toggle_button.clicked.connect(self.toggle_monitoring)
        top.addWidget(self.toggle_button)

        add_button = QtWidgets.QPushButton("+ 添加账号")
        add_button.clicked.connect(self.add_account)
        top.addWidget(add_button)
        root.addLayout(top)

        param_group = QtWidgets.QGroupBox("算法参数")
        param_group.setCheckable(True)
        param_group.toggled.connect(lambda on: self._toggle_param_group(param_group, on))
        grid = QtWidgets.QGridLayout(param_group)
        grid.setSpacing(6)
        headers = ["", "启用", "matches≥", "inliers≥", "ratio<", "score≥"]
        for ci, h in enumerate(headers):
            lbl = QtWidgets.QLabel(h)
            lbl.setStyleSheet("font-size: 10px; color: #888;")
            grid.addWidget(lbl, 0, ci)

        self.algo_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self.algo_params: dict[str, dict] = {}
        for ri, (key, name) in enumerate(ALGORITHM_LABELS.items()):
            d = self.detectors[key]
            grid.addWidget(QtWidgets.QLabel(name), ri + 1, 0)

            cb = QtWidgets.QCheckBox()
            cb.setChecked(key in self.active_algorithms)
            cb.toggled.connect(lambda checked, k=key: self._on_algo_toggled(k, checked))
            self.algo_checkboxes[key] = cb
            grid.addWidget(cb, ri + 1, 1)

            match_spin = QtWidgets.QSpinBox()
            match_spin.setRange(4, 50)
            match_spin.setValue(d.min_matches)
            match_spin.valueChanged.connect(lambda v, k=key: self._update_algo_param(k, "min_matches", v))
            grid.addWidget(match_spin, ri + 1, 2)

            inlier_spin = QtWidgets.QSpinBox()
            inlier_spin.setRange(3, 50)
            inlier_spin.setValue(d.min_inliers)
            inlier_spin.valueChanged.connect(lambda v, k=key: self._update_algo_param(k, "min_inliers", v))
            grid.addWidget(inlier_spin, ri + 1, 3)

            ratio_spin = QtWidgets.QDoubleSpinBox()
            ratio_spin.setRange(0.50, 0.95)
            ratio_spin.setSingleStep(0.01)
            ratio_spin.setDecimals(2)
            ratio_spin.setValue(d.ratio)
            ratio_spin.valueChanged.connect(lambda v, k=key: self._update_algo_param(k, "ratio", v))
            grid.addWidget(ratio_spin, ri + 1, 4)

            score_spin = QtWidgets.QDoubleSpinBox()
            score_spin.setRange(0.05, 0.95)
            score_spin.setSingleStep(0.05)
            score_spin.setDecimals(2)
            score_spin.setValue(d.min_template_score)
            score_spin.valueChanged.connect(lambda v, k=key: self._update_algo_param(k, "min_template_score", v))
            grid.addWidget(score_spin, ri + 1, 5)

            self.algo_params[key] = {"matches": match_spin, "inliers": inlier_spin, "ratio": ratio_spin, "score": score_spin}
        self._toggle_param_group(param_group, False)
        root.addWidget(param_group)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        root.addWidget(self.tab_widget, 1)

        for account in self.accounts:
            self.add_account_tab(account)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.monitor_label)
        bottom.addWidget(self.status_label, 1)
        self.algo_info_label = QtWidgets.QLabel()
        self.algo_info_label.setStyleSheet("color: #666666;")
        self._update_algo_info()
        bottom.addWidget(self.algo_info_label)
        root.addLayout(bottom)


    def _update_algo_info(self) -> None:
        names = [ALGORITHM_LABELS[a] for a in sorted(self.active_algorithms)]
        self.algo_info_label.setText(f"算法: {', '.join(names)} | 模板: {self.template_path.name}")

    def _toggle_param_group(self, group: QtWidgets.QGroupBox, visible: bool) -> None:
        for child in group.findChildren(QtWidgets.QWidget):
            if child is not group:
                child.setVisible(visible)

    def _on_algo_toggled(self, algo: str, checked: bool) -> None:
        if checked:
            self.active_algorithms.add(algo)
        else:
            if len(self.active_algorithms) <= 1:
                self.algo_checkboxes[algo].blockSignals(True)
                self.algo_checkboxes[algo].setChecked(True)
                self.algo_checkboxes[algo].blockSignals(False)
                return
            self.active_algorithms.discard(algo)
        self._update_algo_info()

    def _update_algo_param(self, algo: str, param: str, value: float) -> None:
        d = self.detectors[algo]
        if param == "min_matches":
            d.min_matches = int(value)
        elif param == "min_inliers":
            d.min_inliers = int(value)
        elif param == "ratio":
            d.ratio = float(value)
        elif param == "min_template_score":
            d.min_template_score = float(value)

    def _sync_roi(self) -> tuple[int, int, int, int] | None:
        for d in self.detectors.values():
            if d.has_locked_roi():
                return d.locked_roi
        return None

    def _set_all_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        for d in self.detectors.values():
            d.locked_roi = roi

    def _reset_all_roi(self) -> None:
        for d in self.detectors.values():
            d.reset_locked_roi()

    def add_account_tab(self, account: AccountState) -> None:
        aid = account.account_id
        page = AccountPage(
            on_hits_changed=lambda v, a=aid: self.set_account_hits(a, v),
            on_pity_triggered=lambda checked=False, a=aid: self.trigger_pity(a),
            on_clear_clicked=lambda checked=False, a=aid: self.clear_account(a),
        )
        self.account_pages[aid] = page
        self.tab_widget.addTab(page, account.name)
        self.refresh_account(aid)

    def add_account(self) -> None:
        with self.state_lock:
            next_index = len(self.accounts) + 1
            account = AccountState(
                account_id=f"account_{next_index}",
                name=f"账号{next_index}",
            )
            self.accounts.append(account)
            self.save_state()

        self.add_account_tab(account)
        self.tab_widget.setCurrentIndex(len(self.accounts) - 1)
        self.status_label.setText(f"已新增 {account.name}")

    def on_tab_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.accounts):
            return
        self.active_account_id = self.accounts[index].account_id
        self.refresh_current_page()
        self.status_label.setText(f"当前账号切换为 {self.accounts[index].name}")

    def get_active_account_id(self) -> str:
        return self.active_account_id

    def get_account_by_id(self, account_id: str) -> AccountState:
        for account in self.accounts:
            if account.account_id == account_id:
                return account
        raise KeyError(account_id)

    def build_account_summary(self, account: AccountState) -> str:
        progress = account.pity_progress
        return (
            f"{account.name} | 累计命中 {account.total_hits} 次 | "
            f"本轮保底进度 {progress}/{PITY_LIMIT} | "
            f"距离保底 {account.remaining_to_pity} 次 | "
            f"已触发保底 {account.pity_cycles} 次"
        )

    def format_relative_time(self, captured_at: datetime) -> str:
        delta = datetime.now() - captured_at
        seconds = max(int(delta.total_seconds()), 0)
        if seconds < 60:
            return f"{seconds}秒前"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}分钟前"
        return captured_at.strftime("%H:%M")

    def refresh_account(self, account_id: str) -> None:
        account = self.get_account_by_id(account_id)
        page = self.account_pages[account_id]
        page.set_summary(self.build_account_summary(account))
        page.set_hits(account.total_hits)
        widgets = [
            CaptureCard(
                image_path=record.image_path,
                relative_text=self.format_relative_time(record.captured_dt),
                exact_text=record.captured_dt.strftime("%Y-%m-%d %H:%M:%S"),
                locked_roi=record.locked_roi,
                gif_path=record.gif_path,
                on_open_clicked=self.open_image_preview,
                on_delete_clicked=lambda checked=False, a=account_id, r=record: self.delete_capture(a, r),
            )
            for record in account.captures
        ]
        page.set_captures(widgets)

    def refresh_current_page(self) -> None:
        self.refresh_account(self.active_account_id)

    def open_image_preview(self, image_path: str, locked_roi: tuple[int, int, int, int] | None = None) -> None:
        dialog = ImagePreviewDialog(image_path=image_path, locked_roi=locked_roi, parent=self)
        self.preview_dialogs.append(dialog)
        dialog.finished.connect(lambda _result, d=dialog: self._forget_preview_dialog(d))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _forget_preview_dialog(self, dialog: ImagePreviewDialog) -> None:
        if dialog in self.preview_dialogs:
            self.preview_dialogs.remove(dialog)

    def should_run_detection(self) -> bool:
        return not self.foreground_pause

    def update_foreground_pause_state(self) -> None:
        app = QtWidgets.QApplication.instance()
        active_window = app.activeWindow() if app is not None else None
        self.foreground_pause = active_window is not None
        if self.foreground_pause and self.monitoring_enabled:
            self.monitor_label.setText("前台暂停")
        elif self.monitoring_enabled:
            self.monitor_label.setText("运行中")

    def toggle_monitoring(self) -> None:
        self.monitoring_enabled = not self.monitoring_enabled
        self.worker.set_enabled(self.monitoring_enabled)
        if self.monitoring_enabled:
            self.monitor_label.setText("前台暂停" if self.foreground_pause else "运行中")
            self.toggle_button.setText("暂停检测")
            self.status_label.setText("检测已恢复")
        else:
            self.monitor_label.setText("已暂停")
            self.toggle_button.setText("继续检测")
            self.status_label.setText("检测已暂停")

    def reset_detection_roi(self) -> None:
        self._reset_all_roi()
        self.clear_roi_config()
        self.status_label.setText("固定检测框和 config.json 已重置，下一次将重新整屏检测并锁定")

    def start_region_selection(self) -> None:
        self.showMinimized()
        QtCore.QTimer.singleShot(300, self._show_region_selector)

    def _show_region_selector(self) -> None:
        screenshot = ImageGrab.grab(all_screens=True)
        frame_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        selector = RegionSelector(frame_bgr)
        if selector.exec_() == QtWidgets.QDialog.Accepted and selector.selected_roi is not None:
            roi = selector.selected_roi
            self._set_all_roi(roi)
            self.save_roi_config(roi)
            self.status_label.setText(f"已锁定手动框选区域: {roi}")
        else:
            self.status_label.setText("已取消框选")
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def set_account_hits(self, account_id: str, value: int) -> None:
        with self.state_lock:
            account = self.get_account_by_id(account_id)
            account.total_hits = value
            self.save_state()
        self.refresh_account(account_id)

    def trigger_pity(self, account_id: str) -> None:
        with self.state_lock:
            account = self.get_account_by_id(account_id)
            account.pity_cycles += 1
            self.save_state()
        self.refresh_account(account_id)
        self.status_label.setText(f"{account.name} 手动触发保底，进度已清零")

    def delete_capture(self, account_id: str, record: CaptureRecord) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "确认删除", "确定要删除这张截图吗？")
        if reply != QtWidgets.QMessageBox.Yes:
            return
        account = self.get_account_by_id(account_id)
        if record in account.captures:
            account.captures.remove(record)
        try:
            Path(record.image_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if record.gif_path:
                Path(record.gif_path).unlink(missing_ok=True)
        except Exception:
            pass
        self.save_state()
        self.refresh_account(account_id)
        self.status_label.setText("截图已删除")

    def clear_account(self, account_id: str) -> None:
        account = self.get_account_by_id(account_id)
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认清除",
            f"要清除 {account.name} 的全部截图和保底记录吗？",
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        for record in account.captures:
            try:
                Path(record.image_path).unlink(missing_ok=True)
            except Exception:
                pass

        account.captures.clear()
        account.total_hits = 0
        account.pity_cycles = 0
        self.save_state()
        self.refresh_account(account_id)
        self.status_label.setText(f"{account.name} 已清空")

    def process_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                return

            event_type = event[0]
            if event_type == "status":
                self.status_label.setText(event[1])
            elif event_type == "capture":
                _, account_id, record, good_matches, inliers = event
                account = self.get_account_by_id(account_id)
                pity_triggered = account.register_hit(record)
                if record.locked_roi is not None:
                    self.save_roi_config(record.locked_roi)
                self.save_state()
                self.refresh_account(account_id)
                if pity_triggered:
                    self.status_label.setText(
                        f"{account.name} 命中一次并触发保底 | good={good_matches} inliers={inliers} | 累计 {account.total_hits}"
                    )
                else:
                    self.status_label.setText(
                        f"{account.name} 命中一次 | good={good_matches} inliers={inliers} | 累计 {account.total_hits}"
                    )

    def load_state(self) -> list[AccountState]:
        if not self.state_path.exists():
            return []
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        accounts: list[AccountState] = []
        for item in raw.get("accounts", []):
            captures = []
            for capture in item.get("captures", []):
                locked_roi = capture.get("locked_roi")
                captures.append(
                    CaptureRecord(
                        id=capture["id"],
                        image_path=capture["image_path"],
                        captured_at=capture["captured_at"],
                        locked_roi=tuple(locked_roi) if locked_roi else None,
                        gif_path=capture.get("gif_path", ""),
                    )
                )
            accounts.append(
                AccountState(
                    account_id=item["account_id"],
                    name=item["name"],
                    total_hits=item.get("total_hits", 0),
                    pity_cycles=item.get("pity_cycles", 0),
                    captures=captures,
                )
            )
        return accounts

    def save_state(self) -> None:
        data = {
            "accounts": [
                {
                    "account_id": account.account_id,
                    "name": account.name,
                    "total_hits": account.total_hits,
                    "pity_cycles": account.pity_cycles,
                    "captures": [
                        {
                            "id": record.id,
                            "image_path": record.image_path,
                            "captured_at": record.captured_at,
                            "locked_roi": list(record.locked_roi) if record.locked_roi is not None else None,
                            "gif_path": record.gif_path,
                        }
                        for record in account.captures
                    ],
                }
                for account in self.accounts
            ]
        }
        try:
            self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            fallback = Path.home() / ".box_detector_client" / "state.json"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.state_path = fallback

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.worker.stop()
        self.worker.join(timeout=1.5)
        super().closeEvent(event)


def run_self_test(detector, data_dir: Path) -> int:
    lines: list[str] = []
    passed = 0
    for image_path in sorted(data_dir.glob("*.png")):
        detector.reset_locked_roi()
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        result = detector.detect(frame_bgr)
        status = "PASS" if result.found else "FAIL"
        passed += int(result.found)
        lines.append(
            f"{image_path.name}\t{status}\t{result.debug_text or f'good={result.good_matches} inliers={result.inliers} score={result.template_score:.2f}'}"
        )
    print("\n".join(lines))
    print(f"Self-test: {passed}/{len(lines)} passed")
    return 0 if passed == len(lines) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="箱子保底检测客户端")
    parser.add_argument("--template", type=Path, default=Path("标准盒子.png"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--algorithm", choices=list(ALGORITHM_LABELS.keys()), default="orb")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = app_root()
    resource_dir = bundled_root()
    template_path = args.template if args.template.is_absolute() else resource_dir / args.template

    if args.self_test:
        data_dir = args.data_dir if args.data_dir.is_absolute() else base_dir / args.data_dir
        detector = create_detector(args.algorithm, template_path)
        return run_self_test(detector=detector, data_dir=data_dir)

    if sys.platform.startswith("win"):
        os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QtWidgets.QApplication(sys.argv)
    window = BoxDetectorWindow(
        template_path=template_path,
        resource_dir=resource_dir,
        storage_dir=resolve_storage_dir(base_dir),
        state_path=resolve_state_path(base_dir),
        config_path=resolve_config_path(base_dir),
        algorithm=args.algorithm,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
