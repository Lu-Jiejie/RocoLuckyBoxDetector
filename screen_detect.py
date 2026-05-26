from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


@dataclass
class Template:
    name: str
    image_gray: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray
    width: int
    height: int


@dataclass
class Detection:
    template_name: str
    polygon: np.ndarray
    inliers: int
    good_matches: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime screen detection for the object in ./data using OpenCV."
    )
    parser.add_argument("--templates", type=Path, default=Path("data"))
    parser.add_argument("--ratio", type=float, default=0.72, help="Lowe ratio test threshold.")
    parser.add_argument("--min-matches", type=int, default=12, help="Minimum good matches before homography.")
    parser.add_argument("--min-inliers", type=int, default=8, help="Minimum inliers after homography.")
    parser.add_argument("--screen-scale", type=float, default=1.0, help="Resize factor for the captured screen.")
    parser.add_argument("--fps-limit", type=float, default=8.0, help="Maximum preview FPS.")
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Optional screen region to capture.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an offline test against synthetic canvases and exit.",
    )
    return parser.parse_args()


def create_feature_extractor() -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=1800)


def load_templates(template_dir: Path, sift: cv2.SIFT) -> list[Template]:
    paths = sorted(template_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG templates found in {template_dir}")

    templates: list[Template] = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 8:
            continue
        h, w = image.shape[:2]
        templates.append(
            Template(
                name=path.name,
                image_gray=image,
                keypoints=keypoints,
                descriptors=descriptors,
                width=w,
                height=h,
            )
        )

    if not templates:
        raise RuntimeError("Templates were loaded, but none produced enough features.")
    return templates


def create_matcher() -> cv2.FlannBasedMatcher:
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)


def detect_object(
    frame_bgr: np.ndarray,
    templates: list[Template],
    sift: cv2.SIFT,
    matcher: cv2.FlannBasedMatcher,
    ratio: float,
    min_matches: int,
    min_inliers: int,
) -> Detection | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_desc = sift.detectAndCompute(gray, None)
    if frame_desc is None or len(frame_kp) < min_matches:
        return None

    best: Detection | None = None
    frame_h, frame_w = gray.shape[:2]
    for template in templates:
        pairs = matcher.knnMatch(template.descriptors, frame_desc, k=2)
        good_matches = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        if len(good_matches) < min_matches:
            continue

        src_pts = np.float32(
            [template.keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            continue

        inliers = int(mask.ravel().sum())
        if inliers < min_inliers:
            continue

        corners = np.float32(
            [[0, 0], [template.width - 1, 0], [template.width - 1, template.height - 1], [0, template.height - 1]]
        ).reshape(-1, 1, 2)
        polygon = cv2.perspectiveTransform(corners, homography)

        if not polygon_is_valid(polygon, frame_w, frame_h):
            continue

        score = inliers / max(len(template.keypoints), 1)
        candidate = Detection(
            template_name=template.name,
            polygon=polygon,
            inliers=inliers,
            good_matches=len(good_matches),
            score=score,
        )
        if best is None or (candidate.inliers, candidate.score) > (best.inliers, best.score):
            best = candidate

    return best


def polygon_is_valid(polygon: np.ndarray, frame_w: int, frame_h: int) -> bool:
    pts = polygon.reshape(-1, 2)
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return False

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w < 20 or box_h < 20:
        return False

    area = cv2.contourArea(pts.astype(np.float32))
    frame_area = frame_w * frame_h
    if area < 500 or area > frame_area * 0.85:
        return False

    out_of_bounds = np.sum((pts[:, 0] < -20) | (pts[:, 0] > frame_w + 20) | (pts[:, 1] < -20) | (pts[:, 1] > frame_h + 20))
    return out_of_bounds <= 1


def grab_screen(region: tuple[int, int, int, int] | None, screen_scale: float) -> np.ndarray:
    image = ImageGrab.grab(bbox=region, all_screens=True)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if screen_scale != 1.0:
        frame = cv2.resize(frame, None, fx=screen_scale, fy=screen_scale, interpolation=cv2.INTER_AREA)
    return frame


def draw_detection(frame: np.ndarray, detection: Detection | None, fps: float) -> np.ndarray:
    vis = frame.copy()
    status = "NOT FOUND"
    color = (0, 0, 255)

    if detection is not None:
        polygon = detection.polygon.astype(np.int32)
        cv2.polylines(vis, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)
        x, y = polygon.reshape(-1, 2).min(axis=0)
        status = (
            f"FOUND {detection.template_name} "
            f"inliers={detection.inliers} matches={detection.good_matches} score={detection.score:.3f}"
        )
        color = (0, 255, 0)
        cv2.putText(
            vis,
            detection.template_name,
            (int(x), max(30, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(vis, f"FPS {fps:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)
    return vis


def run_self_test(
    templates: list[Template],
    sift: cv2.SIFT,
    matcher: cv2.FlannBasedMatcher,
    ratio: float,
    min_matches: int,
    min_inliers: int,
) -> int:
    passed = 0
    for template in templates:
        canvas = np.full((900, 1400, 3), 35, dtype=np.uint8)
        sample = cv2.cvtColor(template.image_gray, cv2.COLOR_GRAY2BGR)
        scale = min(1.2, 380 / max(template.width, template.height))
        resized = cv2.resize(sample, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        rh, rw = resized.shape[:2]
        top = (canvas.shape[0] - rh) // 2
        left = (canvas.shape[1] - rw) // 2
        canvas[top : top + rh, left : left + rw] = resized

        detection = detect_object(canvas, templates, sift, matcher, ratio, min_matches, min_inliers)
        ok = detection is not None
        passed += int(ok)
        print(
            f"{template.name}: {'PASS' if ok else 'FAIL'}"
            + (f" -> {detection.template_name}, inliers={detection.inliers}" if ok else "")
        )

    print(f"Self-test: {passed}/{len(templates)} passed")
    return 0 if passed == len(templates) else 1


def main() -> int:
    args = parse_args()
    sift = create_feature_extractor()
    templates = load_templates(args.templates, sift)
    matcher = create_matcher()

    if args.self_test:
        return run_self_test(
            templates,
            sift,
            matcher,
            ratio=args.ratio,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
        )

    frame_interval = 1.0 / args.fps_limit if args.fps_limit > 0 else 0.0
    region = tuple(args.region) if args.region else None
    window_name = "Object Detector"

    while True:
        start = time.perf_counter()
        frame = grab_screen(region=region, screen_scale=args.screen_scale)
        detection = detect_object(
            frame,
            templates,
            sift,
            matcher,
            ratio=args.ratio,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
        )
        elapsed = max(time.perf_counter() - start, 1e-6)
        fps = 1.0 / elapsed
        vis = draw_detection(frame, detection, fps=fps)
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        remaining = frame_interval - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
