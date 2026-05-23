from __future__ import annotations

import argparse
import os
import time
from statistics import mean

from PIL import ImageGrab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure CPU usage of screenshot capture only.")
    parser.add_argument("--interval", type=float, default=0.25, help="Seconds between captures.")
    parser.add_argument("--duration", type=float, default=30.0, help="Total test duration in seconds.")
    parser.add_argument("--report-every", type=float, default=1.0, help="Seconds between reports.")
    parser.add_argument("--all-screens", action="store_true", help="Capture all screens instead of the primary screen.")
    return parser.parse_args()


def format_float(value: float) -> str:
    return f"{value:.2f}"


def main() -> int:
    args = parse_args()
    logical_cores = max(os.cpu_count() or 1, 1)

    print(f"interval={args.interval}s duration={args.duration}s report_every={args.report_every}s")
    print(f"logical_cores={logical_cores} all_screens={args.all_screens}")
    print("Columns: elapsed_s grabs avg_ms min_ms max_ms cpu_single_core_pct cpu_all_cores_pct")

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    report_wall = start_wall
    report_cpu = start_cpu
    report_samples: list[float] = []
    total_grabs = 0

    while True:
        now = time.perf_counter()
        elapsed = now - start_wall
        if elapsed >= args.duration:
            break

        grab_started = time.perf_counter()
        _image = ImageGrab.grab(all_screens=args.all_screens)
        grab_elapsed = time.perf_counter() - grab_started
        report_samples.append(grab_elapsed)
        total_grabs += 1

        after_grab = time.perf_counter()
        report_elapsed = after_grab - report_wall
        if report_elapsed >= args.report_every and report_samples:
            cpu_elapsed = time.process_time() - report_cpu
            cpu_single_core_pct = cpu_elapsed / report_elapsed * 100.0
            cpu_all_cores_pct = cpu_single_core_pct / logical_cores
            print(
                " ".join(
                    [
                        format_float(after_grab - start_wall),
                        str(len(report_samples)),
                        format_float(mean(report_samples) * 1000.0),
                        format_float(min(report_samples) * 1000.0),
                        format_float(max(report_samples) * 1000.0),
                        format_float(cpu_single_core_pct),
                        format_float(cpu_all_cores_pct),
                    ]
                )
            )
            report_wall = after_grab
            report_cpu = time.process_time()
            report_samples.clear()

        remaining = args.interval - (time.perf_counter() - grab_started)
        if remaining > 0:
            time.sleep(remaining)

    total_wall = time.perf_counter() - start_wall
    total_cpu = time.process_time() - start_cpu
    total_cpu_single_core_pct = total_cpu / max(total_wall, 1e-9) * 100.0
    total_cpu_all_cores_pct = total_cpu_single_core_pct / logical_cores
    print(
        "TOTAL "
        + " ".join(
            [
                format_float(total_wall),
                str(total_grabs),
                format_float(total_cpu_single_core_pct),
                format_float(total_cpu_all_cores_pct),
            ]
        )
    )
    print("TOTAL columns: elapsed_s grabs cpu_single_core_pct cpu_all_cores_pct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
