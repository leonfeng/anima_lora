#!/usr/bin/env python3
"""Scan ``image_dataset/``, recommend ``target_res`` tiers, run full preprocess.

Walks the configured source image directory (default ``image_dataset/``),
ignores non-image files, assigns each image to the tier ``choose_edge`` would
pick when all allowed tiers are active, then invokes ``make preprocess`` with
that minimal tier set.

Wrapped by ``make preprocess-auto``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from library.config.io import load_path_overrides
from library.env import resolve_under_home
from library.preprocess.target_res import (
    analyze_target_res,
    format_analysis_report,
    scan_image_dataset,
)

ROOT = Path(__file__).resolve().parents[2]


def _default_source_dir() -> Path:
    overrides = load_path_overrides()
    raw = overrides.get("source_image_dir", "image_dataset")
    return Path(resolve_under_home(str(raw)))


def _default_min_pixels() -> int:
    overrides = load_path_overrides()
    if overrides.get("drop_lowres_images") is False:
        return 0
    raw = overrides.get("min_pixels", 500_000)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 500_000


def _run_preprocess(target_res: list[int]) -> None:
    argv = ["--target_res", *(str(e) for e in target_res)]
    make = shutil.which("make")
    if make:
        subprocess.run(
            [make, "preprocess", f"ARGS={' '.join(argv)}"],
            cwd=ROOT,
            check=True,
        )
        return
    subprocess.run(
        [sys.executable, str(ROOT / "tasks.py"), "preprocess", *argv],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Image source tree (default: source_image_dir from configs)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Ignore images below this pixel count when analyzing tiers "
        "(default: preprocess.toml min_pixels when drop_lowres_images is on)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recommended tiers and exit without running preprocess",
    )
    args = parser.parse_args()

    source_dir = args.source_dir or _default_source_dir()
    if not source_dir.is_absolute():
        source_dir = Path(resolve_under_home(str(source_dir)))

    min_pixels = _default_min_pixels() if args.min_pixels is None else max(0, args.min_pixels)
    image_paths = scan_image_dataset(source_dir)

    if not image_paths:
        print(f"No images found under {source_dir}. Nothing to preprocess.")
        sys.exit(1)

    analysis = analyze_target_res(image_paths, min_pixels=min_pixels)
    if analysis.n_usable == 0:
        print(
            f"Found {analysis.n_scanned} image path(s) under {source_dir}, "
            "but none were usable for tier analysis "
            f"(below min_pixels={min_pixels}: {analysis.n_below_min_pixels}, "
            f"unreadable: {analysis.n_unreadable})."
        )
        sys.exit(1)

    print(format_analysis_report(analysis, source_dir=source_dir))

    if args.dry_run:
        return

    print("\nRunning preprocess with recommended tiers …")
    _run_preprocess(analysis.recommended)


if __name__ == "__main__":
    main()
