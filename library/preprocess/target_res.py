"""Recommend ``target_res`` tiers from a dataset's native resolution distribution."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from library.datasets.buckets import ALLOWED_TARGET_RES, choose_edge
from library.datasets.image_utils import glob_images_pathlib


@dataclass
class TargetResAnalysis:
    """Outcome of scanning a source tree for optimal multi-scale tiers."""

    recommended: list[int]
    tier_counts: dict[int, int] = field(default_factory=dict)
    n_scanned: int = 0
    n_usable: int = 0
    n_below_min_pixels: int = 0
    n_unreadable: int = 0
    min_size: tuple[int, int] | None = None
    max_size: tuple[int, int] | None = None
    median_megapixels: float | None = None


def _read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                w, h = im.size
        if w <= 0 or h <= 0:
            return None
        return int(w), int(h)
    except Exception:
        return None


def analyze_target_res(
    image_paths: list[Path],
    *,
    min_pixels: int = 0,
    allowed_tiers: tuple[int, ...] = ALLOWED_TARGET_RES,
) -> TargetResAnalysis:
    """Pick tiers so each image lands in its least-resize bucket.

    For every readable image at or above ``min_pixels``, assigns the tier
    ``choose_edge`` would pick when all allowed tiers are active, then returns
    the sorted unique set of those tiers.
    """
    tier_counts: Counter[int] = Counter()
    usable_sizes: list[tuple[int, int]] = []
    n_below = 0
    n_unreadable = 0

    for path in image_paths:
        size = _read_image_size(path)
        if size is None:
            n_unreadable += 1
            continue
        w, h = size
        if w * h < min_pixels:
            n_below += 1
            continue
        edge = choose_edge(w, h, allowed_tiers)
        tier_counts[edge] += 1
        usable_sizes.append(size)

    recommended = sorted(tier_counts.keys())
    if not recommended and usable_sizes:
        # Fallback: should not happen when allowed_tiers is non-empty.
        recommended = [allowed_tiers[len(allowed_tiers) // 2]]

    megapixels = sorted(w * h / 1_000_000 for w, h in usable_sizes)
    median_mp = megapixels[len(megapixels) // 2] if megapixels else None

    return TargetResAnalysis(
        recommended=recommended,
        tier_counts=dict(sorted(tier_counts.items())),
        n_scanned=len(image_paths),
        n_usable=len(usable_sizes),
        n_below_min_pixels=n_below,
        n_unreadable=n_unreadable,
        min_size=min(usable_sizes, key=lambda s: s[0] * s[1]) if usable_sizes else None,
        max_size=max(usable_sizes, key=lambda s: s[0] * s[1]) if usable_sizes else None,
        median_megapixels=median_mp,
    )


def scan_image_dataset(
    source_dir: Path,
    *,
    recursive: bool = True,
) -> list[Path]:
    """Return sorted image paths under ``source_dir`` (non-image files ignored)."""
    if not source_dir.is_dir():
        return []
    return glob_images_pathlib(source_dir, recursive=recursive)


def format_analysis_report(analysis: TargetResAnalysis, *, source_dir: Path) -> str:
    """Human-readable summary for CLI output."""
    lines = [
        f"Scanned {analysis.n_scanned} image(s) under {source_dir}",
    ]
    if analysis.n_below_min_pixels:
        lines.append(
            f"  excluded {analysis.n_below_min_pixels} below min_pixels threshold"
        )
    if analysis.n_unreadable:
        lines.append(f"  skipped {analysis.n_unreadable} unreadable file(s)")
    lines.append(f"  usable for tier assignment: {analysis.n_usable}")
    if analysis.min_size and analysis.max_size:
        lines.append(
            f"  size range: {analysis.min_size[0]}x{analysis.min_size[1]}"
            f" .. {analysis.max_size[0]}x{analysis.max_size[1]}"
        )
    if analysis.median_megapixels is not None:
        lines.append(f"  median area: {analysis.median_megapixels:.2f} MP")
    if analysis.tier_counts:
        lines.append("  per-tier assignment (all allowed tiers):")
        for edge, count in analysis.tier_counts.items():
            pct = 100.0 * count / analysis.n_usable
            lines.append(f"    {edge}px: {count} ({pct:.1f}%)")
    tiers = " ".join(str(e) for e in analysis.recommended)
    lines.append(f"Recommended target_res: [{tiers}]")
    return "\n".join(lines)
