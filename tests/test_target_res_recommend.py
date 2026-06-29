"""Tests for ``library.preprocess.target_res`` tier recommendation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from library.datasets.buckets import ALLOWED_TARGET_RES, choose_edge
from library.preprocess.target_res import (
    analyze_target_res,
    scan_image_dataset,
)


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_scan_image_dataset_ignores_non_images(tmp_path: Path) -> None:
    src = tmp_path / "image_dataset"
    _write_image(src / "a.png", (1024, 1024))
    (src / "readme.txt").write_text("not an image", encoding="utf-8")
    (src / "nested" / "note.md").parent.mkdir(parents=True, exist_ok=True)
    (src / "nested" / "note.md").write_text("also skipped", encoding="utf-8")

    paths = scan_image_dataset(src)
    assert len(paths) == 1
    assert paths[0].name == "a.png"


def test_analyze_target_res_mixed_dataset(tmp_path: Path) -> None:
    src = tmp_path / "images"
    _write_image(src / "small.png", (768, 768))
    _write_image(src / "medium.png", (1024, 1024))
    _write_image(src / "large.png", (1536, 1536))

    paths = scan_image_dataset(src)
    analysis = analyze_target_res(paths, min_pixels=0)

    assert analysis.n_usable == 3
    assert analysis.recommended == sorted(set(analysis.tier_counts))
    for path in paths:
        with Image.open(path) as im:
            w, h = im.size
        expected = choose_edge(w, h, ALLOWED_TARGET_RES)
        assert expected in analysis.recommended


def test_analyze_respects_min_pixels(tmp_path: Path) -> None:
    src = tmp_path / "images"
    _write_image(src / "tiny.png", (200, 200))
    _write_image(src / "ok.png", (1024, 1024))

    analysis = analyze_target_res(scan_image_dataset(src), min_pixels=500_000)
    assert analysis.n_usable == 1
    assert analysis.n_below_min_pixels == 1
    assert analysis.recommended == [1024]


def test_scan_empty_dir_returns_no_images(tmp_path: Path) -> None:
    src = tmp_path / "empty"
    src.mkdir()
    assert scan_image_dataset(src) == []


def test_auto_preprocess_exits_when_no_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts/preprocess/auto_preprocess.py"
    spec = importlib.util.spec_from_file_location("auto_preprocess", script)
    assert spec and spec.loader
    auto_preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auto_preprocess)

    src = tmp_path / "image_dataset"
    src.mkdir()
    monkeypatch.setattr(auto_preprocess, "_default_source_dir", lambda: src)
    monkeypatch.setattr(sys, "argv", ["auto_preprocess.py"])

    with pytest.raises(SystemExit) as exc:
        auto_preprocess.main()
    assert exc.value.code == 1
