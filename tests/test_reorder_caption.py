"""Tests for scripts/anima_tagger/reorder.py caption slot reordering."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

reorder = importlib.import_module("scripts.anima_tagger.reorder")
TagClassifier = reorder.TagClassifier
reorder_caption = reorder.reorder_caption
reorder_tags = reorder.reorder_tags


@pytest.fixture
def tag_cache() -> dict[str, str]:
    return {
        "nilou (genshin impact)": "character",
        "genshin impact": "copyright",
        "sincos": "artist",
        "blue hair": "general",
        "looking at viewer": "general",
        "safe": "metadata",
        "2024": "metadata",
        "1girl": "general",  # count regex wins in slot_for_tag before cache
        "english commentary": "metadata",
        "masterpiece": "metadata",
        "absurdres": "metadata",
        "score_9": "metadata",
    }


@pytest.fixture
def classifier(tag_cache: dict[str, str]) -> TagClassifier:
    return TagClassifier(tag_cache)


def test_drops_commentary_and_quality(classifier: TagClassifier):
    tags = [
        "1girl",
        "english commentary",
        "masterpiece",
        "absurdres",
        "score_9",
        "blue hair",
    ]
    out = reorder_tags(tags, classifier)
    assert "english commentary" not in out
    assert "masterpiece" not in out
    assert "absurdres" not in out
    assert "score_9" not in out
    assert out == ["1girl", "blue hair"]


def test_slot_order(classifier: TagClassifier):
    caption = (
        "blue hair, explicit, genshin impact, 1girl, nilou (genshin impact), "
        "2024, safe, @sincos, looking at viewer"
    )
    out = reorder_caption(caption, classifier)
    assert out == (
        "explicit, 2024, safe, 1girl, nilou (genshin impact), "
        "genshin impact, @sincos, blue hair, looking at viewer"
    )


def test_artist_tag_prefix(monkeypatch: pytest.MonkeyPatch, tag_cache: dict[str, str]):
    monkeypatch.setattr(reorder, "ARTIST_TAG", "sincos")
    classifier = TagClassifier(tag_cache)
    out = reorder_tags(["1girl", "sincos", "blue hair"], classifier)
    assert out == ["1girl", "@sincos", "blue hair"]


def test_preserves_within_slot_order(classifier: TagClassifier):
    caption = "yellow shirt, blue hair, 1girl, red dress"
    out = reorder_caption(caption, classifier)
    assert out == "1girl, yellow shirt, blue hair, red dress"


def test_solo_goes_to_count_band(classifier: TagClassifier):
    out = reorder_tags(["solo", "1girl", "smile"], classifier)
    assert out[:2] == ["solo", "1girl"]
    assert "smile" in out


def test_should_drop_helpers():
    assert reorder._should_drop("untranslatable_commentary")
    assert reorder._should_drop("commentary_request")
    assert reorder._should_drop("masterpiece")
    assert reorder._should_drop("score_7")
    assert not reorder._should_drop("safe")


def test_year_tag_metadata_without_cache():
    classifier = TagClassifier({})
    out = reorder_tags(["2023", "1girl"], classifier)
    assert out == ["2023", "1girl"]


def test_dry_run_no_write(tmp_path: Path, tag_cache: dict[str, str]):
    src = tmp_path / "image_dataset"
    src.mkdir()
    cap = src / "img.txt"
    cap.write_text("blue hair, 1girl, explicit", encoding="utf-8")
    classifier = TagClassifier(tag_cache)
    n_seen, n_changed = reorder._process_tree(src, classifier, dry_run=True)
    assert n_seen == 1
    assert n_changed == 1
    assert cap.read_text(encoding="utf-8") == "blue hair, 1girl, explicit"
