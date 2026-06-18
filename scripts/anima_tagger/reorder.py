"""Reorder comma-separated Danbooru tags in Anima Base caption slot order.

Walks ``image_dataset/**/*.txt``, classifies each tag via the corpus tag cache
(with Danbooru ``/tags.json`` fallback for unknowns), drops commentary and
quality tags, prefixes the configured artist with ``@``, and overwrites the
sidecars in place.

Slot order::

    rating → metadata → count → character → copyright → artist → general

Auth for API fallback (optional — only used when a tag is missing from the
local cache): set ``DANBOORU_LOGIN`` and ``DANBOORU_API_KEY`` in the environment
or ``.env`` (HTTP Basic Auth).

Usage::

    python -m scripts.anima_tagger.reorder
    python -m scripts.anima_tagger.reorder --dry-run
    python -m scripts.anima_tagger.reorder --src image_dataset --tag_cache /path/.tag_cache.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from library.captioning.anima_tagger import TAG_TYPE_NAMES
from library.captioning.taxonomy import (
    CAPTION_RATINGS,
    is_artist_tag,
    is_count_tag,
    strip_artist_prefix,
)
from library.env import load_dotenv, resolve_under_home
from library.io.walk import safe_walk

from .vocab import categorize, load_tag_cache

logger = logging.getLogger(__name__)

# Edit this to match the artist whose tags should receive an ``@`` prefix.
ARTIST_TAG = "example_artist"

REORDER_SLOTS: Tuple[str, ...] = (
    "rating",
    "metadata",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)

_QUALITY_DROP = frozenset(
    {
        "masterpiece",
        "best quality",
        "worst quality",
        "low quality",
        "normal quality",
        "highres",
        "absurdres",
        "lowres",
        "incredibly absurdres",
        "incredible absurdres",
    }
)
_SCORE_RE = re.compile(r"^score_\d+$", re.I)
_YEAR_RE = re.compile(r"^\d{4}$")
_COUNT_EXTRA = frozenset({"solo"})

_DANBOORU_TAGS_URL = "https://danbooru.donmai.us/tags.json"
_USER_AGENT = "anima-lora-reorder/1.0"


def _corpus_default(rel: str) -> Optional[str]:
    root = os.environ.get("CAPTION_CORPUS_DIR")
    if not root:
        return None
    return str(Path(root) / rel)


def _should_drop(tag: str) -> bool:
    lower = tag.lower()
    if "commentary" in lower:
        return True
    if lower in _QUALITY_DROP or _SCORE_RE.match(lower):
        return True
    return False


def _matches_artist_tag(bare: str) -> bool:
    if not ARTIST_TAG:
        return False
    return bare.lower() == ARTIST_TAG.lower()


def _format_artist(tag: str, *, cache_category: Optional[str] = None) -> str:
    """Return the artist tag with Anima's ``@`` prefix when appropriate."""
    bare = strip_artist_prefix(tag)
    if _matches_artist_tag(bare):
        canonical = ARTIST_TAG.replace("_", " ")
        return f"@{canonical}"
    if is_artist_tag(tag):
        return tag
    if cache_category == "artist":
        return f"@{bare}" if not tag.startswith("@") else tag
    return tag


class TagClassifier:
    """Resolve Danbooru tag names to reorder slots via cache + optional API."""

    def __init__(
        self,
        cache: Dict[str, str],
        *,
        login: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._cache = cache
        self._login = login
        self._api_key = api_key
        self._session_cache: Dict[str, str] = {}

    @classmethod
    def from_paths(
        cls,
        tag_cache_path: Optional[Path],
        *,
        login: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "TagClassifier":
        cache: Dict[str, str] = {}
        if tag_cache_path is not None and tag_cache_path.exists():
            cache = load_tag_cache(tag_cache_path)
            logger.info("loaded tag cache (%d entries) from %s", len(cache), tag_cache_path)
        elif tag_cache_path is not None:
            logger.warning("tag cache not found at %s — API/heuristics only", tag_cache_path)
        return cls(cache, login=login, api_key=api_key)

    def lookup_category(self, tag: str) -> str:
        """Return raw Danbooru category name for ``tag`` (may differ from reorder slot)."""
        bare = strip_artist_prefix(tag.strip().lower())
        if bare in self._cache:
            return self._cache[bare]
        if bare in self._session_cache:
            return self._session_cache[bare]
        fetched = self._fetch_category(bare)
        self._session_cache[bare] = fetched or "general"
        return self._session_cache[bare]

    def _fetch_category(self, bare: str) -> Optional[str]:
        if not self._login or not self._api_key:
            return None
        api_name = bare.replace(" ", "_")
        params = urllib.parse.urlencode({"search[name]": api_name, "limit": 1})
        url = f"{_DANBOORU_TAGS_URL}?{params}"
        token = base64.b64encode(f"{self._login}:{self._api_key}".encode()).decode("ascii")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Authorization": f"Basic {token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            logger.debug("danbooru lookup failed for %r: %s", bare, exc)
            return None
        if not rows:
            return None
        row = rows[0]
        row_name = row.get("name", "").replace("_", " ")
        if row_name != bare and row.get("name") != api_name:
            return None
        type_id = row.get("category")
        if type_id is None:
            return None
        return TAG_TYPE_NAMES.get(int(type_id), "general")

    def slot_for_tag(self, tag: str) -> Optional[str]:
        """Map a tag to a reorder slot, or ``None`` if it should be dropped."""
        tag = tag.strip().lower()
        if not tag or _should_drop(tag):
            return None
        if tag in CAPTION_RATINGS:
            return "rating"
        if is_count_tag(tag) or tag in _COUNT_EXTRA:
            return "count"
        bare = strip_artist_prefix(tag)
        if is_artist_tag(tag) or _matches_artist_tag(bare):
            return "artist"

        cat = categorize(tag, self._cache)
        if cat == "general" and bare not in self._cache:
            cat = self.lookup_category(tag)
        elif bare in self._session_cache:
            cat = self._session_cache[bare]

        if cat == "metadata" or _YEAR_RE.match(tag):
            return "metadata"
        if cat == "character":
            return "character"
        if cat == "copyright":
            return "copyright"
        if cat == "artist":
            return "artist"
        return "general"


def parse_tags(text: str) -> List[str]:
    """Split a caption string into normalized lowercase tags."""
    text = text.strip()
    if not text:
        return []
    return [t.strip().lower() for t in text.split(",") if t.strip()]


def reorder_tags(tags: Iterable[str], classifier: TagClassifier) -> List[str]:
    """Classify and reorder tags into Anima Base slot order."""
    buckets: Dict[str, List[str]] = {slot: [] for slot in REORDER_SLOTS}

    for raw in tags:
        tag = raw.strip().lower()
        if not tag:
            continue
        slot = classifier.slot_for_tag(tag)
        if slot is None:
            continue
        if slot == "artist":
            cat = classifier.lookup_category(tag)
            buckets[slot].append(_format_artist(tag, cache_category=cat))
        else:
            buckets[slot].append(tag)

    out: List[str] = []
    for slot in REORDER_SLOTS:
        out.extend(buckets[slot])
    return out


def reorder_caption(text: str, classifier: TagClassifier) -> str:
    """Return a reordered caption string."""
    tags = reorder_tags(parse_tags(text), classifier)
    return ", ".join(tags)


def _iter_caption_files(src: Path) -> Iterable[Path]:
    root = Path(os.path.realpath(src))
    for dirpath, _dirnames, filenames in safe_walk(root, followlinks=True):
        for name in filenames:
            if name.endswith(".txt"):
                yield Path(dirpath) / name


def _process_tree(
    src: Path,
    classifier: TagClassifier,
    *,
    dry_run: bool,
) -> Tuple[int, int]:
    n_seen = 0
    n_changed = 0
    for path in sorted(_iter_caption_files(src)):
        n_seen += 1
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("skip %s: %s", path, exc)
            continue
        reordered = reorder_caption(original, classifier)
        if reordered == original.strip():
            continue
        n_changed += 1
        rel = os.path.relpath(path, src)
        if dry_run:
            logger.info("[dry-run] %s\n  was: %s\n  now: %s", rel, original.strip(), reordered)
        else:
            path.write_text(reordered + ("\n" if original.endswith("\n") else ""), encoding="utf-8")
            logger.info("updated %s", rel)
    return n_seen, n_changed


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Reorder Danbooru caption tags into Anima Base slot order."
    )
    p.add_argument(
        "--src",
        default="image_dataset",
        help="Root directory of caption .txt sidecars (default: image_dataset).",
    )
    p.add_argument(
        "--tag_cache",
        default=_corpus_default("retrieved/.tag_cache.json"),
        help="Path to .tag_cache.json (default: $CAPTION_CORPUS_DIR/retrieved/.tag_cache.json).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without overwriting caption files.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    src = resolve_under_home(args.src)
    if not src.exists():
        raise SystemExit(f"source directory not found: {src}")

    cache_path = Path(args.tag_cache).expanduser() if args.tag_cache else None
    login = os.environ.get("DANBOORU_LOGIN")
    api_key = os.environ.get("DANBOORU_API_KEY")
    classifier = TagClassifier.from_paths(cache_path, login=login, api_key=api_key)

    n_seen, n_changed = _process_tree(src, classifier, dry_run=args.dry_run)
    logger.info(
        "done: %d caption(s) scanned, %d updated%s",
        n_seen,
        n_changed,
        " (dry-run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
