"""Official-boundary base manifests and cross-model instance intersections."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .config import DatasetConfig


@dataclass(frozen=True)
class ManifestRow:
    sentence_id: str
    treebank: str
    language: str
    role: str
    original_split: str
    sent_id: str
    normalized_text: str
    text_sha256: str
    n_words: int


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def stable_sentence_id(treebank: str, split: str, sent_id: str, text: str) -> str:
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    return f"{treebank}:{split}:{sent_id}:{digest[:16]}"


def manifest_hash(rows: Iterable[ManifestRow]) -> str:
    payload = [asdict(row) for row in rows]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_official_manifests(
    dataset: DatasetConfig,
    sentences: dict[str, list],
) -> dict[str, list[ManifestRow]]:
    """Sample train/dev/test independently without crossing their boundaries."""
    roles = {"select": dataset.select_from, "dev": dataset.dev_from, "test": dataset.test_from}
    budgets = {"select": dataset.n_select, "dev": dataset.n_dev, "test": dataset.n_test}
    seen_text: set[str] = set()
    output: dict[str, list[ManifestRow]] = {}
    for role in ("select", "dev", "test"):
        original = roles[role]
        candidates: list[ManifestRow] = []
        for index, sentence in enumerate(sentences.get(original, [])):
            metadata = getattr(sentence, "metadata", {})
            text = metadata.get("text", "")
            normalized = normalize_text(text)
            words = [token for token in sentence if isinstance(token.get("id"), int)]
            if not normalized or len(words) < dataset.min_words:
                continue
            if dataset.max_words is not None and len(words) > dataset.max_words:
                continue
            sent_id = str(metadata.get("sent_id") or f"missing-{index}")
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            candidates.append(
                ManifestRow(
                    sentence_id=stable_sentence_id(dataset.treebank, original, sent_id, normalized),
                    treebank=dataset.treebank,
                    language=dataset.language,
                    role=role,
                    original_split=original,
                    sent_id=sent_id,
                    normalized_text=normalized,
                    text_sha256=digest,
                    n_words=len(words),
                )
            )
        random.Random(dataset.seed).shuffle(candidates)
        selected: list[ManifestRow] = []
        for row in candidates:
            if dataset.dedupe_normalized_text and row.text_sha256 in seen_text:
                continue
            seen_text.add(row.text_sha256)
            selected.append(row)
            if budgets[role] is not None and len(selected) >= budgets[role]:
                break
        output[role] = selected
    assert_zero_overlap(output)
    return output


def assert_zero_overlap(manifests: dict[str, list[ManifestRow]]) -> None:
    seen: dict[str, str] = {}
    for role, rows in manifests.items():
        for row in rows:
            previous = seen.setdefault(row.text_sha256, role)
            if previous != role:
                raise ValueError(f"normalized text overlaps {previous} and {role}: {row.sentence_id}")


def common_valid_instances(eligible_ids: dict[str, set[str]]) -> set[str]:
    if not eligible_ids:
        return set()
    values = iter(eligible_ids.values())
    shared = set(next(values))
    for ids in values:
        shared.intersection_update(ids)
    return shared
