from dlmrel.config import DatasetConfig
from dlmrel.splits import (
    assert_zero_overlap,
    build_official_manifests,
    common_valid_instances,
    manifest_hash,
)


class Sentence(list):
    def __init__(self, sent_id, text, words=4):
        super().__init__({"id": i + 1} for i in range(words))
        self.metadata = {"sent_id": sent_id, "text": text}


def test_official_boundaries_determinism_deduplication():
    dataset = DatasetConfig(revision="abc", n_select=2, n_dev=2, n_test=2)
    sentences = {
        "train": [
            Sentence("tr1", "same text here now"),
            Sentence("tr2", "unique train sentence now"),
        ],
        "dev": [Sentence("dv1", "same text here now"), Sentence("dv2", "unique dev sentence now")],
        "test": [Sentence("te1", "unique test sentence now")],
    }
    first = build_official_manifests(dataset, sentences)
    second = build_official_manifests(dataset, sentences)
    assert [row.original_split for row in first["select"]] == ["train"] * len(first["select"])
    assert [row.original_split for row in first["dev"]] == ["dev"] * len(first["dev"])
    assert [row.original_split for row in first["test"]] == ["test"] * len(first["test"])
    assert manifest_hash(first["select"]) == manifest_hash(second["select"])
    assert sum(row.normalized_text == "same text here now" for rows in first.values() for row in rows) == 1
    assert_zero_overlap(first)


def test_instance_intersection_not_sentence_intersection():
    sets = {"a": {"s1:i1", "s1:i2"}, "b": {"s1:i2", "s2:i1"}}
    assert common_valid_instances(sets) == {"s1:i2"}
