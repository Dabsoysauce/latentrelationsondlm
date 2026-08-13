import pytest

from dlmrel.artifacts import ArtifactError, initialize_run, merge_shards, validate_run, write_shard


def test_non_overwrite_resume_and_duplicate_shards(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, {"x": 1}, "command", {"select": "abc"})
    with pytest.raises(ArtifactError, match="already exists"):
        initialize_run(run, {"x": 1}, "command", {"select": "abc"})
    initialize_run(run, {"x": 1}, "command", {"select": "abc"}, resume=True)
    write_shard(run, 0, [{"x": 1}])
    write_shard(run, 0, [{"x": 1}])
    with pytest.raises(ArtifactError, match="different content"):
        write_shard(run, 0, [{"x": 2}])
    assert merge_shards(run) == [{"x": 1}]


def test_validator_fails_incomplete_run(tmp_path):
    run = tmp_path / "run"
    initialize_run(run, {"x": 1}, "command", {})
    result = validate_run(run)
    assert not result["valid"]
    assert any("missing files" in error for error in result["errors"])
