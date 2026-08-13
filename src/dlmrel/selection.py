"""Select/dev-only head locking and locked-head evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .artifacts import SelectionLock, canonical_hash

SCORE_KEYS = ["relation", "layer", "head"]


def rank_candidates(select_scores: pd.DataFrame, *, relation: str, top_k: int = 5) -> pd.DataFrame:
    required = {*SCORE_KEYS, "accuracy", "n_total"}
    if missing := required - set(select_scores):
        raise ValueError(f"select scores missing columns: {sorted(missing)}")
    frame = select_scores[select_scores["relation"] == relation].copy()
    if frame.empty:
        raise ValueError(f"no select scores for relation {relation!r}")
    return (
        frame.sort_values(
            ["accuracy", "n_total", "layer", "head"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def choose_on_dev(
    candidates: pd.DataFrame, dev_scores: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Choose among select top-K using dev only and a deterministic tie-break."""
    candidate_keys = candidates[SCORE_KEYS]
    scored = candidate_keys.merge(dev_scores, on=SCORE_KEYS, how="left", validate="one_to_one")
    if scored["accuracy"].isna().any():
        raise ValueError("dev scores are missing one or more select candidates")
    scored = scored.sort_values(
        ["accuracy", "n_total", "layer", "head"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return scored.iloc[0], scored


def create_selection_lock(
    select_scores: pd.DataFrame,
    dev_scores: pd.DataFrame,
    *,
    relation: str,
    top_k: int,
    track: str,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    config_hash: str,
    select_manifest_hash: str,
    dev_manifest_hash: str,
) -> tuple[SelectionLock, pd.DataFrame, pd.DataFrame]:
    candidates = rank_candidates(select_scores, relation=relation, top_k=top_k)
    winner, dev_candidates = choose_on_dev(candidates, dev_scores)
    lock = SelectionLock.create(
        track=track,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        relation=relation,
        layer=int(winner["layer"]),
        head=int(winner["head"]),
        top_k=top_k,
        metric="receiver_span_top1_accuracy",
        decision_rule="select top-K; maximize dev accuracy then denominator",
        tie_break="lowest layer then lowest head",
        config_hash=config_hash,
        select_manifest_hash=select_manifest_hash,
        dev_manifest_hash=dev_manifest_hash,
        candidate_scores_hash=canonical_hash(
            {"select": candidates.to_dict("records"), "dev": dev_candidates.to_dict("records")}
        ),
    )
    return lock, candidates, dev_candidates


def locked_test_view(test_rows: pd.DataFrame, lock: SelectionLock) -> pd.DataFrame:
    """Return only rows for the locked head; all-head test rankings are forbidden."""
    required = {"relation", "layer", "head"}
    if missing := required - set(test_rows):
        raise ValueError(f"test rows missing columns: {sorted(missing)}")
    mask = (
        (test_rows["relation"] == lock.relation)
        & (test_rows["layer"] == lock.layer)
        & (test_rows["head"] == lock.head)
    )
    return test_rows.loc[mask].copy().reset_index(drop=True)


def write_lock_bundle(
    path: str | Path, lock: SelectionLock, select: pd.DataFrame, dev: pd.DataFrame
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    lock.write_once(path / "selection_lock.json")
    select.to_csv(path / "select_candidates.csv", index=False)
    dev.to_csv(path / "dev_candidates.csv", index=False)
