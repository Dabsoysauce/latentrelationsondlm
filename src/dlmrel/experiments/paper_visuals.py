"""Evidence generation and plotting for old attention heatmaps/trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..checkpoints import CheckpointIdentity, SentenceCheckpointStore
from ..config import RELATION_NAMES, RunConfig
from ..data import load_manifest_examples
from ..diffusion import attentions_for_state, teacher_forced_trajectory, tokenize
from ..paper_protocol import PaperLockSet, write_resolved_selection_locks
from .shared import write_frames


@dataclass
class HeatmapCase:
    sentence_id: str
    relation: str
    example: Any
    instance: Any


def _qualitative_sentences(path: str | Path) -> list[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "dlmrel-attention-diagnostics-v1":
        raise ValueError("attention diagnostic manifest is incompatible")
    return [str(value) for value in raw["sentences"]]


def _surface_summary(attention: torch.Tensor, punctuation: set[int]):
    # attention: heads x query x source
    sequence_length = attention.shape[-1]
    diagonal = attention.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    previous = torch.stack(
        [attention[:, query, max(query - 1, 0)] for query in range(sequence_length)]
    ).mean(dim=0)
    following = torch.stack(
        [attention[:, query, min(query + 1, sequence_length - 1)] for query in range(sequence_length)]
    ).mean(dim=0)
    punctuation_mass = (
        attention[:, :, sorted(punctuation)].sum(dim=-1).mean(dim=-1)
        if punctuation
        else torch.zeros(attention.shape[0])
    )
    return diagonal, previous, following, punctuation_mass


def fully_visible_evidence(model, tokenizer, sentences: list[str]):
    attention_rows, summary_rows = [], []
    for sentence_index, sentence in enumerate(sentences):
        input_ids, tokens = tokenize(tokenizer, sentence, model.device, True)
        _logits, attentions = model.forward_attentions(input_ids)
        punctuation = {
            index
            for index, token in enumerate(tokens)
            if token.strip() and all(character in ".,;:!?-'\"()[]{}" for character in token.strip())
        }
        for layer, values in enumerate(attentions):
            matrix = values[0].detach().float().cpu()
            diagonal, previous, following, punctuation_mass = _surface_summary(
                matrix, punctuation
            )
            for head in range(matrix.shape[0]):
                attention_rows.append(
                    {
                        "sentence_id": f"qualitative-{sentence_index}",
                        "sentence": sentence,
                        "tokens": tokens,
                        "layer": layer,
                        "head": head,
                        "attention": matrix[head].tolist(),
                    }
                )
                summary_rows.append(
                    {
                        "sentence_id": f"qualitative-{sentence_index}",
                        "layer": layer,
                        "head": head,
                        "self_token_mass": float(diagonal[head]),
                        "previous_token_mass": float(previous[head]),
                        "next_token_mass": float(following[head]),
                        "punctuation_mass": float(punctuation_mass[head]),
                    }
                )
    return pd.DataFrame(attention_rows), pd.DataFrame(summary_rows)


def _frozen_cases(examples, locks: PaperLockSet) -> list[HeatmapCase]:
    cases = []
    seen = set()
    for example in examples:
        for instance in example.relations:
            if instance.relation in seen:
                continue
            locks.resolve(instance.relation)
            cases.append(
                HeatmapCase(
                    sentence_id=f"{example.sentence_id}:{instance.relation}",
                    relation=instance.relation,
                    example=example,
                    instance=instance,
                )
            )
            seen.add(instance.relation)
        if seen == set(RELATION_NAMES):
            break
    if seen != set(RELATION_NAMES):
        missing = sorted(set(RELATION_NAMES) - seen)
        raise ValueError(f"test manifest lacks qualitative cases for {missing}")
    return cases


def trajectory_chunk(model, tokenizer, cases, *, seed: int, locks: PaperLockSet):
    rows = []
    for case in cases:
        lock = locks.resolve(case.relation)
        states = teacher_forced_trajectory(
            model, tokenizer, case.example.text, steps=64, seed=seed, include_bos=True
        )
        for timestep, state in enumerate(states):
            if timestep in {0, 63} and seed != 42:
                continue
            attentions = attentions_for_state(model, state)
            matrix = attentions[lock.layer][0, lock.head].detach().float().cpu()
            probability = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(
                probability * probability.clamp_min(1e-12).log()
            ).sum(dim=-1)[1:].mean()
            labels = [
                f"{token} [{'V' if visible else 'M'}]"
                for token, visible in zip(state.tokens, state.is_visible, strict=True)
            ]
            relation_mass = float(
                matrix[case.instance.attender_span[-1], case.instance.receiver_span].sum()
            )
            rows.append(
                {
                    "sentence_id": case.sentence_id,
                    "source_sentence_id": case.example.sentence_id,
                    "sentence": case.example.text,
                    "relation": case.relation,
                    "direction": (
                        "right"
                        if case.instance.receiver_word_idx > case.instance.attender_word_idx
                        else "left"
                    ),
                    "attender_span": case.instance.attender_span,
                    "receiver_span": case.instance.receiver_span,
                    "layer": lock.layer,
                    "head": lock.head,
                    "seed": seed,
                    "timestep": timestep,
                    "normalized_progress": timestep / 63,
                    "token_labels": labels,
                    "is_visible": state.is_visible,
                    "entropy": float(entropy),
                    "relation_attention_mass": relation_mass,
                    "attention": matrix.tolist(),
                }
            )
    return pd.DataFrame(rows)


def plot_saved_evidence(run_dir: str | Path) -> list[str]:
    """Plot only saved evidence so visual selection cannot change scoring."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    trajectories = pd.read_parquet(run_dir / "trajectory_evidence.parquet")
    outputs = []
    for (relation, seed), group in trajectories.groupby(["relation", "seed"], observed=True):
        path = figure_dir / f"{relation}__seed-{seed}__trajectory.pdf"
        with PdfPages(path) as pdf:
            for row in group.sort_values("timestep").itertuples(index=False):
                matrix = torch.tensor(row.attention).numpy()
                figure, axis = plt.subplots(figsize=(9, 8))
                image = axis.imshow(matrix, cmap="viridis", aspect="auto")
                axis.set_title(
                    f"{relation} L{row.layer}H{row.head} t={row.timestep} "
                    f"entropy={row.entropy:.3f} direction={row.direction}"
                )
                axis.set_xticks(range(len(row.token_labels)), row.token_labels, rotation=90, fontsize=6)
                axis.set_yticks(range(len(row.token_labels)), row.token_labels, fontsize=6)
                for position in row.receiver_span:
                    axis.axvline(position, color="red", linewidth=0.8)
                for position in row.attender_span:
                    axis.axhline(position, color="white", linewidth=0.8)
                figure.colorbar(image, ax=axis)
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)
        outputs.append(str(path))

    full = pd.read_parquet(run_dir / "fully_visible_all_head_attention.parquet")
    for sentence_id, sentence_group in full.groupby("sentence_id", observed=True):
        path = figure_dir / f"{sentence_id}__fully-visible-all-head-grid.pdf"
        with PdfPages(path) as pdf:
            for layer, layer_group in sentence_group.groupby("layer", observed=True):
                heads = list(layer_group.sort_values("head").itertuples(index=False))
                columns = 4
                rows = math.ceil(len(heads) / columns)
                figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
                for axis, head_row in zip(axes.flat, heads, strict=False):
                    axis.imshow(head_row.attention, cmap="viridis", aspect="auto")
                    axis.set_title(f"L{layer}H{head_row.head}")
                    axis.set_xticks([])
                    axis.set_yticks([])
                for axis in axes.flat[len(heads) :]:
                    axis.axis("off")
                figure.suptitle(heads[0].sentence)
                figure.tight_layout()
                pdf.savefig(figure)
                plt.close(figure)
        outputs.append(str(path))
    return outputs


def run(
    model,
    tokenizer,
    cfg: RunConfig,
    run_dir: Path,
    *,
    source_locks: PaperLockSet,
    **_unused: Any,
):
    sentences = _qualitative_sentences(cfg.experiment.settings["qualitative_manifest"])
    full, surface = fully_visible_evidence(model, tokenizer, sentences)
    full.to_parquet(run_dir / "fully_visible_all_head_attention.parquet", index=False)
    surface.to_csv(run_dir / "surface_pattern_summaries.csv", index=False)
    examples, exclusions = load_manifest_examples(cfg, tokenizer, "test")
    cases = _frozen_cases(examples, source_locks)
    pd.DataFrame(
        [
            {
                "case_id": case.sentence_id,
                "source_sentence_id": case.example.sentence_id,
                "relation": case.relation,
                "sentence": case.example.text,
            }
            for case in cases
        ]
    ).to_csv(run_dir / "frozen_trajectory_cases.csv", index=False)
    store = SentenceCheckpointStore(run_dir)
    frames = []
    for seed in cfg.experiment.seeds:
        identity = CheckpointIdentity(
            stage="paper-selected-head-heatmap-trajectories",
            seed=seed,
            normalized_progress=-1.0,
            timestep=-1,
            heads=tuple(sorted(source_locks.heads)),
        )
        frames.append(
            store.run(
                cases,
                identity,
                lambda chunk, _start, current_seed=seed: trajectory_chunk(
                    model, tokenizer, chunk, seed=current_seed, locks=source_locks
                ),
            )
        )
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.to_parquet(run_dir / "trajectory_evidence.parquet", index=False)
    raw = trajectories.drop(columns=["attention", "token_labels", "is_visible"])
    write_frames(run_dir, raw=raw, exclusions=exclusions)
    keys = ["seed", "relation", "layer", "head", "timestep"]
    per_seed = raw.groupby(keys, as_index=False).agg(
        entropy=("entropy", "mean"),
        relation_attention_mass=("relation_attention_mass", "mean"),
        raw_denominator=("sentence_id", "nunique"),
    )
    per_seed.to_csv(run_dir / "per_seed_metrics.csv", index=False)
    per_seed.groupby(keys[1:], as_index=False).agg(
        entropy_mean=("entropy", "mean"),
        entropy_std=("entropy", "std"),
        relation_attention_mass_mean=("relation_attention_mass", "mean"),
        raw_denominator=("raw_denominator", "sum"),
        n_seeds=("seed", "nunique"),
    ).to_csv(run_dir / "metrics.csv", index=False)
    figure_paths = plot_saved_evidence(run_dir)
    write_resolved_selection_locks(run_dir, source_locks)
    return {
        "development_used": False,
        "qualitative_source": "preserved_diffugpt_attention_head_trajectories",
        "qualitative_sentence_count": len(sentences),
        "case_selection": "first_manifest_instance_per_relation_without_performance_access",
        "relations": list(RELATION_NAMES),
        "timesteps": list(range(64)),
        "seeds": list(cfg.experiment.seeds),
        "deterministic_steps_deduplicated": [0, 63],
        "vector_pdf_figures": figure_paths,
        "plotting_reads_saved_evidence_only": True,
    }
