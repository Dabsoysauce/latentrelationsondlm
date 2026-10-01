"""Paired behavioral interventions using existing gold examples and native sampler."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from ..models.decomposition import _layers, projection_module
from ..models.interventions import HeadScale, scale_projection_heads, temporal_gate
from ..models.native import random_reveal_trajectories
from .shared import instance_metadata


@dataclass(frozen=True)
class CausalSettings:
    relations: tuple = ("object_to_verb",)
    seeds: tuple = (42, 43, 44)
    alphas: tuple = (0.0, 1.0)
    windows: tuple = ("entire", "early_pre", "late_pre", "pre", "post")
    controls: tuple = ("random", "same_layer", "low_relation", "other_relation", "matched_attention")
    examples: int = 10
    batch_size: int = 1
    start: int = 0
    stop: int = 64
    step_ranges: tuple = ()
    split: float = 0.5
    temperature: float = 0.95
    top_p: float = 0.9
    control_seed: int = 1729
    bootstrap_samples: int = 2000
    heads: tuple = ()  # optional joint [(layer, head), ...], explicitly exploratory

    @classmethod
    def from_dict(cls, raw):
        if unknown := set(raw) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown causal settings: {sorted(unknown)}")
        result = cls(**raw)
        from ..config import RELATION_NAMES

        for name in ("relations", "seeds", "alphas", "windows", "controls"):
            values = getattr(result, name)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name}")
        if any(type(seed) is not int for seed in result.seeds):
            raise ValueError("seeds must be integers")

        if not result.relations or not set(result.relations) <= set(RELATION_NAMES):
            raise ValueError("use the repository's six exact relation labels")
        if result.examples < 1 or result.batch_size < 1 or result.bootstrap_samples < 100:
            raise ValueError("examples/batch_size must be positive and bootstrap_samples >= 100")
        if not result.seeds or len(set(result.seeds)) != len(result.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if not result.alphas or not np.isfinite(result.alphas).all():
            raise ValueError("alphas must be nonempty and finite")
        if not result.windows or not set(result.windows) <= {
            "entire",
            "pre",
            "post",
            "early_pre",
            "late_pre",
        }:
            raise ValueError("unknown temporal window")
        if not set(result.controls) <= {
            "random",
            "same_layer",
            "low_relation",
            "other_relation",
            "matched_attention",
        }:
            raise ValueError("unknown head control")
        if not 0 <= result.start < result.stop <= 64 or not 0 < result.split < 1:
            raise ValueError("invalid temporal bounds")
        if any(len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 64 for bounds in result.step_ranges):
            raise ValueError("step_ranges must contain half-open [start, stop] pairs in [0,64]")
        if result.temperature <= 0 or not 0 < result.top_p <= 1:
            raise ValueError("invalid sampling settings")
        if len(set(map(tuple, result.heads))) != len(result.heads):
            raise ValueError("duplicate explicitly requested heads")
        return result


def select_controls(model, locks, relation, scores, settings):
    """Select once from selection evidence, never from intervention outcomes."""
    target = locks.resolve(relation)
    targets = tuple(map(tuple, settings.heads)) or ((target.layer, target.head),)
    universe = [
        (layer, head)
        for layer in range(len(_layers(model)[0]))
        for head in range(projection_module(model, layer)[1])
    ]
    if not set(targets) <= set(universe):
        raise ValueError("requested head outside model")
    excluded = locks.heads | set(targets)
    available = [h for h in universe if h not in excluded]
    rng = np.random.default_rng(settings.control_seed)
    selected = {"target": targets}
    audit = []
    for kind in settings.controls:
        pool = available
        reason = ""
        if kind == "same_layer":
            # Joint interventions preserve the number of heads in each layer.
            picked = []
            for layer, _head in targets:
                candidates = [h for h in available if h[0] == layer and h not in picked]
                if not candidates:
                    picked = []
                    break
                picked.append(candidates[int(rng.integers(len(candidates)))])
            pool = picked
        elif kind == "other_relation":
            pool = sorted(locks.heads - set(targets))
        elif kind in {"low_relation", "matched_attention"}:
            if scores is None:
                pool, reason = [], "selection_all_head_scores.csv unavailable"
            else:
                frame = scores[scores.relation == relation].copy()
                frame = frame[[tuple(v) in available for v in frame[["layer", "head"]].to_numpy()]]
                if kind == "matched_attention":
                    frame = frame[frame.accuracy <= frame.accuracy.quantile(0.25)]
                    cols = [c for c in ("attention_entropy", "attention_magnitude") if c in scores]
                    if not cols or len(targets) != 1:
                        frame = frame.iloc[:0]
                        reason = "requires selection-only attention metadata and a single target head"
                    else:
                        ref = scores[
                            (scores.relation == relation)
                            & (scores.layer == targets[0][0])
                            & (scores["head"] == targets[0][1])
                        ]
                        if len(ref) != 1 or ref[cols].isna().any().any():
                            frame = frame.iloc[:0]
                            reason = "missing target attention metadata"
                        else:
                            frame = frame.dropna(subset=cols)
                            scale = scores[cols].std().replace(0, 1).fillna(1)
                            frame["match_distance"] = (((frame[cols] - ref[cols].iloc[0]) / scale) ** 2).sum(
                                axis=1
                            )
                            frame = frame.sort_values(["match_distance", "layer", "head"])
                else:
                    # Preserve the existing low-relation control's same-layer preference.
                    same_layer = frame[frame.layer.isin([h[0] for h in targets])]
                    if len(same_layer) >= len(targets):
                        frame = same_layer
                    frame = frame.sort_values(["accuracy", "layer", "head"])
                pool = list(map(tuple, frame[["layer", "head"]].to_numpy()))
        if len(pool) < len(targets):
            audit.append(
                {
                    "relation": relation,
                    "control": kind,
                    "status": "unavailable",
                    "reason": reason or "insufficient distinct non-target heads",
                }
            )
            continue
        if kind == "random":
            indices = rng.choice(len(pool), len(targets), replace=False)
            chosen = tuple(pool[int(i)] for i in indices)
        else:
            chosen = tuple(pool[: len(targets)])
        selected[kind] = chosen
        audit.append(
            {
                "relation": relation,
                "control": kind,
                "status": "available",
                "heads": chosen,
                "associated_relations": [
                    name for name, lock in locks.locks.items() if (lock.layer, lock.head) in chosen
                ],
            }
        )
    return selected, audit


def exogenous_reveal_plan(mask, seed, steps=64):
    """Same 1/remaining hazard as native sampling, in an independent RNG stream."""
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x5DEECE66D)
    current = mask.cpu().clone()
    plan = []
    for step in range(steps):
        reveal = current & (torch.rand(current.shape, generator=generator) < 1 / (steps - step))
        if step == steps - 1:
            reveal = current.clone()
        plan.append(reveal)
        current &= ~reveal
    return torch.stack(plan)


def reconstruct(model, tokenizer, gold, instance, seed, conditions, settings):
    """One equal-length batch of independent conditions for a single example."""
    count = len(conditions)
    initial_mask = torch.ones_like(gold, dtype=torch.bool).cpu()
    initial_mask[:, 0] = False
    single_plan = exogenous_reveal_plan(initial_mask, seed)
    plan = single_plan.expand(-1, count, -1).clone()
    reveal_steps = single_plan[:, 0].long().argmax(dim=0)
    endpoints = sorted(set(instance.attender_span + instance.receiver_span))
    first = int(reveal_steps[endpoints].min())
    records = [
        {
            "trace": [],
            "commit_logprob": [None] * gold.shape[1],
            "commit_logit": [None] * gold.shape[1],
            "active_steps": [],
        }
        for _ in conditions
    ]

    def context(step, mask):
        active = []
        requests = []
        for row, condition in enumerate(conditions):
            if condition is None:
                active.append(False)
                requests.append([])
                continue
            heads, alpha, window = condition[:3]
            start, stop = condition[3:] if len(condition) == 5 else (settings.start, settings.stop)
            gate = temporal_gate(
                mask[row : row + 1],
                [instance.attender_span],
                [instance.receiver_span],
                step=step,
                steps=64,
                window=window,
                start=start,
                stop=stop,
                split=settings.split,
                first_reveal=[first],
            )
            active.append(bool(gate[0]))
            requests.append([HeadScale(int(layer), int(head), alpha) for layer, head in heads])
            if active[-1]:
                records[row]["active_steps"].append(step)
        return scale_projection_heads(model, requests, torch.tensor(active, device=mask.device))

    def observe(step, row, ids, mask, logits, reveal):
        true = gold[0].to(logits.device)
        target_logits = logits.float().gather(-1, true[:, None]).squeeze(-1)
        lp = target_logits - logits.float().logsumexp(dim=-1)
        for pos in reveal.nonzero().flatten().tolist():
            records[row]["commit_logprob"][pos] = float(lp[pos])
            records[row]["commit_logit"][pos] = float(target_logits[pos])
        records[row]["trace"].append(
            {
                "step": step,
                "input_ids": ids.tolist(),
                "masked": mask.tolist(),
                "revealed": reveal.tolist(),
                "gold_logprob": lp.tolist(),
                "gold_logits": target_logits.tolist(),
            }
        )

    trajectories = random_reveal_trajectories(
        model,
        tokenizer,
        [""] * count,
        seed=seed,
        generation_length=gold.shape[1],
        temperature=settings.temperature,
        top_p=settings.top_p,
        reconstruction_ids=gold.expand(count, -1),
        reconstruction_mask=initial_mask.expand(count, -1),
        reveal_plan=plan,
        forward_context=context,
        observer=observe,
    )
    for record, trajectory in zip(records, trajectories, strict=True):
        record["final_ids"] = trajectory.final_ids.tolist()
        record["prediction"] = tokenizer.decode(trajectory.final_ids.tolist())
        record["reveal_steps"] = reveal_steps.tolist()
    return records


def outcome(record, gold, instance):
    predicted = np.asarray(record["final_ids"])
    correct = predicted == np.asarray(gold)
    gov, dep = instance.receiver_span, instance.attender_span
    target = sorted(set(gov + dep))
    non_target = sorted(set(range(1, len(gold))) - set(target))
    lp = np.asarray([np.nan if v is None else v for v in record["commit_logprob"]])
    logits = np.asarray([np.nan if v is None else v for v in record["commit_logit"]])
    return {
        "governor_exact": float(correct[gov].all()),
        "dependent_exact": float(correct[dep].all()),
        "endpoint_exact": float(correct[target].all()),
        "governor_logprob": float(lp[gov].mean()),
        "governor_logit": float(logits[gov].mean()),
        "governor_probability": float(np.exp(lp[gov]).mean()),
        "target_accuracy": float(correct[target].mean()),
        "non_target_accuracy": float(correct[non_target].mean()) if non_target else None,
        "overall_accuracy": float(correct[1:].mean()),
        "sequence_exact": float(correct[1:].all()),
        "relation_preserved": None,
    }


def paired_rows(
    example, anchor, gold, baseline, intervention, *, seed, heads, alpha, window, control, model_id, settings
):
    rows = []
    # All pairs are retained; inference clusters these correlated rows by sentence.
    for evaluated in example.relations:
        b, i = outcome(baseline, gold, evaluated), outcome(intervention, gold, evaluated)
        row = {
            **instance_metadata(example, evaluated, "test"),
            "model": model_id,
            "anchor_instance_id": anchor.instance_id,
            "head_relation": anchor.relation,
            "is_anchor": evaluated.instance_id == anchor.instance_id,
            "primary_anchor": anchor.instance_id
            == next(r.instance_id for r in example.relations if r.relation == anchor.relation),
            "shares_anchor_tokens": bool(
                set(evaluated.attender_span + evaluated.receiver_span)
                & set(anchor.attender_span + anchor.receiver_span)
            ),
            "seed": seed,
            "heads": [list(h) for h in heads],
            "layer": heads[0][0],
            "head": heads[0][1],
            "control": control,
            "alpha": alpha,
            "window": window,
            "start": settings.start,
            "stop": settings.stop,
            "split": settings.split,
            "intervention_type": "knockout" if alpha == 0 else "scale",
            "active_steps": intervention["active_steps"],
            "reveal_steps": intervention["reveal_steps"],
            "baseline_prediction": baseline["prediction"],
            "intervention_prediction": intervention["prediction"],
            "baseline_ids": baseline["final_ids"],
            "intervention_ids": intervention["final_ids"],
            "relation_evaluation_status": "unavailable_no_validated_generated_text_parser",
        }
        for key in b:
            row[f"baseline_{key}"] = b[key]
            row[f"intervention_{key}"] = i[key]
        row["specificity_degradation"] = (
            (
                (b["target_accuracy"] - i["target_accuracy"])
                - (b["non_target_accuracy"] - i["non_target_accuracy"])
            )
            if b["non_target_accuracy"] is not None
            else None
        )
        rows.append(row)
    return rows
