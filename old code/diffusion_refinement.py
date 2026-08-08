#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze how Dream/DiffuGPT diffusion histories refine token predictions.

This script addresses two related order-analysis questions:

1) Diffusion refinement curves:
   For each generated token position and each diffusion history step, run the
   model on that step sequence, take the final-layer token prediction, and
   measure whether the argmax already equals the final generated token.

2) Prediction-before-unmask deltas:
   For each generated token position, find the first step where the final-layer
   argmax equals the final generated token (found_time) and the first step where
   the masked position becomes unmasked as that final token (unmask_time). The
   signed delta is found_time - unmask_time; negative values mean the model
   "knew" the final token before it was actually unmasked.

Outputs are written to:
  <output_dir>/<exp_name>/
    refinement_by_step.csv
    token_deltas.csv
    prediction_texts.jsonl
    large_leads.json
    summaries.json
    plots/refinement_curve.png|pdf
    plots/delta_histogram.png|pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

import dream_infer


DEFAULT_MODEL_NAME = "Dream-org/Dream-v0-Base-7B"

DEFAULT_PROMPT = """You are a helpful math tutor. Solve the problem step by step and give the final answer as an integer on the last line.
Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Reasoning: Let's think step by step.
Answer:"""


@dataclass
class PromptRecord:
    prompt_id: str
    task: str
    prompt: str


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_str_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_json_dict(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("--extra_generate_kwargs must parse to a JSON object.")
    return obj


def _prompt_from_mapping(row: Dict[str, Any], idx: int) -> PromptRecord:
    prompt = row.get("prompt", row.get("text", row.get("input", None)))
    if prompt is None:
        raise ValueError(
            "Prompt records must contain one of: prompt, text, input. "
            f"Bad record index={idx}: {row}"
        )
    prompt_id = str(row.get("id", row.get("prompt_id", idx)))
    task = str(row.get("task", row.get("category", row.get("label", "default"))))
    return PromptRecord(prompt_id=prompt_id, task=task, prompt=str(prompt))


def load_prompt_records(args: argparse.Namespace) -> List[PromptRecord]:
    if args.prompt_records:
        path = Path(args.prompt_records)
        suffix = path.suffix.lower()
        records: List[PromptRecord] = []

        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    records.append(_prompt_from_mapping(json.loads(line), idx))
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "records" in obj:
                obj = obj["records"]
            if not isinstance(obj, list):
                raise ValueError("JSON prompt records must be a list or {'records': [...]} object.")
            for idx, row in enumerate(obj):
                if not isinstance(row, dict):
                    raise ValueError(f"JSON prompt record {idx} is not an object.")
                records.append(_prompt_from_mapping(row, idx))
        elif suffix == ".csv":
            df = pd.read_csv(path)
            for idx, row in df.iterrows():
                records.append(_prompt_from_mapping(row.to_dict(), int(idx)))
        else:
            raise ValueError("--prompt_records must be .jsonl, .json, or .csv")

        if args.tasks:
            keep = set(parse_str_list(args.tasks))
            records = [r for r in records if r.task in keep]
        if args.max_prompts > 0:
            records = records[: int(args.max_prompts)]
        if not records:
            raise ValueError("No prompt records selected.")
        return records

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT
    return [PromptRecord(prompt_id="0", task=args.task, prompt=prompt)]


@torch.inference_mode()
def predict_argmax_by_step(
    model,
    step_seqs: Sequence[Sequence[int]],
    gen_start: int,
    gen_end_abs: int,
    batch_steps: int,
) -> List[List[int]]:
    """
    Return per-step argmax token ids over generated positions.

    Shape as Python lists: predictions[t][pos_rel_gen].
    """
    device = next(model.parameters()).device
    predictions: List[List[int]] = []

    for start in range(0, len(step_seqs), int(batch_steps)):
        chunk = step_seqs[start: start + int(batch_steps)]
        max_len = max(len(ids) for ids in chunk)
        batch_ids = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(batch_ids, dtype=torch.bool, device=device)

        for i, ids in enumerate(chunk):
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
            batch_ids[i, : len(ids)] = ids_tensor
            # False means keep, True means pad/masked out.
            attention_mask[i, : len(ids)] = False

        out = model(input_ids=batch_ids, attention_mask=attention_mask, return_dict=True)
        logits = out.logits[:, int(gen_start): int(gen_end_abs), :]
        pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        predictions.extend([[int(x) for x in row] for row in pred])

    return predictions


def first_unmask_time(
    *,
    step_seqs: Sequence[Sequence[int]],
    pos_abs: int,
    mask_id: Optional[int],
    final_token_id: int,
) -> Optional[int]:
    for t, ids in enumerate(step_seqs):
        if pos_abs >= len(ids):
            continue
        tok = int(ids[pos_abs])
        if mask_id is None:
            if tok == int(final_token_id):
                return int(t)
        elif tok != int(mask_id):
            if tok == int(final_token_id):
                return int(t)
            # The issue defines unmask_time as unmasked as the final token. If a
            # different token appears first, keep searching for the final token.
    return None


def first_found_time(
    predictions: Sequence[Sequence[int]],
    pos_rel: int,
    final_token_id: int,
) -> Optional[int]:
    for t, row in enumerate(predictions):
        if pos_rel < len(row) and int(row[pos_rel]) == int(final_token_id):
            return int(t)
    return None


def token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return text.replace("\n", "\\n").replace("\t", "\\t")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def analyze_prompt_alg(
    *,
    model,
    tokenizer,
    record: PromptRecord,
    alg: str,
    args: argparse.Namespace,
    extra_generate_kwargs: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    hist = dream_infer.run_dream_history(
        model=model,
        tokenizer=tokenizer,
        prompt=record.prompt,
        alg=alg,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        alg_temp=args.alg_temp,
        do_sample=args.do_sample,
        seed=args.seed,
        strict_determinism=args.strict_determinism,
        use_chat_template=args.use_chat_template,
        mask_token_str=args.mask_token_str,
        extra_generate_kwargs=extra_generate_kwargs,
    )

    max_steps = min(
        len(hist.step_seqs),
        int(args.max_steps_analyze) if args.max_steps_analyze > 0 else len(hist.step_seqs),
    )
    step_seqs = hist.step_seqs[:max_steps]

    predictions = predict_argmax_by_step(
        model=model,
        step_seqs=step_seqs,
        gen_start=hist.input_len,
        gen_end_abs=hist.gen_end_abs,
        batch_steps=args.forward_batch_steps,
    )

    final_ids = [int(x) for x in hist.final_ids]
    gen_positions = list(range(int(hist.input_len), int(hist.gen_end_abs)))
    gen_len = len(gen_positions)

    refinement_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    for t, pred_row in enumerate(predictions):
        correct = 0
        masked = 0
        for pos_rel, pos_abs in enumerate(gen_positions):
            final_tok = int(final_ids[pos_abs])
            if pos_rel < len(pred_row) and int(pred_row[pos_rel]) == final_tok:
                correct += 1
            if hist.mask_id is not None and pos_abs < len(step_seqs[t]):
                masked += int(int(step_seqs[t][pos_abs]) == int(hist.mask_id))

        refinement_rows.append(
            {
                "prompt_id": record.prompt_id,
                "task": record.task,
                "alg": alg,
                "step": int(t),
                "progress": float(t / max(max_steps - 1, 1)),
                "num_generated_tokens": int(gen_len),
                "num_correct_predictions": int(correct),
                "correct_fraction": float(correct / gen_len) if gen_len else 0.0,
                "num_masked_tokens": int(masked),
                "masked_fraction": float(masked / gen_len) if gen_len else 0.0,
            }
        )

        current_gen_ids = [
            int(step_seqs[t][pos_abs])
            for pos_abs in gen_positions
            if pos_abs < len(step_seqs[t])
        ]
        prediction_rows.append(
            {
                "prompt_id": record.prompt_id,
                "task": record.task,
                "alg": alg,
                "step": int(t),
                "progress": float(t / max(max_steps - 1, 1)),
                "correct_fraction": float(correct / gen_len) if gen_len else 0.0,
                "current_generated_text": tokenizer.decode(
                    current_gen_ids, skip_special_tokens=False
                ),
                "argmax_generated_text": tokenizer.decode(
                    [int(x) for x in pred_row], skip_special_tokens=False
                ),
            }
        )

    token_rows: List[Dict[str, Any]] = []
    large_leads: List[Dict[str, Any]] = []
    for pos_rel, pos_abs in enumerate(gen_positions):
        final_tok = int(final_ids[pos_abs])
        found = first_found_time(predictions, pos_rel=pos_rel, final_token_id=final_tok)
        unmask = first_unmask_time(
            step_seqs=step_seqs,
            pos_abs=pos_abs,
            mask_id=hist.mask_id,
            final_token_id=final_tok,
        )
        signed_delta = None if found is None or unmask is None else int(found - unmask)
        lead_steps = None if signed_delta is None else int(-signed_delta)
        final_piece = token_text(tokenizer, final_tok)

        row = {
            "prompt_id": record.prompt_id,
            "task": record.task,
            "alg": alg,
            "pos_abs": int(pos_abs),
            "pos_rel_gen": int(pos_rel),
            "final_token_id": int(final_tok),
            "final_token_str": final_piece,
            "found_time": -1 if found is None else int(found),
            "unmask_time": -1 if unmask is None else int(unmask),
            "delta_found_minus_unmask": "" if signed_delta is None else int(signed_delta),
            "lead_steps_unmask_minus_found": "" if lead_steps is None else int(lead_steps),
        }
        token_rows.append(row)

        if lead_steps is not None and lead_steps >= int(args.large_lead_threshold):
            large_leads.append(
                {
                    **row,
                    "prompt_preview": record.prompt[:500],
                    "final_text": hist.final_text,
                }
            )

    summary = {
        "prompt_id": record.prompt_id,
        "task": record.task,
        "alg": alg,
        "seed": int(args.seed),
        "input_len": int(hist.input_len),
        "gen_end_abs": int(hist.gen_end_abs),
        "num_generated_tokens": int(gen_len),
        "num_steps_in_history": int(len(hist.step_seqs)),
        "num_steps_analyzed": int(max_steps),
        "mask_id": hist.mask_id,
        "final_sha256": hist.final_sha256,
        "final_text": hist.final_text,
    }
    return refinement_rows, token_rows, prediction_rows, summary, large_leads


def plot_refinement_curve(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    grouped = (
        df.groupby(["task", "alg", "step"], as_index=False)
        .agg(correct_fraction=("correct_fraction", "mean"), progress=("progress", "mean"))
    )

    plt.figure(figsize=(10, 5))
    for (task, alg), sub in grouped.groupby(["task", "alg"]):
        sub = sub.sort_values("progress")
        plt.plot(sub["progress"], sub["correct_fraction"], linewidth=2.2, label=f"{task} / {alg}")
    plt.xlabel("Diffusion Progress")
    plt.ylabel("Correct Argmax Fraction")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"refinement_curve.{ext}", dpi=300)
    plt.close()


def plot_delta_histogram(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)
    work = df.copy()
    work["delta_found_minus_unmask"] = pd.to_numeric(
        work["delta_found_minus_unmask"], errors="coerce"
    )
    work = work.dropna(subset=["delta_found_minus_unmask"])
    if work.empty:
        return

    vmin = int(work["delta_found_minus_unmask"].min())
    vmax = int(work["delta_found_minus_unmask"].max())
    bins = range(vmin, vmax + 2)

    plt.figure(figsize=(10, 5))
    for (task, alg), sub in work.groupby(["task", "alg"]):
        plt.hist(
            sub["delta_found_minus_unmask"],
            bins=bins,
            alpha=0.45,
            label=f"{task} / {alg}",
        )
    plt.axvline(0, color="black", linewidth=1.2)
    plt.xlabel("found_time - unmask_time")
    plt.ylabel("Token Count")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(out_dir / f"delta_histogram.{ext}", dpi=300)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true", default=True)

    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument(
        "--prompt_records",
        type=str,
        default=None,
        help="Optional .jsonl/.json/.csv with prompt/text/input plus optional id/task fields.",
    )
    ap.add_argument("--task", type=str, default="default")
    ap.add_argument("--tasks", type=str, default="", help="Comma-separated task labels to keep.")
    ap.add_argument("--max_prompts", type=int, default=0, help="0 means use all selected prompts.")
    ap.add_argument("--use_chat_template", action="store_true", default=True)

    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--alg_temp", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true", default=True)
    ap.add_argument("--algs", type=str, default="origin,entropy")
    ap.add_argument("--extra_generate_kwargs", type=str, default="")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict_determinism", action="store_true", default=True)
    ap.add_argument("--mask_token_str", type=str, default=None)

    ap.add_argument("--forward_batch_steps", type=int, default=4)
    ap.add_argument("--max_steps_analyze", type=int, default=0)
    ap.add_argument("--large_lead_threshold", type=int, default=16)

    ap.add_argument("--output_dir", type=str, default="./outputs/diffusion_refinement")
    ap.add_argument("--exp_name", type=str, default=None)

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    prompts = load_prompt_records(args)
    algs = parse_str_list(args.algs)
    if not algs:
        raise ValueError("--algs must contain at least one algorithm.")
    extra_generate_kwargs = parse_json_dict(args.extra_generate_kwargs)

    exp_name = args.exp_name or time.strftime("diffusion_refinement_%Y%m%d_%H%M%S")
    exp_dir = Path(args.output_dir) / exp_name
    ensure_dir(exp_dir)

    write_json(
        exp_dir / "config.json",
        {
            "model_name": args.model_name,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "algs": algs,
            "num_prompts": len(prompts),
            "extra_generate_kwargs": extra_generate_kwargs,
            "args": vars(args),
        },
    )

    model, tokenizer = dream_infer.build_model_and_tokenizer(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    all_refinement_rows: List[Dict[str, Any]] = []
    all_token_rows: List[Dict[str, Any]] = []
    all_prediction_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    large_leads: List[Dict[str, Any]] = []

    jobs = [(record, alg) for record in prompts for alg in algs]
    for record, alg in tqdm(jobs, desc="[Analyze]", unit="prompt/alg", dynamic_ncols=True):
        refinement_rows, token_rows, prediction_rows, summary, leads = analyze_prompt_alg(
            model=model,
            tokenizer=tokenizer,
            record=record,
            alg=alg,
            args=args,
            extra_generate_kwargs=extra_generate_kwargs,
        )
        all_refinement_rows.extend(refinement_rows)
        all_token_rows.extend(token_rows)
        all_prediction_rows.extend(prediction_rows)
        summaries.append(summary)
        large_leads.extend(leads)

    refinement_fields = [
        "prompt_id", "task", "alg", "step", "progress", "num_generated_tokens",
        "num_correct_predictions", "correct_fraction", "num_masked_tokens", "masked_fraction",
    ]
    token_fields = [
        "prompt_id", "task", "alg", "pos_abs", "pos_rel_gen", "final_token_id",
        "final_token_str", "found_time", "unmask_time", "delta_found_minus_unmask",
        "lead_steps_unmask_minus_found",
    ]

    refinement_path = exp_dir / "refinement_by_step.csv"
    token_path = exp_dir / "token_deltas.csv"
    write_csv(refinement_path, all_refinement_rows, refinement_fields)
    write_csv(token_path, all_token_rows, token_fields)
    write_jsonl(exp_dir / "prediction_texts.jsonl", all_prediction_rows)
    write_json(exp_dir / "summaries.json", summaries)
    write_json(
        exp_dir / "large_leads.json",
        sorted(
            large_leads,
            key=lambda row: int(row.get("lead_steps_unmask_minus_found") or 0),
            reverse=True,
        ),
    )

    plot_dir = exp_dir / "plots"
    plot_refinement_curve(pd.DataFrame(all_refinement_rows), plot_dir)
    plot_delta_histogram(pd.DataFrame(all_token_rows), plot_dir)

    print(f"[DONE] Outputs written to: {exp_dir}")


if __name__ == "__main__":
    main()
