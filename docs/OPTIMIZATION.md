# Compute optimization for the remaining paper experiments

Scope: only the experiments that had not yet been run. Nothing here changes a
scientific definition, a config value, a seed, a lock, or a completed result.

Completed and deliberately untouched: `relation_head_receiver_prediction`, the
six selection locks, `relation_head_receiver_prediction_over_diffusion_time`,
and `attention_entropy`.

## What changed

### POS token-class linear probes: host-transfer batching

`feature_rows` previously moved every feature vector to the host on its own:

```python
"feature": values[span].float().mean(dim=0).cpu().tolist()
```

That `.cpu()` sits inside three nested loops (words x depths x feature kinds).
With three relative depths and a 32-head model that is 99 feature tensors per
word, so a sentence with 20 eligible words forces roughly 2,000 separate device
synchronizations. Each one drains the pipeline.

The reduction is now still computed per span, unchanged, but stays on the device
until every word is done; one transfer then carries all of them per feature
tensor. Synchronizations per sentence fall from `words x depths x (1 + heads)` to
`depths x (1 + heads)` -- about a 20x reduction at 20 eligible words -- while the
arithmetic per span is byte-for-byte the same operation in the same order.

A second, smaller fix: the inner loop scanned every entry of `features` once per
depth and skipped non-matching ones, so it performed `depths^2 x (1 + heads)`
iterations to emit `depths x (1 + heads)` rows. Feature keys are now grouped per
depth once.

**Row order is preserved exactly.** This is not cosmetic: `_evaluate` builds the
shuffled-label control with `rng.shuffle(train_y)` over the stored order, so
emitting rows in a different sequence would silently change a control.

### Matched relation-head ablation: one forward per distinct head

`ablation_chunk` ran a full ablated forward inside the per-instance loop:

```python
for instance in example.relations:
    for control_kind, layer, head in interventions:
        with capture_or_ablate_projection(model, layer, ablate_head=head):
            ablated_logits, _ = model.forward_attentions(state.input_ids)
```

The ablated logits depend only on `state.input_ids` and the zeroed head slice --
never on which instance requested them. Every POS-ranked pair is identical for
all instances, so each was recomputed once per instance.

Interventions are now collected first, each distinct `(layer, head)` is forwarded
once per sentence and state, and the logits are reused. In the test fixture this
takes 21 forwards down to 8.

**Distinct `control_kind` labels that resolve to the same head still emit their
own rows.** Only the forward is shared, never the row identity; a test pins this.

## Compatibility with the existing 5-hour DiffuLLaMA POS run

Checkpoint identity is defined in `SentenceCheckpointStore._expected_metadata`:
schema version, stage, seed, normalized progress, timestep, heads, chunk bounds,
sentence-id hash, scientific config hash, and manifest hashes -- plus a row count
and a parquet SHA-256 recorded at write time.

**No implementation or code hash participates.** Changing how a feature vector
reaches the host therefore cannot invalidate a stored chunk. The stage string,
chunk size, filename format, and DataFrame schema are all unchanged, and the
feature values are proven identical by test.

Existing files such as

```text
paper-pos-selection-features__seed-42__p-0.500000__t-32__heads-all__sentences-003300-003313.parquet
paper-pos-selection-features__seed-42__p-0.750000__t-47__heads-all__sentences-001800-002100.parquet
```

remain valid and are skipped on resume. No migration is required and nothing
needs to be deleted.

Verify before launching a long job:

```bash
python -m pytest tests/test_paper_optimizations.py -q -k checkpoint
```

## Audited, left unchanged

- **Direct logit attribution** — already captures projection inputs for all
  needed layers in one forward per sentence and state. No redundant expensive
  computation found. No change.
- **Multilingual relation-head transfer** — already uses
  `teacher_forced_trajectory` with `attention_batches_for_states` and frozen
  heads. Audited; no justified code change.
- **Attention heatmaps and trajectories** — qualitative case selection and
  plotting are unchanged by design; batching infrastructure already exists.

## Identified but NOT implemented

**Shared native trajectory cache.** `final_token_prediction_by_layer` and
`prediction_before_unmasking_timing_analysis` use identical generation settings:
the same prompt manifest, seeds `[42, 43, 44]`, 64 steps, generation length 96,
temperature 0.95, top-p 0.9, and the `random_one_over_remaining_steps` reveal
policy. But `generate_trajectories` keys its checkpoint stage on
`cfg.experiment.id`:

```python
stage=f"{cfg.experiment.id}-native-trajectories-{prompt_hash[:12]}"
```

so the two experiments write to different stages inside different run
directories and **generate the same trajectories twice**. Timing analysis needs
no model at all — `timing_rows` takes only a tokenizer and a stored row — so the
second generation is pure waste.

This was not implemented here because a cross-run cache needs a scientific
identity covering model/tokenizer/remote-code revisions, prompt manifest hash,
seed, steps, generation length, temperature, top-p, and reveal policy, with
fail-closed validation. That is a larger and more sensitive change than the two
above, and it cannot be verified without real trajectories. Recommended as the
next patch; expected saving is one full native generation pass.

## Recommended run order

1. POS feature extraction (GPU) — resumes on existing checkpoints
2. POS probe fitting (CPU)
3. Validate POS
4. Native trajectory generation
5. Final-token prediction by layer
6. Prediction-before-unmasking timing analysis
7. Direct logit attribution
8. Matched relation-head ablation — consumes POS rankings
9. Attention heatmaps and trajectories
10. German locked transfer
11. Japanese locked transfer

Step 8 reads POS head rankings through `DLMREL_POS_HEAD_RANKINGS`, pointed at the
completed POS run directory. If POS has not completed, `_pos_control_pairs`
returns an empty list and the POS comparison silently disappears from the
ablation rather than failing, so confirm POS finished before starting step 8.
