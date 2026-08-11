"""Dream-7B smoke test, ported to Modal from notebooks/smoke_test_dream7b.ipynb.

Same logic, same tests, same comments/rationale as the notebook — just running
on a Modal A100 instead of Colab. One advantage over Colab here: Modal's image
isolation means there's no "fresh runtime" dance to avoid clashing with
DiffuLLaMA's transformers==4.44.2 pin — this container only ever has 4.51.3.

Usage:
    modal run scripts/modal_dream_smoke_test.py

First run downloads ~14GB of bf16 weights into a persistent Modal Volume
(`dream7b-hf-cache`), so subsequent runs skip the download.

If `Dream-org/Dream-v0-Base-7B` turns out to be gated on HuggingFace, you'll
need a token available to the container:
    modal secret create huggingface-secret HF_TOKEN=<your token>
and uncomment the `secrets=[...]` line below.
"""

import modal

app = modal.App("dream7b-smoke-test")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.51.3",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "numpy",
)

hf_cache = modal.Volume.from_name("dream7b-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": hf_cache},
    timeout=30 * 60,
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_dream_smoke_test() -> None:
    import os
    import time

    os.environ["HF_HOME"] = "/cache/huggingface"

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # ---- 1 - GPU ----------------------------------------------------------
    assert torch.cuda.is_available(), "No GPU on this container"
    p = torch.cuda.get_device_properties(0)
    print(f"\n{p.name} | {p.total_memory / 1e9:.0f} GB")

    import transformers

    print("transformers", transformers.__version__)
    assert transformers.__version__ == "4.51.3"

    # ---- 3 - Load Dream-7B --------------------------------------------------
    MODEL = "Dream-org/Dream-v0-Base-7B"

    if "default" not in ROPE_INIT_FUNCTIONS and "rope" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["rope"]
        print("applied RoPE compatibility patch")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    t0 = time.time()
    try:
        model = AutoModel.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="eager",
        ).eval()
        print("loaded with attn_implementation=eager")
    except (TypeError, ValueError) as exc:
        print(f"eager kwarg rejected ({exc}); falling back to remote-code default")
        model = AutoModel.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        ).eval()

    cfg = model.config
    N_LAYERS = cfg.num_hidden_layers
    N_HEADS = cfg.num_attention_heads
    print(f"\nloaded in {time.time() - t0:.0f}s")
    print(f"{N_LAYERS} layers x {N_HEADS} heads = {N_LAYERS * N_HEADS} heads")
    print(
        f'hidden {cfg.hidden_size} | kv heads {getattr(cfg, "num_key_value_heads", "n/a")} (GQA)'
    )
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    hf_cache.commit()  # persist the downloaded weights for next run

    # ---- 4 - Special tokens -------------------------------------------------
    MASK_ID = tokenizer.mask_token_id
    if MASK_ID is None:
        for cand in ("<|mask|>", "<mask>", "[MASK]"):
            tid = tokenizer.convert_tokens_to_ids(cand)
            if tid is not None and tid != tokenizer.unk_token_id:
                MASK_ID = tid
                print(f"mask_token_id was None; resolved {cand!r} -> {tid}")
                break
    assert MASK_ID is not None, "no mask token — the diffusion schedule needs one"

    if not tokenizer.is_fast:
        try:
            _fast = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
            if _fast.is_fast:
                tokenizer = _fast
                MASK_ID = tokenizer.mask_token_id or MASK_ID
                print("reloaded a fast tokenizer")
        except Exception as exc:  # noqa: BLE001 - capability probe; any failure means "no fast tokenizer"
            print(f"no fast tokenizer available ({type(exc).__name__}); using manual offsets")

    try:
        tokenizer("probe", return_offsets_mapping=True)
        FAST_OFFSETS = True
    except Exception:  # noqa: BLE001 - slow tokenizers raise several types here
        FAST_OFFSETS = False
    print(f"tokenizer is_fast={tokenizer.is_fast} | offset_mapping available={FAST_OFFSETS}")

    print(f"mask  {MASK_ID:>7}  {tokenizer.decode([MASK_ID])!r}")
    print(f"bos   {tokenizer.bos_token_id!s:>7}  {tokenizer.bos_token!r}")
    print(f"eos   {tokenizer.eos_token_id!s:>7}  {tokenizer.eos_token!r}")
    print(f"pad   {tokenizer.pad_token_id!s:>7}  {tokenizer.pad_token!r}")

    probe = tokenizer("The chef prepared the meal.", add_special_tokens=True)["input_ids"]
    PREPENDS_BOS = tokenizer.bos_token_id is not None and probe[0] == tokenizer.bos_token_id
    print(f"\nadd_special_tokens prepends BOS: {PREPENDS_BOS}")
    print(f"tokens: {[tokenizer.decode([i]) for i in probe]}")

    # ---- 5 - Helpers ----------------------------------------------------------
    @torch.no_grad()
    def forward_with_attentions(input_ids):
        out = model(
            input_ids=input_ids, attention_mask=None, output_attentions=True, return_dict=True
        )
        if getattr(out, "attentions", None) is None:
            raise RuntimeError(
                "no attentions returned — remote code is ignoring output_attentions, "
                "likely running sdpa/flash. Every accuracy would be zero."
            )
        return out.attentions

    def manual_offsets(ids, text):
        offs, cursor = [], 0
        for tid in ids:
            piece = tokenizer.decode([tid]).strip()
            if not piece:
                offs.append((cursor, cursor))
                continue
            idx = text.find(piece, cursor)
            if idx < 0:
                offs.append((cursor, cursor))
                continue
            offs.append((idx, idx + len(piece)))
            cursor = idx + len(piece)
        return offs

    def encode(text):
        enc = tokenizer(text, add_special_tokens=True)
        ids = enc["input_ids"]
        if FAST_OFFSETS:
            offs = list(
                tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)[
                    "offset_mapping"
                ]
            )
        else:
            offs = manual_offsets(ids, text)
        return torch.tensor([ids], device=model.device), offs

    def span_for_word(text, offsets, word):
        start = text.find(word)
        assert start >= 0, f"{word!r} not in {text!r}"
        end = start + len(word)
        span = [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]
        assert span, f"no tokens aligned to {word!r}"
        return span

    def receiver_predictions(attentions, layer, attender_span, exclude_pos0=True):
        row = attentions[layer][0, :, attender_span[-1], :].detach().float().clone()
        if exclude_pos0:
            row[:, 0] = float("-inf")
        for c in attender_span:
            row[:, c] = float("-inf")
        return row.argmax(dim=1).cpu().numpy()

    def teacher_forced_state(true_ids, diffusion_time, steps=64, seed=42, protect=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        maskable = torch.ones_like(true_ids, dtype=torch.bool)
        if protect:
            maskable[:, :protect] = False
        xt = true_ids.masked_fill(maskable, MASK_ID)
        remaining = maskable.clone()
        for progress in range(diffusion_time):
            p_reveal = 1.0 / (steps - progress)
            reveal = remaining & (torch.rand_like(remaining, dtype=torch.float) < p_reveal)
            xt = xt.clone()
            xt[reveal] = true_ids[reveal]
            remaining &= ~reveal
        if diffusion_time == steps - 1:
            xt, remaining = true_ids.clone(), torch.zeros_like(remaining)
        return xt, (~remaining[0]).cpu().tolist()

    PROTECT = 1 if PREPENDS_BOS else 0
    print(f"helpers defined | protecting {PROTECT} leading position(s) from masking")

    _probe = "The chef prepared the meal carefully."
    _ids, _offs = encode(_probe)
    print(f'\n{"word":>10}  tokens  decoded')
    for _w in ["The", "chef", "prepared", "meal", "carefully"]:
        _sp = span_for_word(_probe, _offs, _w)
        _dec = "".join(tokenizer.decode([_ids[0, i].item()]) for i in _sp).strip()
        assert _w in _dec or _dec in _w, f"MISALIGNED {_w!r} -> {_dec!r}"
        print(f"{_w:>10}  {_sp!s:>7}  {_dec!r}")
    print("\nalignment OK")

    # ---- 6 - Test 1: are attentions returned? --------------------------------
    ids, offs = encode("The chef prepared the meal carefully.")
    attentions = forward_with_attentions(ids)

    assert len(attentions) == N_LAYERS, f"{len(attentions)} layers, expected {N_LAYERS}"
    _, h, q, k = attentions[0].shape
    assert (h, q, k) == (N_HEADS, ids.shape[1], ids.shape[1]), attentions[0].shape

    rows = attentions[0][0].float().sum(-1)
    print(f"\nPASS  {len(attentions)} layers, each {tuple(attentions[0].shape)}")
    print(f"      rows sum to {rows.mean():.4f} (should be ~1.0)")
    print(f"      tokens: {[tokenizer.decode([i]) for i in ids[0].tolist()]}")

    # ---- 7 - Test 2: is there a position-0 attention sink? -------------------
    sink_mass, sink_argmax = [], []
    for layer in range(N_LAYERS):
        a = attentions[layer][0].float()
        sink_mass.append(a[:, :, 0].mean().item())
        sink_argmax.append((a.argmax(-1) == 0).float().mean().item())

    mean_mass = float(np.mean(sink_mass))
    print(f"\nmean attention mass on position 0 : {mean_mass:.1%}")
    print(f"rows whose argmax IS position 0   : {np.mean(sink_argmax):.1%}\n")
    for layer in range(0, N_LAYERS, max(1, N_LAYERS // 8)):
        print(f'  L{layer:02d} {sink_mass[layer]:6.1%} {"#" * int(sink_mass[layer] * 50)}')

    EXCLUDE_POS0 = mean_mass > 0.10
    print(f'\n-> position 0 {"IS" if EXCLUDE_POS0 else "is NOT"} behaving as a sink')
    print(f"-> EXCLUDE_POS0 = {EXCLUDE_POS0} for the head tests below")

    # ---- 8 - Test 3: object -> verb across all heads -------------------------
    CASES = [
        ("The chef prepared the meal carefully.", "meal", "prepared"),
        ("She wrote a long letter yesterday.", "letter", "wrote"),
        ("They finally opened the heavy door.", "door", "opened"),
        ("The student answered the difficult question.", "question", "answered"),
        ("He quietly closed the wooden window.", "window", "closed"),
        ("The gardener planted several young trees.", "trees", "planted"),
    ]

    def score_relation(rows):
        hits = np.zeros((N_LAYERS, N_HEADS))
        for text, attender, receiver in rows:
            ids, offs = encode(text)
            att = forward_with_attentions(ids)
            a_span = span_for_word(text, offs, attender)
            r_span = set(span_for_word(text, offs, receiver))
            for layer in range(N_LAYERS):
                preds = receiver_predictions(att, layer, a_span, EXCLUDE_POS0)
                hits[layer] += np.isin(preds, list(r_span))
        return hits / len(rows)

    obj_acc = score_relation(CASES)
    print(f"\ntop heads, object -> verb  (n={len(CASES)})\n")
    for flat in np.argsort(obj_acc, axis=None)[::-1][:10]:
        layer, head = np.unravel_index(flat, obj_acc.shape)
        print(f"  L{layer:02d} H{head:02d}  {obj_acc[layer, head]:.2f}")
    print(f"\nmean over all {N_LAYERS * N_HEADS} heads: {obj_acc.mean():.3f}")

    # ---- 9 - Test 4: positional vs relational profile ------------------------
    DET_CASES = [
        ("The chef prepared the meal carefully.", "The", "chef"),
        ("They finally opened the heavy door.", "the", "door"),
        ("The student answered the difficult question.", "The", "student"),
        ("He quietly closed the wooden window.", "the", "window"),
    ]
    det_acc = score_relation(DET_CASES)

    def offset_baseline(rows, k):
        hit = 0
        for text, attender, receiver in rows:
            _, offs = encode(text)
            a = span_for_word(text, offs, attender)[-1]
            hit += (a + k) in set(span_for_word(text, offs, receiver))
        return hit / len(rows)

    print("\nfixed-offset null (k=+1), the thing a head must beat:")
    print(f"  object -> verb  {offset_baseline(CASES, 1):.2f}")
    print(f"  det    -> noun  {offset_baseline(DET_CASES, 1):.2f}\n")

    cands = np.argsort(obj_acc, axis=None)[::-1][:5]
    print(f'{"head":10s} {"obj->verb":>10s} {"det->noun":>10s}   reading')
    for flat in cands:
        layer, head = np.unravel_index(flat, obj_acc.shape)
        o, d = obj_acc[layer, head], det_acc[layer, head]
        reading = "positional?" if d >= 0.75 else ("relational?" if d <= 0.25 else "mixed")
        print(f"L{layer:02d} H{head:02d}   {o:>10.2f} {d:>10.2f}   {reading}")

    flat = int(np.argmax(det_acc))
    layer, head = np.unravel_index(flat, det_acc.shape)
    print(
        f"\nbest det->noun head: L{layer:02d} H{head:02d} = {det_acc[layer, head]:.2f} "
        f"(obj->verb {obj_acc[layer, head]:.2f})"
    )

    # ---- 10 - Test 5: the denoising schedule ----------------------------------
    STEPS = 64
    text = "The chef prepared the meal carefully."
    true_ids, offs = encode(text)

    print(f'\n{"t":>4} {"visible":>9}   state')
    for t in [0, 8, 16, 32, 48, 63]:
        xt, visible = teacher_forced_state(true_ids, t, STEPS, protect=PROTECT)
        shown = " ".join(
            tokenizer.decode([i]).strip() if v else "_" for i, v in zip(xt[0].tolist(), visible)
        )
        print(f"{t:>4} {sum(visible):>3}/{len(visible):<5}   {shown}")

    _, v0 = teacher_forced_state(true_ids, 0, STEPS, protect=PROTECT)
    xf, vf = teacher_forced_state(true_ids, STEPS - 1, STEPS, protect=PROTECT)
    assert sum(v0) == PROTECT, f"expected {PROTECT} visible at t=0, got {sum(v0)}"
    assert all(vf) and torch.equal(xf, true_ids), "final frame must be the true sentence"
    print(f"\nPASS  t=0 masked down to {PROTECT} protected position(s), t=63 exact")

    # ---- 11 - Test 6: a head across diffusion time ----------------------------
    flat = int(np.argmax(obj_acc))
    LAYER, HEAD = (int(x) for x in np.unravel_index(flat, obj_acc.shape))
    text, obj, verb = CASES[0]
    true_ids, offs = encode(text)
    a_span = span_for_word(text, offs, obj)
    r_span = set(span_for_word(text, offs, verb))

    print(f'\nL{LAYER} H{HEAD} (best from test 3) | "{obj}" -> "{verb}"')
    print(f"{text}\n")
    print(f'{"t":>4} {"n_masked":>9} {"endpoints":>11} {"correct":>8}   predicted')
    for t in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
        xt, visible = teacher_forced_state(true_ids, t, STEPS, protect=PROTECT)
        att = forward_with_attentions(xt)
        pred = int(receiver_predictions(att, LAYER, a_span, EXCLUDE_POS0)[HEAD])
        both_masked = not any(visible[p] for p in list(a_span) + list(r_span))
        print(
            f"{t:>4} {len(visible) - sum(visible):>9} "
            f'{"masked" if both_masked else "visible":>11} '
            f'{pred in r_span!s:>8}   {tokenizer.decode([xt[0, pred].item()])!r}'
        )

    # ---- 12 - Test 7: native generation ---------------------------------------
    if not hasattr(model, "diffusion_generate"):
        print("\nno diffusion_generate on this model revision — skipping")
    else:
        prompt = "The capital city of France is"
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        t0 = time.time()
        out = model.diffusion_generate(
            enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=32,
            steps=32,
            temperature=0.0,
            top_p=1.0,
            alg="origin",
            alg_temp=0.0,
            output_history=False,
            return_dict_in_generate=True,
        )
        seq = out.sequences if hasattr(out, "sequences") else out
        print(f"\ngenerated in {time.time() - t0:.1f}s\n")
        print(repr(tokenizer.decode(seq[0].tolist())))

    print(
        "\n---\nSummary: passing tests 1, 2, 5, 7 means the stack is sound. "
        "Nothing here is a scored result — see the notebook's closing note."
    )


@app.local_entrypoint()
def main():
    run_dream_smoke_test.remote()
