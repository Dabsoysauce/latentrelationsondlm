import time

import modal

app = modal.App("dream7b-loading-check")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers==4.51.3",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "numpy",
)

hf_cache = modal.Volume.from_name("dream7b-hf-cache", create_if_missing=True)

MODEL = "Dream-org/Dream-v0-Base-7B"


@app.function(image=image, gpu="A100", volumes={"/cache": hf_cache}, timeout=30 * 60)
def check() -> None:
    import os

    os.environ["HF_HOME"] = "/cache/huggingface"

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | {p.total_memory / 1e9:.0f} GB")

    if "default" not in ROPE_INIT_FUNCTIONS and "rope" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["rope"]
        print("applied RoPE compatibility patch")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    print(f"loaded in {time.time() - t0:.0f}s")
    print(f"{n_layers} layers x {n_heads} heads = {n_layers * n_heads} heads")
    print(f"hidden {cfg.hidden_size} | kv heads {cfg.num_key_value_heads}")
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    hf_cache.commit()

    mask_id = tokenizer.mask_token_id
    assert mask_id is not None, "no mask token, the diffusion schedule needs one"

    probe = tokenizer("The chef prepared the meal.", add_special_tokens=True)
    prepends_bos = (
        tokenizer.bos_token_id is not None
        and probe["input_ids"][0] == tokenizer.bos_token_id
    )
    print(f"mask {mask_id} {tokenizer.decode([mask_id])!r}")
    print(f"bos  {tokenizer.bos_token_id} {tokenizer.bos_token!r}")
    print(f"tokenizer prepends bos: {prepends_bos}")

    text = "The chef prepared the meal carefully."
    ids = torch.tensor(
        [tokenizer(text, add_special_tokens=True)["input_ids"]], device=model.device
    )

    with torch.no_grad():
        out = model(
            input_ids=ids,
            attention_mask=None,
            output_attentions=True,
            return_dict=True,
        )
    attentions = out.attentions

    assert attentions is not None, (
        "no attentions returned, the remote code is running sdpa and every "
        "accuracy downstream would be zero"
    )
    assert len(attentions) == n_layers
    _, h, q, k = attentions[0].shape
    assert (h, q, k) == (n_heads, ids.shape[1], ids.shape[1])
    rows = attentions[0][0].float().sum(-1)
    print(f"\nattentions: {len(attentions)} layers, each {tuple(attentions[0].shape)}")
    print(f"rows sum to {rows.mean():.4f}")

    sink_mass = []
    sink_argmax = []
    for layer in range(n_layers):
        a = attentions[layer][0].float()
        sink_mass.append(a[:, :, 0].mean().item())
        sink_argmax.append((a.argmax(-1) == 0).float().mean().item())

    mean_mass = float(np.mean(sink_mass))
    print(f"\nattention mass on position 0: {mean_mass:.1%}")
    print(f"rows whose argmax is position 0: {np.mean(sink_argmax):.1%}")
    for layer in range(0, n_layers, max(1, n_layers // 8)):
        print(f"  L{layer:02d} {sink_mass[layer]:6.1%}")
    print(f"\nposition 0 behaves as a sink: {mean_mass > 0.10}")

    steps = 64
    true_ids = ids
    protect = 1 if prepends_bos else 0

    def state_at(t):
        torch.manual_seed(42)
        maskable = torch.ones_like(true_ids, dtype=torch.bool)
        if protect:
            maskable[:, :protect] = False
        xt = true_ids.masked_fill(maskable, mask_id)
        remaining = maskable.clone()
        for progress in range(t):
            p_reveal = 1.0 / (steps - progress)
            reveal = remaining & (torch.rand_like(remaining, dtype=torch.float) < p_reveal)
            xt = xt.clone()
            xt[reveal] = true_ids[reveal]
            remaining &= ~reveal
        if t == steps - 1:
            xt, remaining = true_ids.clone(), torch.zeros_like(remaining)
        return xt, (~remaining[0]).cpu().tolist()

    print()
    for t in [0, 8, 16, 32, 48, 63]:
        xt, visible = state_at(t)
        shown = " ".join(
            tokenizer.decode([i]).strip() if v else "_"
            for i, v in zip(xt[0].tolist(), visible)
        )
        print(f"t={t:<3} {sum(visible):>2}/{len(visible)} visible   {shown}")

    _, v0 = state_at(0)
    xf, vf = state_at(steps - 1)
    assert sum(v0) == protect
    assert all(vf) and torch.equal(xf, true_ids)
    print(f"\nschedule ok: t=0 leaves {protect} visible, t=63 is exact")

    if not hasattr(model, "diffusion_generate"):
        print("\nno diffusion_generate on this revision, skipping")
        return

    enc = tokenizer("The capital city of France is", return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    t0 = time.time()
    gen = model.diffusion_generate(
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
    seq = gen.sequences if hasattr(gen, "sequences") else gen
    print(f"\ngenerated in {time.time() - t0:.1f}s")
    print(repr(tokenizer.decode(seq[0].tolist())))


@app.local_entrypoint()
def main():
    check.remote()
