from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("dream7b-attention-heatmaps")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.51.3",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pandas",
        "matplotlib",
        "conllu",
        "pyyaml",
    )
    .add_local_dir(REPO_ROOT / "src" / "dlmrel", remote_path="/root/pkgs/dlmrel")
)

hf_cache = modal.Volume.from_name("dream7b-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("dlmrel-work", create_if_missing=True)

SENTENCE = "The chef prepared the meal carefully."
TIMESTEPS = [0, 12, 25, 37, 50, 63]
HEADS = [(2, 11, "relational head"), (0, 14, "sink head")]
GRID_LAYERS = [0, 24]
GRID_HEADS = list(range(9))


@app.function(image=image, gpu="A100", volumes={"/cache": hf_cache, "/work": work}, timeout=20 * 60)
def extract() -> None:
    import os
    import sys

    os.environ["HF_HOME"] = "/cache/huggingface"
    sys.path.insert(0, "/root/pkgs")

    from dlmrel.diffusion import states_at_time
    from dlmrel.evaluation import plotting
    from dlmrel.models.dream import load

    model, tokenizer, _ = load({"checkpoint": "Dream-org/Dream-v0-Base-7B"})
    plotting.set_style()

    frames = {}
    for t in TIMESTEPS:
        attentions, _, state = states_at_time(
            model, tokenizer, SENTENCE, diffusion_time=t, steps=64, seed=42, include_bos=True
        )
        toks = [
            tokenizer.decode([i]).strip() if v else "_"
            for i, v in zip(state.input_ids[0].tolist(), state.is_visible)
        ]
        frames[t] = (attentions, toks, state.n_masked)

    out = Path("/work/results/dream_7b/attention_heatmaps")
    out.mkdir(parents=True, exist_ok=True)

    masked_seq = [frames[t][2] for t in TIMESTEPS]

    for layer, head, label in HEADS:
        mats = [frames[t][0][layer][0, head].float().cpu().numpy() for t in TIMESTEPS]
        rows = [frames[t][1] for t in TIMESTEPS]
        plotting.attention_heatmaps(
            mats,
            rows,
            TIMESTEPS,
            masked_seq,
            f"Dream-7B  .  Layer {layer}, Head {head} ({label})  .  attention across diffusion timesteps",
            out,
            f"dream_L{layer}H{head}_heatmap",
        )

    for layer in GRID_LAYERS:
        grid = [
            [frames[t][0][layer][0, h].float().cpu().numpy() for t in TIMESTEPS]
            for h in GRID_HEADS
        ]
        rows = [frames[t][1] for t in TIMESTEPS]
        plotting.attention_head_grid(
            grid,
            GRID_HEADS,
            TIMESTEPS,
            rows,
            masked_seq,
            f"Dream-7B  .  Layer {layer}, Heads 0-8  .  attention across diffusion timesteps",
            out,
            f"dream_L{layer}_head_grid",
        )
    work.commit()
    print("done")


@app.local_entrypoint()
def main():
    extract.remote()
