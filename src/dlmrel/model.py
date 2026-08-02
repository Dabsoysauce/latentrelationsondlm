"""Loading diffusion language models with attention weights exposed.

Both supported models come from HKUNLP/DiffuLLaMA and wrap a causal-LM backbone
in a `DiscreteDiffusionModel`. That repository is not on PyPI, so it is cloned
at runtime; `ensure_diffullama_repo` handles that and puts it on `sys.path`.

The single most important detail: only the *eager* attention implementation
returns attention weights. `sdpa` and `flash_attention_2` accept
`output_attentions=True` and silently return None, which turns every accuracy
in this project into zero without raising.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from .config import ModelConfig

DIFFULLAMA_REPO = "https://github.com/HKUNLP/DiffuLLaMA.git"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def ensure_diffullama_repo(root: str | Path = "third_party") -> Path:
    """Clone the DiffuLLaMA repo if absent and make its modules importable."""
    dest = Path(root) / "DiffuLLaMA"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[model] cloning {DIFFULLAMA_REPO} -> {dest}")
        subprocess.run(
            ["git", "clone", "--depth", "1", DIFFULLAMA_REPO, str(dest)],
            check=True,
        )
    path = str(dest.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return dest


def load_model(cfg: ModelConfig, repo_root: str | Path = "third_party"):
    """Return `(model, tokenizer, meta)` ready for attention extraction."""
    ensure_diffullama_repo(repo_root)

    from model import DiscreteDiffusionModel
    from transformers import AutoConfig, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(cfg.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    dtype = _DTYPES[cfg.dtype]

    if cfg.family == "diffullama":
        from transformers import LlamaForCausalLM

        backbone = LlamaForCausalLM.from_pretrained(
            cfg.name,
            device_map="auto",
            _attn_implementation=cfg.attn_implementation,
            torch_dtype=dtype,
        )
    elif cfg.family == "diffugpt":
        from transformers import GPT2LMHeadModel

        backbone = GPT2LMHeadModel.from_pretrained(
            cfg.name,
            _attn_implementation=cfg.attn_implementation,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f"unknown model family {cfg.family!r}")

    model = DiscreteDiffusionModel(
        model=backbone,
        config=hf_config,
        tokenizer=tokenizer,
        device=cfg.device,
    ).to(cfg.device)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"{cfg.name} exposes no mask token; the diffusion schedule needs one"
        )

    meta = {
        "name": cfg.name,
        "n_layers": hf_config.num_hidden_layers,
        "n_heads": hf_config.num_attention_heads,
        "hidden_size": hf_config.hidden_size,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    print(
        f"[model] {cfg.name}: {meta['n_layers']} layers x {meta['n_heads']} heads "
        f"({meta['n_layers'] * meta['n_heads']} heads searched per relation)"
    )
    _assert_attentions_returned(model, tokenizer, cfg.device)
    return model, tokenizer, meta


@torch.no_grad()
def _assert_attentions_returned(model, tokenizer, device: str) -> None:
    """Fail loudly at load time rather than silently producing zero accuracy."""
    from .diffusion import forward_with_attentions

    ids = torch.tensor(
        [[tokenizer.bos_token_id] + tokenizer.encode("The cat sat on the mat.")],
        device=device,
    )
    from model import get_anneal_attn_mask

    embeds = model.get_embeds(ids)
    mask = get_anneal_attn_mask(
        seq_len=ids.shape[1],
        bsz=1,
        dtype=embeds.dtype,
        device=ids.device,
        attn_mask_ratio=1.0,
    )
    _, attentions = forward_with_attentions(model, ids, mask)
    if not attentions or attentions[0] is None:
        raise RuntimeError(
            "model returned no attention weights -- set "
            "model.attn_implementation to 'eager' (sdpa and flash_attention_2 "
            "drop them without error)"
        )
