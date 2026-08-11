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


def _load_backbone(cls, cfg: ModelConfig, hf_config, dtype, **extra):
    """`from_pretrained` across transformers versions that disagree on kwargs.

    Two names have moved under us and both fail loudly enough to waste a Colab
    session: `torch_dtype` became `dtype` in transformers 5, and
    `_attn_implementation` became `attn_implementation` in 4.36. Rather than
    pin a version and hope Colab agrees, try the spellings in order and keep
    the first that the installed version accepts.
    """
    attempts = [
        {"dtype": dtype, "attn_implementation": cfg.attn_implementation},
        {"torch_dtype": dtype, "attn_implementation": cfg.attn_implementation},
        {"torch_dtype": dtype, "_attn_implementation": cfg.attn_implementation},
        {"torch_dtype": dtype},
    ]
    if _is_wrapper_checkpoint(cfg.name):
        return _load_wrapper_checkpoint(cls, cfg, hf_config, dtype)

    last: Exception | None = None
    backbone = None
    for kwargs in attempts:
        try:
            backbone = cls.from_pretrained(cfg.name, config=hf_config, **kwargs, **extra)
            break
        except TypeError as exc:  # unexpected keyword for this version
            last = exc
    if backbone is None:
        raise RuntimeError(
            f"could not load {cfg.name} under transformers "
            f"{_transformers_version()}; last error: {last}"
        )
    _assert_pretrained_weights_loaded(backbone, cfg.name)
    return backbone


def _transformers_version() -> str:
    import transformers

    return getattr(transformers, "__version__", "unknown")


def _is_wrapper_checkpoint(name: str) -> bool:
    try:
        return any(k.startswith("denoise_model.") for k in checkpoint_keys(name))
    except Exception:  # noqa: BLE001 - detection is best-effort; fall through
        return False


def _wrapper_key_map(key: str, family: str) -> str | None:
    """Translate a DiscreteDiffusionModel parameter name to the backbone's.

    The wrapper holds the transformer body as `denoise_model` and hoists the
    input embedding out to `embed_tokens` (it deletes the body's own copy), so
    undoing it is a prefix rewrite plus one special case.
    """
    body = "transformer" if family == "diffugpt" else "model"
    if key.startswith("denoise_model."):
        return f"{body}.{key[len('denoise_model.'):]}"
    if key == "embed_tokens.weight":
        return f"{body}.wte.weight" if family == "diffugpt" else f"{body}.embed_tokens.weight"
    if key.startswith("lm_head."):
        return key
    return None


def _load_wrapper_checkpoint(cls, cfg: ModelConfig, hf_config, dtype):
    """Instantiate from config and load a wrapper-namespaced checkpoint."""
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    print(f"[model] {cfg.name} is saved in wrapper namespace; remapping keys")
    backbone = cls(hf_config)

    state: dict[str, torch.Tensor] = {}
    shards = [f for f in list_repo_files(cfg.name) if f.endswith(".safetensors")]
    for shard in shards:
        state.update(load_file(hf_hub_download(cfg.name, shard)))

    remapped, dropped = {}, []
    for key, value in state.items():
        target = _wrapper_key_map(key, cfg.family)
        if target is None:
            dropped.append(key)
        else:
            remapped[target] = value

    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    # GPT-2 ties lm_head to the input embedding, and the checkpoint stores only
    # the embedding, so the head must be re-tied after loading.
    backbone.tie_weights()
    tied = {n for n, _ in backbone.named_parameters()}
    hard_missing = [k for k in missing if k in tied and not k.startswith("lm_head")]

    print(
        f"[model] remapped {len(remapped)} tensors; {len(hard_missing)} missing, "
        f"{len(unexpected)} unexpected, {len(dropped)} dropped"
    )
    if hard_missing:
        raise RuntimeError(
            f"{cfg.name}: {len(hard_missing)} parameters were not loaded, e.g. "
            f"{hard_missing[:5]}. The wrapper key map is wrong for this model."
        )
    return backbone.to(dtype)


def checkpoint_keys(name: str) -> set[str]:
    """Parameter names in a Hub checkpoint, without downloading the weights."""
    import json
    import struct

    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(name)
    index = [f for f in files if f.endswith("safetensors.index.json")]
    if index:
        path = hf_hub_download(name, index[0])
        with open(path) as fh:
            return set(json.load(fh)["weight_map"])

    shards = [f for f in files if f.endswith(".safetensors")]
    if not shards:
        return set()
    path = hf_hub_download(name, shards[0])
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return {k for k in header if k != "__metadata__"}


def _assert_pretrained_weights_loaded(backbone, name: str) -> None:
    """Fail if `from_pretrained` matched almost nothing and silently randomized.

    This is not hypothetical. `diffusionfamily/diffugpt-s` is serialized under
    DiffuLLaMA's *wrapper* parameter names (`denoise_model.*`, `embed_tokens.*`)
    rather than GPT2LMHeadModel's (`transformer.*`), so loading it as a plain
    GPT-2 matches zero tensors, initializes the entire network at random, and
    only emits a warning. Every downstream accuracy then lands at chance with
    nothing raising. `diffusionfamily/diffullama` uses the standard LLaMA
    namespace and is unaffected, which is exactly the kind of asymmetry that
    makes this worth checking at load time rather than trusting.
    """
    try:
        ckpt = checkpoint_keys(name)
    except Exception as exc:  # noqa: BLE001 - offline/private/odd format
        print(f"[model] could not verify checkpoint keys for {name}: {exc}")
        return
    if not ckpt:
        return

    expected = set(backbone.state_dict())
    matched = len(ckpt & expected)
    coverage = matched / max(len(expected), 1)
    print(
        f"[model] checkpoint coverage: {matched}/{len(expected)} parameters "
        f"({coverage:.0%})"
    )
    if coverage >= 0.9:
        return

    stray = sorted({k.split(".")[0] for k in ckpt - expected})[:5]
    raise RuntimeError(
        f"{name} populated only {coverage:.0%} of {type(backbone).__name__}'s "
        f"parameters, so most of the model is randomly initialized. The "
        f"checkpoint's top-level names are {stray}, which suggests it was saved "
        "from a wrapper class rather than the bare backbone. Load it through "
        "that wrapper, or remap the state dict before scoring anything."
    )


class DreamAdapter(torch.nn.Module):
    """Expose Dream-7B through the interface the diffusion code expects.

    Dream is natively bidirectional, so unlike the DiffuLLaMA-family wrappers
    there is no anneal mask to build and no third-party repo to patch in:
    `mask_free=True` tells `diffusion.forward_with_attentions` to call the
    model directly with `attention_mask=None`. Keeping the DiffuLLaMA import
    out of this path matters because importing that repo's `model` module
    rewrites transformers-4.44 internals at import time, and Dream requires
    transformers>=4.51.
    """

    mask_free = True

    def __init__(self, model, device: str):
        super().__init__()
        self.model = model
        self.device = device

    def get_embeds(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)

    def get_logits(self, hidden_state):
        # The head search reads attentions only; no lm_head is needed.
        return None

    @torch.no_grad()
    def forward_attentions(self, input_ids):
        out = self.model(
            input_ids=input_ids,
            attention_mask=None,
            output_attentions=True,
            return_dict=True,
        )
        if getattr(out, "attentions", None) is None:
            raise RuntimeError(
                "Dream returned no attention weights -- the remote code is "
                "ignoring output_attentions (running sdpa/flash). Every "
                "accuracy would be zero. Load with attn_implementation='eager'."
            )
        return None, out.attentions


def _load_dream(cfg: ModelConfig):
    """Load `Dream-org/Dream-v0-Base-7B` (Qwen2-based, remote code).

    Wrinkles established by notebooks/smoke_test_dream7b.ipynb:
      * needs transformers>=4.51 -- mutually exclusive with the DiffuLLaMA pin
      * remote code may ask for rope_type="default" that newer builds spell
        "rope"
      * `trust_remote_code=True` is inherent to running Dream at all
      * the tokenizer does NOT prepend BOS itself, but a BOS token exists in
        the vocab, so `include_bos: true` gives position 0 a dedicated sink
        slot exactly as in the other two models
    """
    import torch as _torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS and "rope" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["rope"]
        print("[model] applied RoPE compatibility patch")

    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=True)
    dtype = _DTYPES[cfg.dtype]

    try:
        backbone = AutoModel.from_pretrained(
            cfg.name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation=cfg.attn_implementation,
        ).eval()
        print(f"[model] loaded with attn_implementation={cfg.attn_implementation}")
    except (TypeError, ValueError) as exc:
        print(f"[model] eager kwarg rejected ({exc}); using remote-code default")
        backbone = AutoModel.from_pretrained(
            cfg.name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
        ).eval()

    model = DreamAdapter(backbone, cfg.device)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"{cfg.name} exposes no mask token; the diffusion schedule needs one"
        )

    hf_config = backbone.config
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

    # Same fail-loudly probe as the legacy families, minus the anneal mask.
    probe_ids = _torch.tensor(
        [
            [tokenizer.bos_token_id]
            + tokenizer.encode("The cat sat on the mat.", add_special_tokens=False)
        ],
        device=cfg.device,
    )
    _, attentions = model.forward_attentions(probe_ids)
    if not attentions or attentions[0] is None:
        raise RuntimeError("Dream returned no attention weights at load time")
    return model, tokenizer, meta


def load_model(cfg: ModelConfig, repo_root: str | Path = "third_party"):
    """Return `(model, tokenizer, meta)` ready for attention extraction."""
    if cfg.family == "dream":
        return _load_dream(cfg)

    ensure_diffullama_repo(repo_root)

    try:
        from model import DiscreteDiffusionModel
    except AttributeError as exc:
        # DiffuLLaMA's attention_patch replaces LlamaModel.forward and
        # GPT2Model.forward with transformers 4.44 implementations, and patches
        # LlamaFlashAttention2, which no longer exists in later releases. There
        # is no forward-compatible workaround: the replacement forwards call
        # 4.44-era internals throughout.
        raise RuntimeError(
            "DiffuLLaMA's attention patch is incompatible with transformers "
            f"{_transformers_version()}. Install the pinned version and restart "
            "the runtime:\n"
            '    pip install "transformers==4.44.2" "tokenizers<0.20" '
            '"huggingface-hub<0.37"\n'
            f"(underlying error: {exc})"
        ) from exc

    from transformers import AutoConfig, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(cfg.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    dtype = _DTYPES[cfg.dtype]

    # Pin the attention implementation on the config as well as the call. The
    # DiffuLLaMA code was written against transformers 4.44, where the kwarg was
    # the private `_attn_implementation`; 4.36+ exposes `attn_implementation`
    # and ignores unknown private kwargs, which would silently leave us on sdpa
    # and return no attention weights at all. Setting the config field is the
    # one route that works across every version.
    hf_config._attn_implementation = cfg.attn_implementation

    if cfg.family == "diffullama":
        from transformers import LlamaForCausalLM

        backbone = _load_backbone(
            LlamaForCausalLM, cfg, hf_config, dtype, device_map="auto"
        )
    elif cfg.family == "diffugpt":
        from transformers import GPT2LMHeadModel

        backbone = _load_backbone(GPT2LMHeadModel, cfg, hf_config, dtype)
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
