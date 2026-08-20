from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from .base import Capabilities, ModelAdapter

DIFFULLAMA_REPO = "https://github.com/HKUNLP/DiffuLLaMA.git"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def ensure_diffullama_repo(revision: str, root: str | Path = "third_party") -> Path:
    dest = Path(root) / "DiffuLLaMA"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", DIFFULLAMA_REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"DiffuLLaMA source revision mismatch: {actual} != {revision}")
    path = str(dest.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return dest


def _transformers_version() -> str:
    import transformers

    return getattr(transformers, "__version__", "unknown")


def checkpoint_keys(name: str, revision: str | None = None) -> set[str]:
    import json
    import struct

    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(name, revision=revision)
    index = [f for f in files if f.endswith("safetensors.index.json")]
    if index:
        path = hf_hub_download(name, index[0], revision=revision)
        with open(path) as fh:
            return set(json.load(fh)["weight_map"])

    shards = [f for f in files if f.endswith(".safetensors")]
    if not shards:
        return set()
    path = hf_hub_download(name, shards[0], revision=revision)
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return {k for k in header if k != "__metadata__"}


def _is_wrapper_checkpoint(name: str, revision: str | None = None) -> bool:
    try:
        return any(key.startswith("denoise_model.") for key in checkpoint_keys(name, revision))
    except Exception:  # noqa: BLE001
        return False


def _wrapper_key_map(key: str, body: str, embed_target: str) -> str | None:
    if key.startswith("denoise_model."):
        return f"{body}.{key[len('denoise_model.') :]}"
    if key == "embed_tokens.weight":
        return embed_target
    if key.startswith("lm_head."):
        return key
    return None


def _load_wrapper_checkpoint(
    cls, name: str, hf_config, dtype, revision: str, body: str = "model", embed_target: str | None = None
):
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    embed_target = embed_target or f"{body}.embed_tokens.weight"
    backbone = cls(hf_config)
    state: dict[str, torch.Tensor] = {}
    for shard in [file for file in list_repo_files(name, revision=revision) if file.endswith(".safetensors")]:
        state.update(load_file(hf_hub_download(name, shard, revision=revision)))

    remapped, dropped = {}, []
    for key, value in state.items():
        target = _wrapper_key_map(key, body, embed_target)
        if target is None:
            dropped.append(key)
        else:
            remapped[target] = value

    missing, _ = backbone.load_state_dict(remapped, strict=False)
    backbone.tie_weights()
    tied = {n for n, _ in backbone.named_parameters()}
    hard_missing = [k for k in missing if k in tied and not k.startswith("lm_head")]
    if hard_missing:
        raise RuntimeError(
            f"{name}: {len(hard_missing)} parameters were not loaded, e.g. "
            f"{hard_missing[:5]}. The wrapper key map is wrong for this model."
        )
    return backbone.to(dtype)


def _assert_pretrained_weights_loaded(backbone, name: str, revision: str | None = None) -> None:
    try:
        ckpt = checkpoint_keys(name, revision)
    except Exception:  # noqa: BLE001
        return
    if not ckpt:
        return
    expected = set(backbone.state_dict())
    coverage = len(ckpt & expected) / max(len(expected), 1)
    if coverage >= 0.9:
        return
    stray = sorted({k.split(".")[0] for k in ckpt - expected})[:5]
    raise RuntimeError(
        f"{name} populated only {coverage:.0%} of {type(backbone).__name__}'s "
        f"parameters, so most of the model is randomly initialized. Top-level "
        f"checkpoint names are {stray}; load it through its wrapper instead."
    )


def load_backbone(
    cls, name: str, hf_config, dtype, body: str = "model", embed_target: str | None = None, **extra
):
    revision = extra.get("revision")
    if _is_wrapper_checkpoint(name, revision):
        return _load_wrapper_checkpoint(cls, name, hf_config, dtype, revision, body, embed_target)

    attempts = [
        {"dtype": dtype, "attn_implementation": hf_config._attn_implementation},
        {"torch_dtype": dtype, "attn_implementation": hf_config._attn_implementation},
        {"torch_dtype": dtype, "_attn_implementation": hf_config._attn_implementation},
        {"torch_dtype": dtype},
    ]
    last = None
    for kwargs in attempts:
        try:
            backbone = cls.from_pretrained(name, config=hf_config, **kwargs, **extra)
            _assert_pretrained_weights_loaded(backbone, name, revision)
            return backbone
        except TypeError as exc:
            last = exc
    raise RuntimeError(
        f"could not load {name} under transformers {_transformers_version()}; last error: {last}"
    )


class WrappedAdapter(ModelAdapter, torch.nn.Module):
    mask_free = False
    final_norm_attr = "norm"
    capabilities = Capabilities(
        logits=True,
        hidden_states=True,
        attentions=True,
    )

    def __init__(self, ddm, tokenizer, device: str):
        torch.nn.Module.__init__(self)
        ModelAdapter.__init__(self, ddm, tokenizer, device)
        self.denoise_model = ddm.denoise_model

    def get_embeds(self, input_ids):
        return self.backbone.get_embeds(input_ids)

    def get_logits(self, hidden_state):
        return self.backbone.get_logits(hidden_state)

    def get_final_norm(self):
        return getattr(self.denoise_model, self.final_norm_attr)

    def get_lm_head(self):
        return self.backbone.lm_head

    @torch.no_grad()
    def forward_attentions(self, input_ids, output_hidden_states: bool = False):
        from model import get_anneal_attn_mask

        embeds = self.get_embeds(input_ids)
        mask = get_anneal_attn_mask(
            seq_len=input_ids.shape[1],
            bsz=input_ids.shape[0],
            dtype=embeds.dtype,
            device=input_ids.device,
            attn_mask_ratio=1.0,
        )
        out = self.denoise_model(
            inputs_embeds=embeds,
            attention_mask=mask,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=False,
        )
        if output_hidden_states:
            return None, out.attentions, out.hidden_states
        return None, out.attentions


def load_wrapped_family(
    model_cfg: dict,
    adapter_cls,
    backbone_cls,
    body: str,
    embed_target: str,
    device_map: str | None = None,
):
    code_revision = model_cfg["remote_code_revision"]
    ensure_diffullama_repo(code_revision)
    from transformers import AutoConfig, AutoTokenizer

    checkpoint = model_cfg.get("checkpoint", model_cfg.get("name"))
    dtype = _DTYPES[model_cfg.get("dtype", "bfloat16")]
    device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    attn = model_cfg.get("attn_implementation", "eager")

    revision = model_cfg["revision"]
    tokenizer_revision = model_cfg["tokenizer_revision"]
    hf_config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    hf_config._attn_implementation = attn
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=tokenizer_revision)

    extra = {"revision": revision}
    if device_map:
        extra["device_map"] = device_map
    backbone = load_backbone(
        backbone_cls, checkpoint, hf_config, dtype, body=body, embed_target=embed_target, **extra
    )

    try:
        from model import DiscreteDiffusionModel
    except AttributeError as exc:
        raise RuntimeError(
            "DiffuLLaMA's attention patch needs transformers==4.44.2; install it "
            "and restart the runtime "
            f"(underlying error: {exc})"
        ) from exc

    ddm = DiscreteDiffusionModel(model=backbone, config=hf_config, tokenizer=tokenizer, device=device).to(
        device
    )
    ddm.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(f"{checkpoint} exposes no mask token")

    adapter = adapter_cls(ddm, tokenizer, device).eval()
    meta = {
        "checkpoint": checkpoint,
        "revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "remote_code_revision": code_revision,
        "capabilities": adapter.capabilities.__dict__,
        "n_layers": hf_config.num_hidden_layers,
        "n_heads": hf_config.num_attention_heads,
        "hidden_size": hf_config.hidden_size,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    return adapter, tokenizer, meta


def load_diffullama(model_cfg: dict, adapter_cls):
    import transformers

    return load_wrapped_family(
        model_cfg,
        adapter_cls,
        transformers.LlamaForCausalLM,
        body="model",
        embed_target="model.embed_tokens.weight",
        device_map="auto",
    )


def load_diffugpt(model_cfg: dict, adapter_cls):
    import transformers

    return load_wrapped_family(
        model_cfg,
        adapter_cls,
        transformers.GPT2LMHeadModel,
        body="transformer",
        embed_target="transformer.wte.weight",
    )
