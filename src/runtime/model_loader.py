from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from src.runtime.expert import SafetensorExpertSource


def _weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    files = list(model_path.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError("expected a safetensors index or exactly one shard")
    with safe_open(files[0], framework="pt", device="cpu") as handle:
        return {name: files[0].name for name in handle}


def load_qwen3_non_expert(
    model_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    pin_experts: bool = True,
):
    """Load Qwen3-MoE without ever materializing routed experts on the GPU."""
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "eager"
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)
    for layer in model.model.layers:
        layer.mlp.experts = torch.nn.ModuleList()

    weights = _weight_map(model_path)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weights.items():
        if ".mlp.experts." not in name:
            by_shard[shard].append(name)
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for name in names:
                set_module_tensor_to_device(
                    model,
                    name,
                    device,
                    value=handle.get_tensor(name),
                    dtype=dtype,
                )
    missing = [name for name, value in model.named_parameters() if value.is_meta]
    if missing:
        raise RuntimeError(f"non-expert parameters remain on meta device: {missing[:3]}")
    source = SafetensorExpertSource(model_path, pin_memory=pin_experts)
    return model.eval(), source
