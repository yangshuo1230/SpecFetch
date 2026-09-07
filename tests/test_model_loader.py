import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.runtime.model_loader import load_qwen3_non_expert


def test_loader_keeps_experts_out_of_model_tree(tmp_path):
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    original = Qwen3MoeForCausalLM(config).eval()
    original.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="10KB")
    loaded, source = load_qwen3_non_expert(
        tmp_path, device="cpu", dtype=torch.float32, pin_experts=False
    )

    assert len(loaded.model.layers[0].mlp.experts) == 0
    assert torch.equal(loaded.model.embed_tokens.weight, original.model.embed_tokens.weight)
    assert torch.equal(
        loaded.model.layers[0].self_attn.q_proj.weight,
        original.model.layers[0].self_attn.q_proj.weight,
    )
    expert = source.get(0, 2)
    assert torch.equal(expert.gate, original.model.layers[0].mlp.experts[2].gate_proj.weight)
    assert expert.size_bytes > 0
