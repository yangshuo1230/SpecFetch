import json

import pytest

from src.workload import fixed_contexts, load_prompt_texts


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return type("Tokens", (), {"input_ids": [ord(value) for value in text]})


def test_fixed_contexts_repeat_without_padding(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(["ab", "xyz"]))
    prompts = load_prompt_texts(path)
    result = fixed_contexts(Tokenizer(), prompts, batch=2, tokens=5)
    assert result.tolist() == [[97, 98, 97, 98, 97], [120, 121, 122, 120, 121]]


def test_fixed_contexts_require_enough_prompts():
    with pytest.raises(ValueError, match="not enough"):
        fixed_contexts(Tokenizer(), ["a"], batch=2, tokens=2)
