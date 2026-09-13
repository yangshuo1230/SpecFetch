from dataclasses import dataclass

import pytest

from scripts.run_vllm_baseline import maximum_worker_memory_gib, run_decode_only


@dataclass
class Candidate:
    token_ids: list[int]


@dataclass
class Output:
    request_id: str
    token_ids: list[int]
    finished: bool = False

    @property
    def outputs(self):
        return [Candidate(self.token_ids)]


class FakeEngine:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.remaining = True

    def has_unfinished_requests(self):
        return self.remaining

    def step(self):
        outputs = next(self.steps)
        self.remaining = not all(output.finished for output in outputs)
        return outputs


class FakeLLM:
    def __init__(self, steps):
        self.llm_engine = FakeEngine(steps)
        self.next_id = 0
        self.prompts = []

    def _add_request(self, prompt, _sampling):
        request_id = str(self.next_id)
        self.next_id += 1
        self.prompts.append(prompt)
        self.llm_engine.remaining = True
        return request_id


def ticking_clock():
    values = iter([10.0, 12.5, 20.0])
    return lambda: next(values)


def test_decode_timer_uses_cached_first_token_prefixes():
    llm = FakeLLM(
        [
            [Output("0", [4], True), Output("1", [5], True)],
            [Output("2", [6, 8], True), Output("3", [7, 9], True)],
        ]
    )

    run = run_decode_only(
        llm,
        [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}],
        object(),
        object(),
        ticking_clock(),
    )

    assert run.prefill_seconds == 2.5
    assert run.decode_wall_seconds == 7.5
    assert run.first_token_ids == [4, 5]
    assert [output.outputs[0].token_ids for output in run.outputs] == [[6, 8], [7, 9]]
    assert llm.prompts == [
        {"prompt_token_ids": [1]},
        {"prompt_token_ids": [2]},
        {"prompt_token_ids": [1, 4]},
        {"prompt_token_ids": [2, 5]},
    ]


def test_decode_timer_rejects_invalid_first_token_preparation():
    llm = FakeLLM([[Output("0", [4, 5], True)]])

    with pytest.raises(RuntimeError, match="first-token preparation"):
        run_decode_only(
            llm,
            [{"prompt_token_ids": [1]}],
            object(),
            object(),
            ticking_clock(),
        )


def test_worker_memory_uses_maximum_rank_peaks():
    allocated, reserved = maximum_worker_memory_gib(
        [(1 * 2**30, 2 * 2**30), (3 * 2**30, 4 * 2**30)]
    )
    assert allocated == 3.0
    assert reserved == 4.0
    with pytest.raises(RuntimeError, match="no worker memory"):
        maximum_worker_memory_gib([])
