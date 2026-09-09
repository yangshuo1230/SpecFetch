from dataclasses import dataclass

import pytest

from scripts.run_vllm_baseline import run_decode_only


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

    def _add_request(self, _prompt, _sampling):
        request_id = str(self.next_id)
        self.next_id += 1
        return request_id


def ticking_clock():
    values = iter([10.0, 12.5, 20.0])
    return lambda: next(values)


def test_decode_timer_starts_after_every_request_has_one_token():
    llm = FakeLLM(
        [
            [Output("0", [4])],
            [Output("0", [4]), Output("1", [5])],
            [Output("0", [4, 6], True), Output("1", [5, 7], True)],
        ]
    )

    run = run_decode_only(
        llm, [{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}], object(), ticking_clock()
    )

    assert run.prefill_seconds == 2.5
    assert run.decode_wall_seconds == 7.5
    assert [output.outputs[0].token_ids for output in run.outputs] == [[4, 6], [5, 7]]


def test_decode_timer_rejects_an_unobservable_multitoken_boundary():
    llm = FakeLLM([[Output("0", [4, 5], True)]])

    with pytest.raises(RuntimeError, match="multiple tokens"):
        run_decode_only(llm, [{"prompt_token_ids": [1]}], object(), ticking_clock())
