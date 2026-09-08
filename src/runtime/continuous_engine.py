from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from src.runtime.batch_scheduler import ContinuousBatchScheduler, ServingRequest
from src.runtime.predictor import DraftSignalProvider
from src.runtime.qwen3_engine import BatchState, Qwen3SparseOffloadEngine, StepPredictions


@dataclass
class RequestExecution:
    provider: DraftSignalProvider
    token_id: torch.Tensor
    horizons: list[StepPredictions]
    generated_token_ids: list[int] = field(default_factory=list)
    pending_actual: list[torch.Tensor] = field(default_factory=list)


@dataclass
class ContinuousBatchResult:
    generated_token_ids: dict[str, list[int]]
    decode_cycles: int
    admission_events: int
    maximum_active_requests: int


def _request_state(state: BatchState, request_id: str) -> BatchState:
    index = state.request_ids.index(request_id)
    return BatchState(
        [request_id],
        [state.lengths[index]],
        {
            (owner, layer): cache
            for (owner, layer), cache in state.kv.items()
            if owner == request_id
        },
        step=state.step,
    )


def merge_predictions(
    request_ids: list[str],
    executions: dict[str, RequestExecution],
    horizon: int,
) -> StepPredictions:
    """Combine independent per-request draft signals in target batch order."""
    merged = StepPredictions()
    per_request = {}
    for request_id in request_ids:
        available = executions[request_id].horizons
        per_request[request_id] = available[horizon] if horizon < len(available) else None
        if per_request[request_id] is not None:
            merged.kv.update(per_request[request_id].kv)

    layers = set().union(
        *(set(prediction.experts) for prediction in per_request.values() if prediction is not None)
    )
    for layer in layers:
        rows = [
            per_request[request_id].experts.get(layer)
            if per_request[request_id] is not None
            else None
            for request_id in request_ids
        ]
        if any(row is None for row in rows):
            raise ValueError("expert predictions must cover every active request")
        merged.experts[layer] = torch.cat(rows, dim=0)
    return merged


class ContinuousBatchRunner:
    """Admission, backfill and variable-length decode for the Qwen runtime."""

    def __init__(
        self,
        engine: Qwen3SparseOffloadEngine,
        provider_factory: Callable[[], DraftSignalProvider],
        *,
        max_batch_size: int,
        prefetch_horizons: int = 1,
        prefetch: bool = True,
    ) -> None:
        if prefetch_horizons <= 0:
            raise ValueError("prefetch_horizons must be positive")
        self.engine = engine
        self.provider_factory = provider_factory
        self.max_batch_size = max_batch_size
        self.prefetch_horizons = prefetch_horizons
        self.prefetch = prefetch

    def _refresh(
        self,
        state: BatchState,
        request_id: str,
        execution: RequestExecution,
    ) -> None:
        if execution.horizons:
            return
        if execution.pending_actual:
            actual = torch.stack(execution.pending_actual)[None]
            execution.provider.advance(actual)
            execution.pending_actual.clear()
        plan = execution.provider.predict(_request_state(state, request_id))
        execution.horizons = plan.horizons

    def run(self, requests: list[ServingRequest]) -> ContinuousBatchResult:
        scheduler = ContinuousBatchScheduler(self.max_batch_size)
        for request in requests:
            scheduler.submit(request)
        state = BatchState([], [], {})
        executions: dict[str, RequestExecution] = {}
        generated = {request.request_id: [] for request in requests}
        decode_cycles = 0
        admission_events = 0
        maximum_active = 0

        while not scheduler.done:
            admitted = scheduler.admit()
            if admitted:
                admission_events += 1
            for request in admitted:
                input_ids = torch.tensor(request.prompt_token_ids, dtype=torch.long)[None]
                output = self.engine.prefill(input_ids, [request.request_id])
                provider = self.provider_factory()
                provider.initialize(input_ids, [request.request_id])
                plan = provider.predict(output.state)
                token = output.logits[0, -1].argmax().detach().cpu()
                generated[request.request_id].append(int(token))
                executions[request.request_id] = RequestExecution(
                    provider,
                    token,
                    plan.horizons,
                    generated[request.request_id],
                )
                self.engine.add_state(state, output.state)

            maximum_active = max(maximum_active, len(scheduler.active))
            first_token_finished = scheduler.record_decode(
                [request.request_id for request in admitted]
            )
            if first_token_finished:
                finished_ids = [request.request_id for request in first_token_finished]
                self.engine.remove_requests(state, finished_ids)
                for request_id in finished_ids:
                    executions.pop(request_id)

            active_ids = scheduler.active_ids
            if not active_ids:
                continue
            for request_id in active_ids:
                self._refresh(state, request_id, executions[request_id])
            available_horizons = min(
                self.prefetch_horizons,
                min(len(executions[request_id].horizons) for request_id in active_ids),
            )
            predictions = [
                merge_predictions(active_ids, executions, horizon)
                for horizon in range(available_horizons)
            ]
            token_ids = torch.stack([executions[item].token_id for item in active_ids])
            output = self.engine.decode(
                token_ids,
                state,
                predictions,
                prefetch=self.prefetch,
            )
            decode_cycles += 1
            for index, request_id in enumerate(active_ids):
                execution = executions[request_id]
                execution.pending_actual.append(execution.token_id)
                execution.horizons.pop(0)
                execution.token_id = output.logits[index, -1].argmax().detach().cpu()
                execution.generated_token_ids.append(int(execution.token_id))

            finished = scheduler.record_decode(active_ids)
            if finished:
                finished_ids = [request.request_id for request in finished]
                self.engine.remove_requests(state, finished_ids)
                for request_id in finished_ids:
                    executions.pop(request_id)

        return ContinuousBatchResult(
            generated,
            decode_cycles,
            admission_events,
            maximum_active,
        )
