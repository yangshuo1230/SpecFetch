from __future__ import annotations

import time
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
    prefill_batches: int
    maximum_prefill_batch: int
    draft_prefill_batches: int
    maximum_draft_prefill_batch: int
    request_timings: dict[str, dict[str, float]]


def _subset_state(state: BatchState, request_ids: list[str]) -> BatchState:
    indices = [state.request_ids.index(request_id) for request_id in request_ids]
    selected = set(request_ids)
    return BatchState(
        list(request_ids),
        [state.lengths[index] for index in indices],
        {(owner, layer): cache for (owner, layer), cache in state.kv.items() if owner in selected},
        step=state.step,
    )


def _request_state(state: BatchState, request_id: str) -> BatchState:
    return _subset_state(state, [request_id])


def _split_predictions(
    predictions: list[StepPredictions], request_ids: list[str]
) -> dict[str, list[StepPredictions]]:
    """将批量 Draft 信号按请求拆分，同时保持 horizon 顺序。"""
    split = {request_id: [] for request_id in request_ids}
    for prediction in predictions:
        for index, request_id in enumerate(request_ids):
            split[request_id].append(
                StepPredictions(
                    kv={
                        key: scores for key, scores in prediction.kv.items() if key[0] == request_id
                    },
                    experts={
                        layer: probabilities[index : index + 1]
                        for layer, probabilities in prediction.experts.items()
                    },
                )
            )
    return split


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


def _group_admitted_by_prompt_length(
    requests: list[ServingRequest],
) -> list[list[ServingRequest]]:
    """保持准入顺序，将可使用同一稠密张量的请求合并预填充。"""
    groups: dict[int, list[ServingRequest]] = {}
    for request in requests:
        groups.setdefault(len(request.prompt_token_ids), []).append(request)
    return list(groups.values())


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
        prefill_batches = 0
        maximum_prefill_batch = 0
        draft_prefill_batches = 0
        maximum_draft_prefill_batch = 0
        run_start = time.perf_counter()
        request_timings = {
            request.request_id: {
                "admission_seconds": 0.0,
                "time_to_first_token_seconds": 0.0,
                "completion_seconds": 0.0,
                "queue_seconds": 0.0,
                "prefill_seconds": 0.0,
                "decode_service_seconds": 0.0,
                "active_service_seconds": 0.0,
                "request_latency_seconds": 0.0,
            }
            for request in requests
        }

        def finish_requests(request_ids: list[str]) -> None:
            completed_at = time.perf_counter() - run_start
            for request_id in request_ids:
                timing = request_timings[request_id]
                timing["completion_seconds"] = completed_at
                timing["decode_service_seconds"] = (
                    completed_at - timing["time_to_first_token_seconds"]
                )
                timing["active_service_seconds"] = completed_at - timing["admission_seconds"]
                # All requests are submitted before run_start, so latency includes
                # pending-queue delay and equals the completion offset.
                timing["request_latency_seconds"] = completed_at

        while not scheduler.done:
            admitted = scheduler.admit()
            if admitted:
                admission_events += 1
                maximum_active = max(maximum_active, len(scheduler.active))
            admitted_at = time.perf_counter() - run_start
            for request in admitted:
                request_timings[request.request_id]["admission_seconds"] = admitted_at
                request_timings[request.request_id]["queue_seconds"] = admitted_at

            for group in _group_admitted_by_prompt_length(admitted):
                prefill_batches += 1
                maximum_prefill_batch = max(maximum_prefill_batch, len(group))
                group_ids = [request.request_id for request in group]
                input_ids = torch.tensor(
                    [request.prompt_token_ids for request in group], dtype=torch.long
                )
                output = self.engine.prefill(input_ids, group_ids)
                tokens = output.logits[:, -1].argmax(dim=-1).detach().cpu()
                first_token_at = time.perf_counter() - run_start
                self.engine.add_state(state, output.state)
                finished_ids = []
                continuing = []
                for index, request in enumerate(group):
                    request_id = request.request_id
                    token = tokens[index]
                    generated[request_id].append(int(token))
                    request_timings[request_id]["time_to_first_token_seconds"] = first_token_at
                    request_timings[request_id]["prefill_seconds"] = first_token_at - admitted_at
                    first_token_finished = scheduler.record_decode([request_id])
                    if first_token_finished:
                        finish_requests([request_id])
                        finished_ids.append(request_id)
                        continue
                    continuing.append((index, request))
                if continuing:
                    draft_prefill_batches += 1
                    maximum_draft_prefill_batch = max(maximum_draft_prefill_batch, len(continuing))
                    continuing_indices = torch.tensor([index for index, _ in continuing])
                    continuing_ids = [request.request_id for _, request in continuing]
                    batch_provider = self.provider_factory()
                    batch_provider.initialize(input_ids[continuing_indices], continuing_ids)
                    batch_plan = batch_provider.predict(_subset_state(output.state, continuing_ids))
                    split_horizons = _split_predictions(batch_plan.horizons, continuing_ids)
                    providers = batch_provider.split_requests()
                else:
                    providers = []
                    split_horizons = {}
                for provider, (index, request) in zip(providers, continuing):
                    request_id = request.request_id
                    executions[request_id] = RequestExecution(
                        provider,
                        tokens[index],
                        split_horizons[request_id],
                        generated[request_id],
                    )
                if finished_ids:
                    self.engine.remove_requests(state, finished_ids)

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
                finish_requests(finished_ids)
                self.engine.remove_requests(state, finished_ids)
                for request_id in finished_ids:
                    executions.pop(request_id)

        return ContinuousBatchResult(
            generated,
            decode_cycles,
            admission_events,
            maximum_active,
            prefill_batches,
            maximum_prefill_batch,
            draft_prefill_batches,
            maximum_draft_prefill_batch,
            request_timings,
        )
