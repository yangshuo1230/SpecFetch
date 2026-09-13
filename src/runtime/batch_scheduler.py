from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass
class ServingRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_tokens: int = 0

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens


class ContinuousBatchScheduler:
    """Framework-neutral admission state for a bounded continuous batch."""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.pending: deque[ServingRequest] = deque()
        self.active: OrderedDict[str, ServingRequest] = OrderedDict()
        self.completed: list[ServingRequest] = []
        self._known: set[str] = set()

    def submit(self, request: ServingRequest) -> None:
        if request.request_id in self._known:
            raise ValueError(f"duplicate request ID {request.request_id}")
        if not request.prompt_token_ids or request.max_new_tokens <= 0:
            raise ValueError("requests need prompt tokens and a positive output limit")
        self._known.add(request.request_id)
        self.pending.append(request)

    def admit(self) -> list[ServingRequest]:
        admitted = []
        while self.pending and len(self.active) < self.max_batch_size:
            request = self.pending.popleft()
            self.active[request.request_id] = request
            admitted.append(request)
        return admitted

    def record_decode(self, request_ids: list[str]) -> list[ServingRequest]:
        finished = []
        for request_id in request_ids:
            request = self.active[request_id]
            request.generated_tokens += 1
            if request.finished:
                finished.append(self.active.pop(request_id))
        self.completed.extend(finished)
        return finished

    @property
    def active_ids(self) -> list[str]:
        return list(self.active)

    @property
    def done(self) -> bool:
        return not self.pending and not self.active
