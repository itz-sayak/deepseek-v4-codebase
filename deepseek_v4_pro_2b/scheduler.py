"""Continuous-batching decode scheduler for the DeepSeek-V4-Pro 2B serving engine.

Implements:
- Iteration-level (continuous) batching: new requests slot in as soon as a
  sequence finishes, without waiting for the whole batch to complete.
- Shared compressed prefix blocks: requests with the same prompt prefix hash
  reuse the same on-disk prefix cache via LongContextServingManager without
  copying the prefix tensors per request.
- Batched token-by-token decode: all active sequences step together through the
  model each iteration; sequences that finish or exceed max_new_tokens are
  drained and replaced.

The scheduler intentionally does NOT duplicate the engine's prefix-cache logic;
it delegates to DeepSeekV4Pro2BServingEngine.prefill_with_reuse() for prefill.

Usage
-----
    from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
    from deepseek_pipeline.serving import LongContextServingManager
    from deepseek_v4_pro_2b.scheduler import DecodeScheduler, GenerationRequest

    engine = DeepSeekV4Pro2BServingEngine(model, prefix_manager=manager)
    scheduler = DecodeScheduler(engine, max_batch_size=8)

    scheduler.submit(GenerationRequest("req-0", prompt_ids=[1, 2, 3], max_new_tokens=128))
    scheduler.submit(GenerationRequest("req-1", prompt_ids=[4, 5],    max_new_tokens=64))

    for batch_result in scheduler.run():
        for finished_seq in batch_result.finished:
            print(finished_seq.seq_id, finished_seq.output_ids)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from queue import Queue
from typing import Dict, Iterator, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    seq_id: str
    prompt_ids: List[int]
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    eos_token_id: Optional[int] = None
    # populated by scheduler after prefill
    _state: object = field(default=None, repr=False)
    _output_ids: List[int] = field(default_factory=list, repr=False)
    _done: bool = field(default=False, repr=False)
    _submit_time: float = field(default_factory=time.perf_counter, repr=False)
    _first_token_time: Optional[float] = field(default=None, repr=False)
    _pending_logits: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class FinishedSequence:
    seq_id: str
    prompt_ids: List[int]
    output_ids: List[int]
    latency_s: float
    ttft_s: Optional[float]  # time-to-first-token


@dataclass
class BatchStepResult:
    finished: List[FinishedSequence]
    active_count: int
    step_time_s: float


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DecodeScheduler:
    """Iteration-level (continuous) batching decode scheduler.

    Parameters
    ----------
    engine :
        A ``DeepSeekV4Pro2BServingEngine`` instance.
    max_batch_size : int
        Maximum number of sequences that run in the same decode step.
    max_queue_size : int
        Maximum number of pending requests waiting for a free slot.
    greedy : bool
        If True, always pick the top-1 token (temperature is ignored).
    """

    def __init__(
        self,
        engine,
        max_batch_size: int = 8,
        max_queue_size: int = 1024,
        greedy: bool = False,
        swa_offload: bool = False,
    ) -> None:
        self._engine = engine
        self._max_batch_size = max_batch_size
        self._greedy = greedy
        self._pending: Queue[GenerationRequest] = Queue(maxsize=max_queue_size)
        self._active: List[GenerationRequest] = []
        self._lock = threading.Lock()
        # Offload SWA tensors to host RAM between decode steps to free GPU memory.
        # Only enabled when the engine exposes offload_swa_to_host/restore_swa_to_device.
        self._swa_offload = (
            swa_offload
            and hasattr(engine, "offload_swa_to_host")
            and hasattr(engine, "restore_swa_to_device")
        )

    def submit(self, request: GenerationRequest) -> None:
        """Enqueue a generation request.  Thread-safe."""
        self._pending.put(request)

    def run(self) -> Iterator[BatchStepResult]:
        """Yield a BatchStepResult after each decode iteration until all requests finish.

        This is a blocking generator.  Call it from the serving thread.
        """
        while True:
            self._fill_active_slots()
            if not self._active:
                if self._pending.empty():
                    break
                # Wait for more requests.
                self._fill_active_slots(block=True)
                if not self._active:
                    break

            step_start = time.perf_counter()
            finished_this_step = self._step()
            step_time = time.perf_counter() - step_start

            yield BatchStepResult(
                finished=finished_this_step,
                active_count=len(self._active),
                step_time_s=step_time,
            )

    def run_until_done(self) -> List[FinishedSequence]:
        """Block until all submitted requests are complete.  Returns all results."""
        all_finished: List[FinishedSequence] = []
        for result in self.run():
            all_finished.extend(result.finished)
        return all_finished

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_active_slots(self, block: bool = False) -> None:
        """Move pending requests into the active batch up to max_batch_size."""
        with self._lock:
            while len(self._active) < self._max_batch_size:
                try:
                    req = self._pending.get(block=block, timeout=0.01 if block else 0)
                except Exception:
                    break
                # Prefill the new request unless the caller already provided a prefetched state.
                if req._state is None:
                    req._state = self._engine.prefill_with_reuse(req.prompt_ids)
                if not req._output_ids:
                    req._output_ids = list(req.prompt_ids)
                if req._pending_logits is None:
                    req._pending_logits = req._state.last_logits
                if self._swa_offload:
                    self._engine.offload_swa_to_host(req._state)
                self._active.append(req)
                block = False  # after first blocking get, be non-blocking

    def _sample(self, logits: torch.Tensor, req: GenerationRequest) -> int:
        if self._greedy or req.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())
        scaled = logits.float() / max(req.temperature, 1e-6)
        if req.top_k is not None and req.top_k > 0:
            top_vals, top_idx = torch.topk(scaled, k=min(req.top_k, scaled.size(-1)))
            probs = torch.softmax(top_vals, dim=-1)
            chosen = top_idx[torch.multinomial(probs, num_samples=1).item()].item()
            return int(chosen)
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _step(self) -> List[FinishedSequence]:
        """Run one decode iteration over all active requests."""
        finished: List[FinishedSequence] = []
        still_active: List[GenerationRequest] = []

        for req in self._active:
            if req._state is None or req._done:
                continue
            if req._pending_logits is None:
                raise RuntimeError(f"Request {req.seq_id} is missing pending logits")
            next_token = self._sample(req._pending_logits[0], req)

            now = time.perf_counter()
            if req._first_token_time is None:
                req._first_token_time = now
            req._output_ids.append(next_token)

            is_eos = (req.eos_token_id is not None and next_token == req.eos_token_id)
            exceeded = (len(req._output_ids) - len(req.prompt_ids)) >= req.max_new_tokens

            if is_eos or exceeded:
                req._done = True
                free_state = getattr(self._engine, "free_state", None)
                if free_state is not None:
                    free_state(req._state)
                finished.append(FinishedSequence(
                    seq_id=req.seq_id,
                    prompt_ids=req.prompt_ids,
                    output_ids=req._output_ids[len(req.prompt_ids):],
                    latency_s=now - req._submit_time,
                    ttft_s=req._first_token_time - req._submit_time if req._first_token_time else None,
                ))
            else:
                if self._swa_offload:
                    self._engine.restore_swa_to_device(req._state)
                logits, new_state = self._engine.step_token(next_token, req._state)
                if self._swa_offload:
                    self._engine.offload_swa_to_host(new_state)
                req._state = new_state
                req._pending_logits = logits
                still_active.append(req)

        self._active = still_active
        # Immediately try to fill freed slots.
        self._fill_active_slots()
        return finished


# ---------------------------------------------------------------------------
# Convenience: shared-prefix batch helper
# ---------------------------------------------------------------------------

def group_by_shared_prefix(
    requests: Sequence[GenerationRequest],
    min_shared_tokens: int = 64,
) -> Dict[str, List[GenerationRequest]]:
    """Group requests by shared prompt prefix so the scheduler can exploit
    prefix-cache reuse across requests in the same group.

    Returns a mapping from prefix hash to the list of requests that share it.
    Requests with unique prompts (or short prompts below *min_shared_tokens*)
    are grouped under the empty string key.
    """
    import hashlib

    groups: Dict[str, List[GenerationRequest]] = {}
    for req in requests:
        prefix = req.prompt_ids[:min_shared_tokens]
        if len(prefix) < min_shared_tokens:
            key = ""
        else:
            hasher = hashlib.sha256()
            for token in prefix:
                hasher.update(int(token).to_bytes(4, "little", signed=False))
            digest = hasher.hexdigest()[:16]
            key = digest
        groups.setdefault(key, []).append(req)
    return groups
