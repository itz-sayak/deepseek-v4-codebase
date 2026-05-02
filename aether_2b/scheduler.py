"""Continuous-batching decode scheduler for the Aether 2B serving engine.

Implements:
- Iteration-level (continuous) batching: new requests slot in as soon as a
  sequence finishes, without waiting for the whole batch to complete.
- Shared compressed prefix blocks: requests with the same prompt prefix hash
  reuse the same on-disk prefix cache via LongContextServingManager without
  copying the prefix tensors per request.
- Batched token-by-token decode: all active sequences step together through the
  model each iteration; sequences that finish or exceed max_new_tokens are
  drained and replaced.
- **Speculative decoding** (May 2026): pass ``draft_engine`` to enable draft-model
  assisted decoding for all requests.  Adaptive-K and sweep-backed defaults are
  loaded automatically from ``spec_defaults.py`` when that file exists (generated
  by ``scripts/apply_best_spec_config.py`` after running the grid sweep).

The scheduler intentionally does NOT duplicate the engine's prefix-cache logic;
it delegates to Aether2BServingEngine.prefill_with_reuse() for prefill.

Usage
-----
    from aether_2b.serving import Aether2BServingEngine
    from aether_pipeline.serving import LongContextServingManager
    from aether_2b.scheduler import DecodeScheduler, GenerationRequest

    # Standard (no speculative decode)
    engine = Aether2BServingEngine(model, prefix_manager=manager)
    scheduler = DecodeScheduler(engine, max_batch_size=8)

    # With speculative decode + adaptive K (defaults from sweep result)
    scheduler = DecodeScheduler(
        engine,
        draft_engine=draft_engine,
        default_draft_steps=6,
        adaptive_draft_steps=True,
    )

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

from .speculative import SpeculativeDecoder

# Pull sweep-validated defaults when the constants file exists.
try:
    from .spec_defaults import (
        SPEC_DEFAULT_SELF_SPEC_LAYERS as _DEF_SELF_SPEC_LAYERS,
        SPEC_DEFAULT_DRAFT_STEPS as _DEF_DRAFT_STEPS,
        SPEC_DEFAULT_TEMPERATURE as _DEF_TEMPERATURE,
        SPEC_DEFAULT_MIN_DRAFT_STEPS as _DEF_MIN_K,
        SPEC_DEFAULT_MAX_DRAFT_STEPS as _DEF_MAX_K,
        SPEC_DEFAULT_ADAPT_UP_THRESHOLD as _DEF_UP_THR,
        SPEC_DEFAULT_ADAPT_DOWN_THRESHOLD as _DEF_DOWN_THR,
        SPEC_DEFAULT_DRAFT_QUANT_BITS as _DEF_DRAFT_QUANT_BITS,
    )
except ImportError:
    _DEF_SELF_SPEC_LAYERS = 16
    _DEF_DRAFT_STEPS = 6
    _DEF_TEMPERATURE = 0.8
    _DEF_MIN_K = 1
    _DEF_MAX_K = 8
    _DEF_UP_THR = 0.90
    _DEF_DOWN_THR = 0.60
    _DEF_DRAFT_QUANT_BITS = None


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
    draft_steps: int = 0
    # populated by scheduler after prefill
    _state: object = field(default=None, repr=False)
    _output_ids: List[int] = field(default_factory=list, repr=False)
    _done: bool = field(default=False, repr=False)
    _submit_time: float = field(default_factory=time.perf_counter, repr=False)
    _first_token_time: Optional[float] = field(default=None, repr=False)
    _pending_logits: Optional[torch.Tensor] = field(default=None, repr=False)
    # speculative decode state (populated by scheduler when draft_engine is set)
    _draft_state: object = field(default=None, repr=False)
    _spec_buffer: List[int] = field(default_factory=list, repr=False)


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
        A ``Aether2BServingEngine`` instance.
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
        draft_engine=None,
        default_draft_steps: int = 0,
        default_spec_temperature: float = _DEF_TEMPERATURE,
        adaptive_draft_steps: bool = True,
        min_draft_steps: int = _DEF_MIN_K,
        max_draft_steps: int = _DEF_MAX_K,
        adapt_up_threshold: float = _DEF_UP_THR,
        adapt_down_threshold: float = _DEF_DOWN_THR,
        draft_quant_bits: int = None,
    ) -> None:
        self._engine = engine
        self._max_batch_size = max_batch_size
        self._greedy = greedy
        self._pending: Queue[GenerationRequest] = Queue(maxsize=max_queue_size)
        self._active: List[GenerationRequest] = []
        self._default_spec_temperature = default_spec_temperature
        self._adaptive_draft_steps = adaptive_draft_steps
        self._min_draft_steps = min_draft_steps
        self._max_draft_steps = max_draft_steps
        self._adapt_up_threshold = adapt_up_threshold
        self._adapt_down_threshold = adapt_down_threshold
        self._lock = threading.Lock()
        # Speculative decode: optional draft engine shared across all requests.
        # Per-request draft_steps overrides; default_draft_steps is the fallback.
        self._draft_engine = draft_engine
        self._default_draft_steps = default_draft_steps
        self._draft_quant_bits = draft_quant_bits
        self._spec_decoders: Dict[str, SpeculativeDecoder] = {}
        # Offload SWA tensors to host RAM between decode steps to free GPU memory.
        # Only enabled when the engine exposes offload_swa_to_host/restore_swa_to_device.
        self._swa_offload = (
            swa_offload
            and hasattr(engine, "offload_swa_to_host")
            and hasattr(engine, "restore_swa_to_device")
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_self_spec_defaults(
        cls,
        engine,
        max_batch_size: int = 8,
        **kwargs,
    ) -> "DecodeScheduler":
        """Build a scheduler pre-wired with the sweep-optimal self-spec config.

        Reads depth, draft steps, temperature, adaptive-K bounds, and quant bits
        from :mod:`spec_defaults` (auto-generated by ``apply_best_spec_config.py``).
        The draft model shares the first ``SPEC_DEFAULT_SELF_SPEC_LAYERS`` layers
        with the target model, preserving any multi-GPU sharding already applied.

        Parameters
        ----------
        engine :
            A ``Aether2BServingEngine`` already placed on GPU(s).
        max_batch_size : int
            Passed to the scheduler.
        **kwargs :
            Extra kwargs forwarded to :class:`DecodeScheduler` (e.g. ``swa_offload``).

        Example
        -------
        ::

            engine = Aether2BServingEngine(model, backend=backend)
            engine.shard_across_gpus(2)
            scheduler = DecodeScheduler.from_self_spec_defaults(engine, max_batch_size=4)
            scheduler.submit(GenerationRequest("r0", prompt_ids, max_new_tokens=256))
            results = scheduler.run_until_done()
        """
        from .speculative import build_self_spec_draft_model

        depth = _DEF_SELF_SPEC_LAYERS

        draft_model = build_self_spec_draft_model(engine.model, depth)
        # Preserve multi-GPU sharding: shared layers are already on the right
        # devices; only set _layer_devices so the engine skips the bulk .to() call.
        target_layer_devices = getattr(engine.model.model, "_layer_devices", None)
        if target_layer_devices is not None:
            draft_model.model._layer_devices = list(target_layer_devices[:depth])

        # Import inline to avoid circular dependency (serving → speculative, not scheduler).
        from .serving import Aether2BServingEngine
        draft_engine = Aether2BServingEngine(
            draft_model,
            backend=engine.backend,
            device=engine.device.type,
            turbo_quant_bits=_DEF_DRAFT_QUANT_BITS,  # None = no quant, best tok/s
        )

        return cls(
            engine,
            max_batch_size=max_batch_size,
            draft_engine=draft_engine,
            default_draft_steps=_DEF_DRAFT_STEPS,
            default_spec_temperature=_DEF_TEMPERATURE,
            adaptive_draft_steps=True,
            min_draft_steps=_DEF_MIN_K,
            max_draft_steps=_DEF_MAX_K,
            adapt_up_threshold=_DEF_UP_THR,
            adapt_down_threshold=_DEF_DOWN_THR,
            **kwargs,
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
                # Prefill draft engine when speculative decode is enabled.
                effective_draft_steps = req.draft_steps or self._default_draft_steps
                if effective_draft_steps > 0 and self._draft_engine is not None and req._draft_state is None:
                    req._draft_state = self._draft_engine.prefill(list(req.prompt_ids))
                    req.draft_steps = effective_draft_steps
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

    def _get_or_create_spec_decoder(self, req: GenerationRequest) -> SpeculativeDecoder:
        """Return (creating if needed) the per-request SpeculativeDecoder."""
        if req.seq_id not in self._spec_decoders:
            temperature = 0.0 if self._greedy else self._default_spec_temperature
            self._spec_decoders[req.seq_id] = SpeculativeDecoder(
                self._engine,
                self._draft_engine,
                draft_steps=req.draft_steps,
                temperature=temperature,
                top_k=req.top_k,
                adaptive_draft_steps=self._adaptive_draft_steps,
                min_draft_steps=self._min_draft_steps,
                max_draft_steps=self._max_draft_steps,
                adapt_up_threshold=self._adapt_up_threshold,
                adapt_down_threshold=self._adapt_down_threshold,
            )
        return self._spec_decoders[req.seq_id]

    def _run_spec_round(self, req: GenerationRequest, prev_token: int):
        """Run one speculative round for *req*, returning (result, new_target_state, new_draft_state)."""
        decoder = self._get_or_create_spec_decoder(req)
        if decoder._shared_layer_fusion_depth > 0:
            return decoder._spec_round_shared_fused(req._state, req._draft_state, prev_token)
        return decoder._spec_round(req._state, req._draft_state, prev_token)

    def _step(self) -> List[FinishedSequence]:
        """Run one decode iteration over all active requests."""
        finished: List[FinishedSequence] = []
        still_active: List[GenerationRequest] = []

        for req in self._active:
            if req._state is None or req._done:
                continue

            # ---- determine next token ----
            if req._spec_buffer:
                # Drain a previously computed spec round — no model call needed.
                next_token = req._spec_buffer.pop(0)
            elif req._pending_logits is not None:
                next_token = self._sample(req._pending_logits[0], req)
            else:
                raise RuntimeError(f"Request {req.seq_id} has no pending logits and empty spec buffer")

            now = time.perf_counter()
            if req._first_token_time is None:
                req._first_token_time = now
            req._output_ids.append(next_token)

            is_eos = (req.eos_token_id is not None and next_token == req.eos_token_id)
            exceeded = (len(req._output_ids) - len(req.prompt_ids)) >= req.max_new_tokens

            if is_eos or exceeded:
                req._done = True
                # Clean up per-request spec decoder.
                self._spec_decoders.pop(req.seq_id, None)
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
            elif req._spec_buffer:
                # Still draining a spec round — no model advance until buffer is empty.
                still_active.append(req)
            elif req._draft_state is not None:
                # Spec buffer drained (or first token): kick off the next spec round.
                # prev_token is the token we just emitted; the round will process it
                # as the first verify step to advance both target and draft states.
                if self._swa_offload:
                    self._engine.restore_swa_to_device(req._state)
                result, req._state, req._draft_state = self._run_spec_round(req, next_token)
                if self._swa_offload:
                    self._engine.offload_swa_to_host(req._state)
                req._spec_buffer = result.accepted_tokens + [result.bonus_token]
                req._pending_logits = req._state.last_logits
                still_active.append(req)
            else:
                # Standard single-token advance.
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
