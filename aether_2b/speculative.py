"""Speculative decoding for Aether2B.

Implements draft-model-based speculative decoding (Chen et al., 2023):
  1. A small *draft* model auto-regressively proposes K tokens.
  2. The large *target* model verifies all K+1 positions in a single batched forward.
  3. Tokens are accepted/rejected via rejection sampling — accepted tokens are
     identical in distribution to what the target model alone would produce.
  4. The first rejected token is resampled from a corrected distribution.

Why speculative decoding improves throughput
--------------------------------------------
The target model's ``step_token`` is memory-bandwidth-bound at batch-size 1
(decode is slow: one forward pass per token).  By verifying K proposed tokens
in one batched forward, throughput increases by the *acceptance rate* factor α:

    effective_tok/s ≈ (K * α + 1) * (1 / target_step_time)

At α ≈ 0.8 and K = 5, this is a ~4× speedup with no quality loss.

Usage
-----
    from aether_2b.speculative import SpeculativeDecoder, DraftConfig

    # Build engines
    target_engine = Aether2BServingEngine(large_model, ...)
    draft_engine  = Aether2BServingEngine(small_model, ...)

    decoder = SpeculativeDecoder(target_engine, draft_engine, draft_steps=5)

    # Greedy decode
    output_ids = decoder.generate(prompt_ids, max_new_tokens=256, temperature=0.0)

    # Sampled decode (temperature > 0)
    output_ids = decoder.generate(prompt_ids, max_new_tokens=256, temperature=1.0, top_k=50)

Self-speculative (same model for draft and target)
---------------------------------------------------
When passing the same engine as both target and draft, SpeculativeDecoder falls
back to standard greedy/sampled decoding (no speedup, but correct).

Architecture note
-----------------
The draft and target *share* the same tokenizer and vocab so token IDs are
directly comparable.  If you use different architectures, both must produce
logits over the same vocabulary (same vocab_size in their configs).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import Aether2BConfig
from .modeling import Aether2BForCausalLM


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpecDecodeResult:
    """Result of one speculative decode round."""
    accepted_tokens: List[int]       # tokens accepted from the draft proposal
    bonus_token: int                  # the guaranteed token from target rejection step
    draft_proposed: int               # how many draft tokens were proposed
    acceptance_rate: float            # accepted / proposed


@dataclass
class SpecDecodeSummary:
    """Aggregate statistics across a full generation."""
    output_ids: List[int]
    total_proposed: int = 0
    total_accepted: int = 0
    total_rounds: int = 0

    @property
    def mean_acceptance_rate(self) -> float:
        if self.total_proposed == 0:
            return 0.0
        return self.total_accepted / self.total_proposed

    @property
    def effective_speedup(self) -> float:
        """Theoretical speedup vs. one target step per token.
        actual speedup depends on hardware; this is the token-per-target-step ratio.
        """
        if self.total_rounds == 0:
            return 1.0
        tokens_per_round = len(self.output_ids) / self.total_rounds
        return tokens_per_round


# ---------------------------------------------------------------------------
# Core speculative decoder
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Draft-model-based speculative decoder.

    Parameters
    ----------
    target_engine :
        ``Aether2BServingEngine`` for the large target model.
    draft_engine :
        ``Aether2BServingEngine`` for the small draft model.
        May be the same engine as target for correctness testing.
    draft_steps : int
        Number of tokens the draft model auto-regressively proposes per round
        (K in the literature).  Typical values: 3–8.
    temperature : float
        Sampling temperature applied to both draft and target distributions.
        0.0 → greedy decoding.
    top_k : Optional[int]
        Top-K filtering applied to both draft and target.  None → no filtering.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        target_engine,
        draft_engine,
        draft_steps: int = 5,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        adaptive_draft_steps: bool = False,
        min_draft_steps: int = 1,
        max_draft_steps: Optional[int] = None,
        adapt_up_threshold: float = 0.90,
        adapt_down_threshold: float = 0.60,
    ) -> None:
        target_vocab = target_engine.model.config.vocab_size
        draft_vocab = draft_engine.model.config.vocab_size
        if target_vocab != draft_vocab:
            raise ValueError(
                f"Speculative decoding requires matching vocab_size, got target={target_vocab} and draft={draft_vocab}"
            )
        self.target = target_engine
        self.draft = draft_engine
        self.draft_steps = draft_steps
        self.temperature = temperature
        self.top_k = top_k
        self._self_spec = target_engine is draft_engine
        self.adaptive_draft_steps = adaptive_draft_steps
        self.min_draft_steps = max(1, int(min_draft_steps))
        self.max_draft_steps = max_draft_steps if max_draft_steps is not None else int(draft_steps)
        self.adapt_up_threshold = float(adapt_up_threshold)
        self.adapt_down_threshold = float(adapt_down_threshold)
        self._shared_layer_fusion_depth = self._detect_shared_layer_fusion_depth()
        if self.max_draft_steps < self.min_draft_steps:
            self.max_draft_steps = self.min_draft_steps
        if seed is not None:
            torch.manual_seed(seed)

    def _detect_shared_layer_fusion_depth(self) -> int:
        """Detect whether draft layers are an object-shared prefix of target layers.

        If true, target verification can skip those shared layers and execute only
        the tail layers using draft-produced hidden states.
        """
        if self._self_spec:
            return 0
        if not hasattr(self.target, "model") or not hasattr(self.draft, "model"):
            return 0
        if not hasattr(self.draft, "step_token_with_hidden"):
            return 0
        if not hasattr(self.target, "step_token_from_hidden"):
            return 0
        try:
            target_layers = list(self.target.model.model.layers)
            draft_layers = list(self.draft.model.model.layers)
        except Exception:
            return 0
        if len(draft_layers) == 0 or len(draft_layers) >= len(target_layers):
            return 0
        for i, layer in enumerate(draft_layers):
            if layer is not target_layers[i]:
                return 0
        return len(draft_layers)

    def _sync_shared_layer_states(self, target_state, draft_state, shared_depth: int) -> None:
        """Copy draft layer states into target, dequantizing compressed fields if needed.

        When the draft engine uses TurboQuant (4/8-bit), compressed KV tensors are
        stored packed (e.g. shape [..., D//2] for 4-bit).  The target engine has no
        polar_quant and would read them raw — causing a shape mismatch on torch.cat.
        We dequantize them here so the target always sees plain bf16 tensors.
        """
        draft_pq = getattr(self.draft, "_polar_quant", None)
        target_pq = getattr(self.target, "_polar_quant", None)
        needs_dequant = draft_pq is not None and target_pq is None

        if not needs_dequant:
            # Fast path: shared prefix layers are not executed by
            # step_token_from_hidden, so direct state aliasing is safe and avoids
            # repeated deep clones in the per-token verify loop.
            for i in range(shared_depth):
                target_state.layer_states[i] = draft_state.layer_states[i]
            return

        for i in range(shared_depth):
            ls = draft_state.layer_states[i].clone()
            from .serving import HCAServingState, CSAServingState
            if isinstance(ls, HCAServingState):
                if ls.compressed_scale is not None:
                    ls.compressed = self.draft._read_compressed(ls.compressed, ls.compressed_scale)
                    ls.compressed_scale = None
            elif isinstance(ls, CSAServingState):
                if ls.compressed_scale is not None:
                    ls.compressed = self.draft._read_compressed(ls.compressed, ls.compressed_scale)
                    ls.compressed_scale = None
                if ls.index_compressed_scale is not None:
                    ls.index_compressed = self.draft._read_compressed(ls.index_compressed, ls.index_compressed_scale)
                    ls.index_compressed_scale = None
            target_state.layer_states[i] = ls

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to a probability distribution with temperature + top-k."""
        if self.temperature <= 0:
            # Greedy: one-hot at argmax
            idx = logits.argmax(dim=-1, keepdim=True)
            p = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
            return p
        scaled = logits.float() / max(self.temperature, 1e-8)
        scaled = torch.nan_to_num(scaled, nan=0.0, posinf=1e9, neginf=-1e9)
        if self.top_k is not None and self.top_k < scaled.size(-1):
            top_v, _ = torch.topk(scaled, k=self.top_k, dim=-1)
            threshold = top_v[..., -1:].expand_as(scaled)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        norm = probs.sum()
        if not torch.isfinite(norm) or norm.item() <= 0.0:
            idx = torch.nan_to_num(logits, nan=float("-inf")).argmax(dim=-1, keepdim=True)
            probs = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        else:
            probs = probs / norm
        return probs

    def _sample(self, probs: torch.Tensor) -> int:
        """Sample a single token id from probability vector."""
        if self.temperature <= 0:
            return int(probs.argmax(dim=-1).item())
        return int(torch.multinomial(probs.view(-1), num_samples=1).item())

    # ------------------------------------------------------------------
    # Single round of speculative decoding
    # ------------------------------------------------------------------

    def _spec_round(
        self,
        target_state,
        draft_state,
        prev_token: int,
    ) -> Tuple[SpecDecodeResult, object, object]:
        """Run one speculative decode round.

        1. Draft model proposes K tokens auto-regressively.
        2. Target model verifies all K+1 positions in parallel (one step each
           here, since we use the serving engine's step-by-step path).
        3. Acceptance/rejection via the standard speculative decoding criterion.

        Returns
        -------
        result : SpecDecodeResult
        new_target_state : updated target serving state
        new_draft_state  : updated draft serving state (reset to match accepted prefix)
        """
        K = self.draft_steps

        # --- Phase 1: draft model proposes K tokens ---
        draft_token_ids: List[int] = []
        draft_probs_list: List[torch.Tensor] = []
        draft_state_snapshots = []  # state after each draft step (for rollback)

        cur_token = prev_token
        cur_draft_state = draft_state
        for _ in range(K):
            draft_state_snapshots.append(cur_draft_state)
            d_logits, cur_draft_state = self.draft.step_token(cur_token, cur_draft_state)
            p_draft = self._probs(d_logits[0])   # [vocab]
            t = self._sample(p_draft)
            draft_token_ids.append(t)
            draft_probs_list.append(p_draft.detach())
            cur_token = t

        # draft_token_ids = [t1, t2, ..., tK]
        # draft_probs_list[i] = p_draft(· | prompt + t1..ti)

        # --- Phase 2: target model verifies K+1 positions ---
        # Run target on: prev_token, t1, t2, ..., tK  (K+1 tokens)
        # Collect target logits at each position.
        target_probs_list: List[torch.Tensor] = []
        cur_target_state = target_state
        verify_tokens = [prev_token] + draft_token_ids  # K+1
        target_states_seq = []  # state snapshot after each target step
        for vt in verify_tokens:
            t_logits, cur_target_state = self.target.step_token(vt, cur_target_state)
            target_states_seq.append(cur_target_state.clone())
            p_target = self._probs(t_logits[0])  # [vocab]
            target_probs_list.append(p_target.detach())

        # target_probs_list[i] = p_target(· | prompt + verify_tokens[0..i])
        # We need: target at positions 1..K+1 (i.e. predicting next after each draft token)
        # target_probs_list[0] = p_target after seeing prev_token  → predicts t1
        # target_probs_list[1] = p_target after seeing prev_token,t1 → predicts t2
        # ...
        # target_probs_list[K] = p_target after seeing all K+1 → bonus token

        # --- Phase 3: acceptance/rejection ---
        # For each draft token i (1-indexed):
        #   accept with probability min(1, p_target(ti) / p_draft(ti))
        accepted: List[int] = []
        accepted_target_state = target_state.clone()  # valid target state = after prev_token (pre-verify)
        n_accepted = 0

        for i in range(K):
            ti = draft_token_ids[i]
            p_d = draft_probs_list[i][ti].clamp(min=1e-12).item()
            p_t = target_probs_list[i][ti].clamp(min=1e-12).item()
            accept_prob = min(1.0, p_t / p_d)

            if self.temperature <= 0:
                # Greedy: accept iff both argmax agree
                accept = (draft_token_ids[i] == int(target_probs_list[i].argmax()))
            else:
                accept = (torch.rand(1).item() < accept_prob)

            if accept:
                accepted.append(ti)
                # Advance accepted target state to after processing ti
                accepted_target_state = target_states_seq[i]
                n_accepted += 1
            else:
                # Resample from corrected distribution: max(0, p_t - p_d)
                if self.temperature <= 0:
                    bonus = int(target_probs_list[i].argmax())
                else:
                    p_corrected = (target_probs_list[i] - draft_probs_list[i]).clamp(min=0.0)
                    norm = p_corrected.sum()
                    if norm < 1e-12:
                        p_corrected = target_probs_list[i]
                    else:
                        p_corrected = p_corrected / norm
                    bonus = int(torch.multinomial(p_corrected, num_samples=1).item())
                # bonus token replaces the rejected draft token so we stop here
                result = SpecDecodeResult(
                    accepted_tokens=accepted,
                    bonus_token=bonus,
                    draft_proposed=K,
                    acceptance_rate=n_accepted / K,
                )
                # New draft state: re-sync draft to match accepted prefix + bonus
                # Run draft forward on accepted + bonus to rebuild its KV cache
                new_draft_state = draft_state  # reset to pre-round state
                for tok in accepted + [bonus]:
                    _, new_draft_state = self.draft.step_token(tok, new_draft_state)
                return result, accepted_target_state, new_draft_state

        # All K tokens accepted → sample bonus from target's last distribution
        bonus = self._sample(target_probs_list[K])
        result = SpecDecodeResult(
            accepted_tokens=accepted,
            bonus_token=bonus,
            draft_proposed=K,
            acceptance_rate=1.0,
        )
        # New draft state: advance draft through tK and bonus.
        new_draft_state = cur_draft_state
        _, new_draft_state = self.draft.step_token(draft_token_ids[-1], new_draft_state)
        _, new_draft_state = self.draft.step_token(bonus, new_draft_state)
        new_target_state = target_states_seq[K]
        return result, new_target_state, new_draft_state

    def _spec_round_shared_fused(
        self,
        target_state,
        draft_state,
        prev_token: int,
    ) -> Tuple[SpecDecodeResult, object, object]:
        """Shared-layer fused speculative round.

        Draft runs first N shared layers; target verification reuses that hidden and
        executes only tail layers [N, L) for the first K verify positions.
        """
        K = self.draft_steps
        shared_depth = self._shared_layer_fusion_depth

        draft_token_ids: List[int] = []
        draft_probs_list: List[torch.Tensor] = []
        shared_hidden_list: List[torch.Tensor] = []
        draft_state_after_steps = []

        cur_token = prev_token
        cur_draft_state = draft_state
        for _ in range(K):
            d_logits, cur_draft_state, shared_hidden = self.draft.step_token_with_hidden(cur_token, cur_draft_state)
            p_draft = self._probs(d_logits[0])
            t = self._sample(p_draft)
            draft_token_ids.append(t)
            draft_probs_list.append(p_draft.detach())
            shared_hidden_list.append(shared_hidden)
            draft_state_after_steps.append(cur_draft_state.clone())
            cur_token = t

        target_probs_list: List[torch.Tensor] = []
        verify_tokens = [prev_token] + draft_token_ids
        cur_target_state = target_state
        target_states_seq = []

        for i, vt in enumerate(verify_tokens):
            if i < K:
                self._sync_shared_layer_states(cur_target_state, draft_state_after_steps[i], shared_depth)
                t_logits, cur_target_state = self.target.step_token_from_hidden(
                    vt,
                    cur_target_state,
                    shared_hidden_list[i],
                    shared_depth,
                )
            else:
                t_logits, cur_target_state = self.target.step_token(vt, cur_target_state)
            target_states_seq.append(cur_target_state.clone())
            p_target = self._probs(t_logits[0])
            target_probs_list.append(p_target.detach())

        accepted: List[int] = []
        accepted_target_state = target_state.clone()
        n_accepted = 0

        for i in range(K):
            ti = draft_token_ids[i]
            p_d = draft_probs_list[i][ti].clamp(min=1e-12).item()
            p_t = target_probs_list[i][ti].clamp(min=1e-12).item()
            accept_prob = min(1.0, p_t / p_d)

            if self.temperature <= 0:
                accept = (draft_token_ids[i] == int(target_probs_list[i].argmax()))
            else:
                accept = (torch.rand(1).item() < accept_prob)

            if accept:
                accepted.append(ti)
                accepted_target_state = target_states_seq[i]
                n_accepted += 1
            else:
                if self.temperature <= 0:
                    bonus = int(target_probs_list[i].argmax())
                else:
                    p_corrected = (target_probs_list[i] - draft_probs_list[i]).clamp(min=0.0)
                    norm = p_corrected.sum()
                    if norm < 1e-12:
                        p_corrected = target_probs_list[i]
                    else:
                        p_corrected = p_corrected / norm
                    bonus = int(torch.multinomial(p_corrected, num_samples=1).item())

                result = SpecDecodeResult(
                    accepted_tokens=accepted,
                    bonus_token=bonus,
                    draft_proposed=K,
                    acceptance_rate=n_accepted / K,
                )

                new_draft_state = draft_state
                for tok in accepted + [bonus]:
                    _, new_draft_state = self.draft.step_token(tok, new_draft_state)
                return result, accepted_target_state, new_draft_state

        bonus = self._sample(target_probs_list[K])
        result = SpecDecodeResult(
            accepted_tokens=accepted,
            bonus_token=bonus,
            draft_proposed=K,
            acceptance_rate=1.0,
        )
        new_draft_state = cur_draft_state
        _, new_draft_state = self.draft.step_token(draft_token_ids[-1], new_draft_state)
        _, new_draft_state = self.draft.step_token(bonus, new_draft_state)
        new_target_state = target_states_seq[K]
        return result, new_target_state, new_draft_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> SpecDecodeSummary:
        """Generate up to *max_new_tokens* new tokens using speculative decoding.

        Parameters
        ----------
        prompt_ids : sequence of ints
            Input prompt token IDs.
        max_new_tokens : int
            Maximum number of new tokens to generate.
        eos_token_id : int, optional
            If set, stop when this token is sampled.

        Returns
        -------
        SpecDecodeSummary with output_ids, acceptance stats, and speedup estimate.
        """
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must be non-empty")

        # Prefill both models on the prompt
        target_state = self.target.prefill(list(prompt_ids))
        if self._self_spec:
            draft_state = target_state
        else:
            draft_state = self.draft.prefill(list(prompt_ids))

        output_ids: List[int] = list(prompt_ids)
        last_token = int(prompt_ids[-1])

        summary = SpecDecodeSummary(output_ids=output_ids)
        new_tokens = 0

        while new_tokens < max_new_tokens:
            remaining = max_new_tokens - new_tokens
            k = min(self.draft_steps, remaining)

            if self._self_spec or k == 0:
                # Fallback: single target step
                logits, target_state = self.target.step_token(last_token, target_state)
                t = self._sample(self._probs(logits[0]))
                output_ids.append(t)
                last_token = t
                new_tokens += 1
                if eos_token_id is not None and t == eos_token_id:
                    break
                continue

            # Temporarily override draft_steps for the final partial round
            orig_steps = self.draft_steps
            self.draft_steps = k

            if self._shared_layer_fusion_depth > 0:
                result, target_state, draft_state = self._spec_round_shared_fused(
                    target_state, draft_state, last_token
                )
            else:
                result, target_state, draft_state = self._spec_round(
                    target_state, draft_state, last_token
                )
            self.draft_steps = orig_steps

            # Collect output tokens
            new_in_round = result.accepted_tokens + [result.bonus_token]
            for t in new_in_round:
                if new_tokens >= max_new_tokens:
                    break
                output_ids.append(t)
                new_tokens += 1
                if eos_token_id is not None and t == eos_token_id:
                    break
            last_token = output_ids[-1]

            summary.total_proposed += result.draft_proposed
            summary.total_accepted += len(result.accepted_tokens)
            summary.total_rounds += 1

            if self.adaptive_draft_steps:
                if result.acceptance_rate >= self.adapt_up_threshold and self.draft_steps < self.max_draft_steps:
                    self.draft_steps += 1
                elif result.acceptance_rate <= self.adapt_down_threshold and self.draft_steps > self.min_draft_steps:
                    self.draft_steps -= 1

            if eos_token_id is not None and last_token == eos_token_id:
                break

        summary.output_ids = output_ids
        return summary


def build_self_spec_draft_model(
    target_model: Aether2BForCausalLM,
    draft_layers: int,
) -> Aether2BForCausalLM:
    """Build a self-spec draft model by sharing the first N layers with target.

    The draft model keeps the same tokenizer/vocab/head but executes fewer layers,
    which removes duplicate model-weight residency and typically lowers draft-step
    latency versus a separate tiny model.
    """
    total_layers = target_model.config.num_hidden_layers
    if draft_layers <= 0 or draft_layers >= total_layers:
        raise ValueError(
            f"draft_layers must be in [1, {total_layers - 1}], got {draft_layers}"
        )

    cfg_dict = target_model.config.to_dict()
    cfg_dict["num_hidden_layers"] = int(draft_layers)
    draft_cfg = Aether2BConfig.from_dict(cfg_dict)
    draft_model = Aether2BForCausalLM(draft_cfg)

    # Share modules to avoid duplicating weights in HBM.
    draft_model.model.embed_tokens = target_model.model.embed_tokens
    draft_model.model.layers = nn.ModuleList(
        [target_model.model.layers[i] for i in range(draft_layers)]
    )
    draft_model.model.final_norm = target_model.model.final_norm
    draft_model.model.post = target_model.model.post
    draft_model.lm_head = target_model.lm_head
    draft_model.mtp_heads = nn.ModuleList()
    draft_model.eval()
    return draft_model
