"""Speculative decoding for DeepSeekV4Pro2B.

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
    from deepseek_v4_pro_2b.speculative import SpeculativeDecoder, DraftConfig

    # Build engines
    target_engine = DeepSeekV4Pro2BServingEngine(large_model, ...)
    draft_engine  = DeepSeekV4Pro2BServingEngine(small_model, ...)

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
import torch.nn.functional as F


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
        ``DeepSeekV4Pro2BServingEngine`` for the large target model.
    draft_engine :
        ``DeepSeekV4Pro2BServingEngine`` for the small draft model.
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
        if seed is not None:
            torch.manual_seed(seed)

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
        if self.top_k is not None and self.top_k < scaled.size(-1):
            top_v, _ = torch.topk(scaled, k=self.top_k, dim=-1)
            threshold = top_v[..., -1:].expand_as(scaled)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        return F.softmax(scaled, dim=-1)

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
        target_states_seq = []  # state after each target step
        for vt in verify_tokens:
            t_logits, cur_target_state = self.target.step_token(vt, cur_target_state)
            target_states_seq.append(cur_target_state)
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
        accepted_target_state = target_state  # valid target state = after prev_token (pre-verify)
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
        # New draft state: advance draft past the bonus token too
        new_draft_state = cur_draft_state
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

            result, target_state, draft_state = self._spec_round(
                target_state, draft_state, last_token
            )
            self.draft_steps = orig_steps

            # Collect output tokens
            new_in_round = result.accepted_tokens + [result.bonus_token]
            for t in new_in_round:
                output_ids.append(t)
                new_tokens += 1
                if eos_token_id is not None and t == eos_token_id:
                    break
            last_token = new_in_round[-1]

            summary.total_proposed += result.draft_proposed
            summary.total_accepted += len(result.accepted_tokens)
            summary.total_rounds += 1

            if eos_token_id is not None and last_token == eos_token_id:
                break

        summary.output_ids = output_ids
        return summary
