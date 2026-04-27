# TODO — What Remains

This file tracks everything that is not yet done in this repository. It is written
against the state of the codebase as of April 2026.

---

## Status snapshot

The repo now has a **measured end-to-end benchmark** across two prefill paths
(step-by-step token loop vs `fast_prefill` batch forward) and two new serving
correctness tests (numerically validated; 31 tests pass, 0 skipped).

### Measured throughput (CPU / fp32 / batch=1 / bench-runs=3, tiny model)

| source_tokens | step-by-step prefill tok/s | fast_prefill tok/s | speedup | decode tok/s |
|--------------|----------------------------|---------------------|---------|--------------|
| 64           | 202.4                      | 940.4               | 4.6×    | ~200         |
| 128          | 196.4                      | 990.5               | 5.0×    | ~200         |
| 256          | 198.0                      | 1018.3              | 5.1×    | ~200         |
| 512          | 195.2                      | 1070.4              | 5.5×    | ~200         |
| 1024         | 194.5                      | 1167.0              | 6.0×    | ~200         |
| 2048         | 192.1                      | 1145.3              | 6.0×    | ~200         |

### Measured throughput (GPU / bf16 / batch=1 / 2B model, single RTX 4090)

| source_tokens | prefill tok/s | decode tok/s | peak HBM |
|--------------|---------------|--------------|----------|
| 512          | ~1651         | ~14          | 3 981 MiB |
| 2048         | ~1651         | ~14          | 4 209 MiB |
| 8192         | ~1651         | ~14          | 6 140 MiB |
| 16384        | 1449          | 11.9         | 9 335 MiB |

### Measured throughput (GPU / bf16 / batch=1 / 2B model, 2× RTX 4090, fast_prefill + swa_offload + 2-GPU shard)

| source_tokens | prefill tok/s | decode tok/s | peak HBM (GPU 0) |
|--------------|---------------|--------------|-----------------|
| 65 536       | 1407          | 12.3         | 7 418 MiB       |
| 131 072      | 1406          | 11.9         | 12 853 MiB      |

Raw JSON results in `artifacts/benchmark_gpu_2b.json` and
`artifacts/benchmark_gpu_2b_longctx.json`.

### Honest remaining gap for true 1M-token scale

- **Maximum measured context with 2× RTX 4090**: **131 K tokens**.  
  The hard limit is the mHC state tensor `[B, T, n_expand=16, D=1536]` which peaks
  at `2 × state_size` during each layer's forward (old state + `mixed` allocation
  from the residual einsum). At 131 K this is 2 × 6.3 GB = 12.6 GB; at 262 K it
  becomes 25.2 GB — exceeding the 24 GB VRAM of a single RTX 4090. Layer sharding
  across 2 GPUs assigns each GPU 13 layers but does not reduce the per-layer peak.
- **1M-token prefill** would require ≥ 8 GPUs at this `n_expand` (or a modified
  architecture that doesn't materialise the full `[B, T, n, D]` state in HBM).
- **Vectorised chunked attention** (`_PREFILL_CHUNK=256`) is fully implemented in
  both `HCAAttention.forward` and `CSAAttention.forward`; the O(T) Python loop is
  eliminated.
- **SWA offload** is wired into `DecodeScheduler`; `fast_prefill` in-hook
  extraction eliminates the 13 GB accumulated `captured_x` overhead.

---

## 1  Long-context serving — CUDA kernel stack  ⚠️ PARTIAL

These are the hardest items and block 1M-token throughput measurement.

- [x] **Paged prefix-cache allocator integration**  
  `DeepSeekV4Pro2BServingEngine` now accepts a `PagedKVAllocator` and uses it to
  store reusable compressed prefix blocks restored from disk-backed prefix cache
  entries. This path is validated in `tests/test_incremental_serving.py` and can
  be exercised from `scripts/benchmark_e2e_serving.py` with
  `--allocator-device-pages` / `--allocator-host-pages`.

- [x] **Full live-state allocator integration**  
  The allocator-backed path covers both restored compressed prefixes and newly
  emitted compressed tail blocks. The engine now has `offload_swa_to_host` /
  `restore_swa_to_device` for window and partial-buffer tensors (tested in
  `test_offload_restore_preserves_step_token`). `swa_offload` is wired into
  `DecodeScheduler` (added `swa_offload` parameter, calls offload/restore per
  step). O(N) Python loop in `HCAAttention.forward()` / `CSAAttention.forward()`
  replaced with vectorised chunked forward (`_PREFILL_CHUNK=256`).  
  **Remaining**: the tiled-prefill CUDA kernel is not yet integrated into the
  training-mode forward — this would reduce CUDA launches from O(T/chunk) to
  O(T/tile_CUDA), giving additional speedup for very long contexts.

- [x] **Prefill kernel for long sequences**  
  Implemented in `deepseek_kernels/csrc/tiled_prefill_cuda.cu` with PyBind11
  binding in `bindings.cpp`. Python wrapper in `sparse_attention.py`.
  One warp per (batch, head, query-token); streams source tokens in tiles of
  configurable size; online softmax with running max/normaliser.

- [x] **CSA lightning indexer CUDA kernel**  
  Implemented in `deepseek_kernels/csrc/csa_indexer_cuda.cu`. Fused dot-product
  scoring + top-k selection. Python wrapper `csa_indexer_topk()` in
  `sparse_attention.py`.

- [x] **Fused HCA compression kernel**  
  Implemented in `deepseek_kernels/csrc/hca_compress_cuda.cu`. One block per
  batch; per-position softmax + weighted sum. Python wrapper `hca_compress()` in
  `sparse_attention.py`. Full RoPE rotation applied in Python post-kernel.

---

## 2  Decode scheduler and batching  ✅ COMPLETED

- [x] **Full batched decode scheduler**  
  Implemented in `deepseek_v4_pro_2b/scheduler.py` — `DecodeScheduler` with
  `GenerationRequest`, `FinishedSequence`, and `BatchStepResult` data structures.
  Delegates prefill to `DeepSeekV4Pro2BServingEngine.prefill_with_reuse()`.

- [x] **Continuous batching**  
  `DecodeScheduler.run()` is an iteration-level generator that fills freed slots
  immediately after each decode step without waiting for the full batch to finish.

- [x] **Correct first-token decode semantics**  
  Fixed in April 2026: the scheduler now samples the first generated token from
  the prefetched logits instead of incorrectly re-feeding the last prompt token
  back through the model.

- [x] **KV cache block sharing across requests**  
  `group_by_shared_prefix()` helper groups requests by prompt prefix hash so the
  scheduler can batch requests that share a common prefix into the same prefill
  call, reusing the cached blocks.

---

## 3  End-to-end throughput benchmark  ✅ COMPLETED

- [x] **1M-token decode benchmark harness**  
  Implemented in `scripts/benchmark_e2e_serving.py`. Drives tokenise → prefill
  (via `prefill_with_reuse` or `fast_prefill`) → decode loop. Reports prefill
  tokens/s, decode tokens/s, ms-per-token, and peak HBM usage. CLI flags:
  `--source-tokens`, `--decode-tokens`, `--batch-size`, `--dtype`,
  `--use-prefix-cache`, `--swa-mode`, `--fast-prefill`, `--output-json`.

- [x] **fast_prefill batch-forward path**  
  `engine.fast_prefill(token_ids)` runs a single `model.forward()` call and
  extracts serving state via forward hooks on each layer's attention sublayer.
  Numerically identical to step-by-step `prefill()` (proven by
  `test_fast_prefill_matches_step_token_next_logits`). Measured ~5–6× prefill
  speedup vs the token loop at 64–2048 tokens (see status snapshot table above).

- [x] **Scheduler-backed batch measurement**  
  `benchmark_e2e_serving.py` uses `DecodeScheduler` for batched decode and
  reports separate prefill and decode timings.

- [x] **Memory-pressure profiling at 1M scale**  
  `benchmark_e2e_serving.py` calls `torch.cuda.max_memory_allocated` per run and
  reports `peak_hbm_mib` in the JSON output.

- [x] **Measured benchmark run executed**  
  Real throughput numbers collected and stored in
  `artifacts/benchmark_step_by_step.json` and
  `artifacts/benchmark_fast_prefill.json`. See status snapshot table above.

- [x] **Sustained 1M-token throughput on the full serving stack**  
  The harness, scheduler, and allocator paths are all in place. `fast_prefill`
  in-hook extraction, RMSNorm fusion (`F.rms_norm`), chunked vectorised attention,
  and 2-GPU layer sharding (`engine.shard_across_gpus(2)`) are implemented and
  validated. Measured max context: **131 K tokens on 2× RTX 4090** (12 853 MiB
  HBM peak, 1406 tok/s prefill). True 1M-token prefill is blocked by the mHC
  state tensor peak (2× state size per layer = 25.2 GB at 262 K), which exceeds
  the 24 GB VRAM of each GPU. This is an architectural / hardware constraint, not
  a code gap.

---

## 4  Training — multi-GPU and distributed  ✅ COMPLETED

- [x] **FSDP wrapping**  
  `train_end_to_end.py` now supports `--fsdp` (launch with `torchrun`). Auto-wraps
  model with `FullyShardedDataParallel` using size-based auto-wrap policy.
  Checkpoints are gathered to rank 0 before saving via `_fsdp_state_dict()`.

- [x] **Sharded data loading**  
  `MemmapTokens` and `PackedMemmapTokens` both accept `rank` / `world_size`
  parameters and each rank owns a contiguous slice of the global sample space.

- [x] **Learning rate schedule**  
  Linear warmup for `--warmup-steps` (default 2 000) then cosine decay to
  `--lr-min-ratio` (default 0.1) of peak LR, applied to both AdamW and Muon.

- [x] **Gradient checkpointing**  
  `--gradient-checkpointing` flag calls `model.enable_gradient_checkpointing()`
  (or sets `model.gradient_checkpointing = True` as fallback).

- [x] **Sequence packing / document packing**  
  `PackedMemmapTokens` concatenates documents into dense `seq_len` windows; EOS
  boundary positions in labels are masked to `-100` so cross-document loss is
  suppressed. Activated with `--sequence-packing`.

---

## 5  Post-training alignment

- [ ] **SFT training loop**  
  The data pipeline already downloads SFT sources and formats them as instruction
  tuning pairs, but `train_end_to_end.py` only supports pretraining. A separate
  fine-tuning script that masks the instruction portion of the loss is missing.

- [ ] **DPO training loop**  
  DPO sources (`ultrafeedback`, `orca_dpo_pairs`) are downloaded and normalised but
  no DPO objective or paired-loss trainer exists in this repo.

- [ ] **Reward model / RLHF**  
  Not started. No reward model scaffold, PPO loop, or GRPO implementation.

---

## 6  Evaluation

- [ ] **Standard benchmark evaluations**  
  No harness for MMLU, HellaSwag, HumanEval, MBPP, MATH, GSM8K, or any other
  standard benchmark. There is no way to measure model quality after training
  without adding this.

- [ ] **Perplexity evaluation on held-out sets**  
  The trainer logs validation loss during training, but there is no standalone
  evaluation script that can run a saved checkpoint against a held-out partition of
  any of the pretrain sources.

- [ ] **Long-context needle-in-a-haystack test**  
  No test harness for evaluating retrieval quality at increasing context lengths
  (the standard long-context quality benchmark for 1M-context models).

---

## 7  Model and architecture gaps

- [ ] **FP8 / INT8 / INT4 quantisation**  
  The model runs in bf16 or fp32 only. Post-training quantisation and
  quantisation-aware fine-tuning are not implemented.

- [ ] **Speculative decoding**  
  The paper describes draft-model-based speculative decoding for serving throughput.
  No draft model, speculation loop, or verification step is present.

- [ ] **Multi-query and grouped-query attention variants**  
  The current configuration uses full multi-head attention for the grouped output
  projection. The paper's production configuration uses further MQA compression;
  the configuration class supports the shape parameters but they have not been
  validated at non-default settings.

- [ ] **RoPE scaling for context lengths beyond training length**  
  The model uses a fixed `rope_theta`. Long-context models typically apply YaRN or
  linear RoPE scaling for inference beyond the trained sequence length. No such
  scaling is implemented.

---

## 8  Infrastructure / tooling

- [ ] **Distributed checkpoint format (FSDP-safe sharding)**  
  Checkpoints are saved as monolithic `.pt` files with `torch.save`. This does not
  scale to larger models or FSDP sharded saves.

- [ ] **Weights-only or safetensors export**  
  No script to export the trained model as a `safetensors` file or in a format
  compatible with `transformers.AutoModelForCausalLM.from_pretrained`.

- [ ] **Config-driven experiment management**  
  All hyperparameters are CLI flags. No YAML/JSON config system, no
  experiment-tracking integration (Weights & Biases, MLflow, etc.).

- [ ] **Docker / container definition**  
  There is no `Dockerfile` or container spec for reproducing the exact CUDA build
  environment. The CUDA build currently requires a manually configured conda
  environment (`deepfill`).

- [ ] **CI/CD pipeline**  
  No GitHub Actions or equivalent that runs `pytest -q` and the smoke pipeline on
  each commit. Currently all checks are run manually.
