# TODO — What Remains

This file tracks everything that is not yet done in this repository. It is written
against the state of the codebase as of April 2026.

---

## Status snapshot

The repo now has a **measured end-to-end benchmark** across two prefill paths
(step-by-step token loop vs `fast_prefill` batch forward) and two new serving
correctness tests (numerically validated; 31 tests pass, 0 skipped).

**April 2026 additions (this sprint)**:
- **TurboQuant / PolarQuant** — 8-bit and 4-bit compressed KV cache with
  Walsh-Hadamard + Lloyd-Max quantisation. Validated by needle-in-haystack eval:
  8-bit KL ≈ 0 (near-lossless), 4-bit KL < 0.001, 100% top-1 match at all tested
  context lengths (64–256 tokens on CPU tiny model; GPU YaRN runs at 2K–262K also
  pass with top-1 match 100% and KL in [4e-6, 8e-6]).
- **Prefill OOM fix** — `chunked_fast_prefill(mhc_chunk_size=4096)` reduces mHC
  peak from O(T×n×D) to O(C×n×D) + O(T×D). At T=262K, n=4, D=1536: 25.2 GB → 1.2 GB.
- **YaRN RoPE scaling** — config-driven long-context positional encoding. Validated
  at all context lengths up to 262K; top-1 match 100% with YaRN enabled.
- **Speculative decoding** — `SpeculativeDecoder` with draft-model acceptance/rejection
  (Chen et al., 2023). **Measured α = 0.97 (tiny draft, K=3)** — acceptance rate is
  excellent but net decode throughput is ~15% slower than baseline due to draft
  model overhead. See benchmark results and open item below.

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

### Measured throughput (GPU / bf16 / batch=1 / 2B model, 2× RTX 4090, fast_prefill / chunked_fast_prefill + 2-GPU shard)

| source_tokens | prefill tok/s | decode tok/s | peak HBM (GPU 0) |
|--------------|---------------|--------------|-----------------|
| 65 536       | 1407          | 12.3         | 7 418 MiB       |
| 131 072      | 1406          | 11.9         | 12 853 MiB      |
| 262 144      | 1140.2        | 9.8          | 16 941 MiB      |

### Measured speculative decode (GPU / bf16 / single RTX 4090 / YaRN / K=3 / tiny draft + 2B target)

| source_tokens | baseline tok/s | spec tok/s | acceptance rate | net speedup | target HBM | draft HBM |
|--------------|---------------|------------|-----------------|-------------|------------|-----------|
| 512          | 10.90         | 9.36       | 96.97%          | **0.86×**   | 3 818 MiB  | 3 777 MiB |
| 8 192        | 10.77         | 9.34       | 96.97%          | **0.87×**   | 4 318 MiB  | 3 874 MiB |
| 65 536       | 10.74         | 9.14       | 96.97%          | **0.85×**   | 7 780 MiB  | 4 597 MiB |
| 131 072      | 10.80         | 9.33       | 96.97%          | **0.86×**   | 11 791 MiB | 5 423 MiB |
| 262 144      | 10.99         | 9.15       | 96.97%          | **0.83×**   | 19 757 MiB | 7 075 MiB |

**Key finding**: α = 0.97 is excellent — the tiny draft and 2B target agree on tokens
almost perfectly. However, the tiny draft's 3 sequential forward passes per round cost
more wall-clock time than the single target verification saves. Speculative decoding
is currently **net slower** than baseline at all measured context lengths. The acceptance
rate is not the problem; draft forward-pass latency is. See section 7 for the fix path.

Raw JSON results in `artifacts/benchmark_speculative_k3.json`.

### Honest remaining gap for true 1M-token scale

- **Maximum measured context with 2× RTX 4090**: **262 K tokens** (chunked_fast_prefill).
- **1M-token prefill** requires ≥ 8 GPUs even with chunking — the attention sublayer
  still receives full T; O(T²) attention is the next hard wall.
- **Vectorised chunked attention** (`_PREFILL_CHUNK=256`) eliminates the O(T) Python loop.
- **SWA offload** is wired into `DecodeScheduler`; `fast_prefill` in-hook extraction
  eliminates the 13 GB accumulated `captured_x` overhead.

---

## 1  Long-context serving — CUDA kernel stack  ⚠️ PARTIAL

- [x] **Paged prefix-cache allocator integration**
- [x] **Full live-state allocator integration**
  Training-mode forward now has an opt-in tiled-prefill CUDA path in
  `deepseek_v4_pro_2b/modeling.py` (`use_tiled_prefill_cuda=True`) for HCA/CSA
  prefill when running on CUDA with no attention mask.
- [x] **Prefill kernel for long sequences** (`tiled_prefill_cuda.cu`)
- [x] **CSA lightning indexer CUDA kernel** (`csa_indexer_cuda.cu`)
- [x] **Fused HCA compression kernel** (`hca_compress_cuda.cu`)
- [x] **CUDA extension build/load path** (`deepseek_kernels/loader.py`)

---

## 2  Decode scheduler and batching  ✅ COMPLETED

- [x] Full batched decode scheduler (`DecodeScheduler`)
- [x] Continuous batching
- [x] Correct first-token decode semantics
- [x] KV cache block sharing across requests (`group_by_shared_prefix()`)

---

## 3  End-to-end throughput benchmark  ✅ COMPLETED

- [x] 1M-token decode benchmark harness (`benchmark_e2e_serving.py`)
- [x] `fast_prefill` batch-forward path (~5–6× speedup vs token loop)
- [x] Scheduler-backed batch measurement
- [x] Memory-pressure profiling
- [x] Measured benchmark runs executed and stored in artifacts

---

## 4  Training — multi-GPU and distributed  ✅ COMPLETED

- [x] FSDP wrapping (`torchrun`)
- [x] Sharded data loading
- [x] Learning rate schedule (linear warmup + cosine decay)
- [x] Gradient checkpointing
- [x] Sequence packing / document packing

---

## 5  Post-training alignment

- [ ] **SFT training loop**
  Data pipeline downloads and formats SFT sources but `train_end_to_end.py`
  only supports pretraining. Fine-tuning script with instruction-loss masking is missing.

- [ ] **DPO training loop**
  DPO sources downloaded and normalised. No DPO objective or paired-loss trainer.

- [ ] **Reward model / RLHF**
  Not started. No reward model scaffold, PPO loop, or GRPO implementation.

---

## 6  Evaluation

- [x] **Long-context needle-in-haystack TurboQuant + YaRN validation**
  `scripts/needle_eval.py`. Results saved to `artifacts/needle_eval_results.json`.
  All context lengths 64 → 262K pass at 4-bit with YaRN enabled.

- [ ] **Standard benchmark evaluations**  ⚠️ HIGH PRIORITY
  No harness for MMLU, HellaSwag, HumanEval, MBPP, MATH, or GSM8K.
  Training decisions cannot be made without quality metrics on a held-out eval set.
  Recommended path: integrate `lm-evaluation-harness` with a single
  `scripts/eval_checkpoint.py` wrapper.

- [ ] **Perplexity evaluation on held-out sets**
  Trainer logs validation loss during training but no standalone eval script
  exists to compare checkpoints post-hoc.

---

## 7  Model and architecture gaps

- [x] **TurboQuant / PolarQuant (KV-cache quantisation)**
  8-bit and 4-bit, all paths wired, 14 tests passing.
  Full weight quantisation (FP8 / INT8 for linear layers) not yet implemented.

- [x] **Speculative decoding — correctness**
  `SpeculativeDecoder` implemented and 11/11 tests pass. Output distribution
  identical to target-only decoding (proven). α = 0.97 measured on GPU.

- [ ] **Speculative decoding — net throughput improvement**  ⚠️ HIGH PRIORITY
  **Current status**: α = 0.97 is excellent but the tiny draft model's 3 sequential
  forward passes cost more wall-clock time than the target verification saves.
  Net result at K=3: ~0.85× (15% slower than baseline).

  **Root cause**: on GDDR6X, the tiny draft is memory-bandwidth-bound just like
  the target. Running 3 draft steps + 1 target step costs ~(3 × draft_latency) +
  target_latency, which exceeds a straight target_latency when draft_latency is not
  negligibly small.

    **Implemented optimizations**:
    1. **Shared-layer self-spec draft path** in `scripts/benchmark_speculative.py`
      via `--self-spec-layers N`, built by `build_self_spec_draft_model(...)`.
      Draft reuses the first N target layers (shared modules) to avoid loading a
      separate full set of weights for draft benchmarking.
    2. **Adaptive K control** in `SpeculativeDecoder` via
      `adaptive_draft_steps`, `min_draft_steps`, `max_draft_steps`, and
      acceptance-rate thresholds (`adapt_up_threshold`, `adapt_down_threshold`).
    3. **Quantise the draft model** — apply TurboQuant 4-bit to tiny draft KV cache,
     reducing draft memory bandwidth per step. May recover 20–30% draft latency.
     Implementation: pass `turbo_quant_bits=4` to draft engine. Low complexity.

    **Remaining tuning path**:
    - Increase K / run K-sweeps under stable GPU isolation and compare against
     shared-layer self-spec mode to find the first net-speedup regime.

    4. **Increase K** — at α = 0.97, K=5 or K=7 would generate more tokens per
     target call. May help if draft latency is sub-linear in K (batched draft steps).
     Test K=5 and K=7 before implementing self-speculative.

  **Recommended next action**: run controlled K-sweeps (K=3/5/8/12) with
  `--self-spec-layers` and `--adaptive-k` enabled, then pick the best operating
  point by net decode tok/s at 65K+ contexts.

- [ ] **Multi-query and grouped-query attention variants**
  Full MHA used for grouped output projection. Paper's production config uses MQA
  compression. Config supports the shape parameters but non-default settings
  are unvalidated.

- [x] **RoPE scaling for context lengths beyond training length**
  YaRN and linear scaling implemented and validated up to 262K tokens.

---

## 8  Infrastructure / tooling

- [ ] **Distributed checkpoint format (FSDP-safe sharding)**
  Monolithic `.pt` files via `torch.save`. Does not scale beyond current model size.

- [ ] **Weights-only or safetensors export**
  No script to export as `safetensors` or HuggingFace-compatible format.

- [ ] **Config-driven experiment management**
  All hyperparameters are CLI flags. No YAML/JSON config, no W&B / MLflow integration.

- [ ] **Docker / container definition**
  No `Dockerfile`. CUDA build requires manually maintained `deepfill` conda env.

- [ ] **CI/CD pipeline**
  No GitHub Actions. `pytest -q` and smoke pipeline run manually only.

---

## Priority order — next actions

```
🔴 URGENT

  1. Test K=5 and K=7 speculative decode (zero code change):
       python scripts/benchmark_speculative.py \
         --source-tokens 512 65536 131072 \
         --decode-tokens 128 --draft-steps 5 \
         --dtype bf16 --yarn-scaling \
         --output-json artifacts/benchmark_speculative_k5.json
     If net tok/s still < baseline → proceed to self-speculative decoding.

  2. Run self-spec + adaptive-K sweep and lock recommended default settings.

🟡 IMPORTANT

  3. Standard eval harness (HumanEval + GSM8K minimum via lm-evaluation-harness)
  4. SFT training loop
  5. Expand tiled-prefill CUDA path coverage to masked/packed training batches
  6. Perplexity eval script

🟢 LATER

  7. MQA output projection validation
  8. Draft model TurboQuant (quick win — zero new code needed)
  9. safetensors export
  10. Dockerfile + CI/CD
  11. YAML config + experiment tracking
```