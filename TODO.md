# TODO — What Remains

This file tracks everything that is not yet done in this repository. It is written
against the state of the codebase as of April 2026.

---

## Status snapshot — May 2026

The repo now has a **measured end-to-end benchmark** across two prefill paths
(step-by-step token loop vs `fast_prefill` batch forward) and comprehensive
serving correctness tests (numerically validated; **73 tests pass, 0 skipped**).

All tests run unconditionally including the full 262K-token YaRN needle
diagnostic (`test_yarn_needle_top1_match`). Model renamed from DeepSeek-V4-Pro
to **Aether 2B**; packages renamed to `aether_2b`, `aether_pipeline`,
`aether_kernels`. CUDA kernel loader now auto-detects `nvcc` from
`CUDA_HOME/bin` and standard system paths so the 4 CUDA tests pass without
manually adding `/usr/local/cuda/bin` to `PATH`.

**May 2026 additions (this sprint)**:
- **NaN loss fix** — `ManifoldConstrainedHyperConnection` was passing raw (unnormalized)
  state to the sublayer, causing quadratic FFN blow-up (`O(0.02) → O(10²¹)` in 8
  layers). Fixed by using `norm_state = flat.view(bsz, seq_len, n, d)` for the
  sublayer input `x` while keeping the residual path on the unnormalized state.
  Training loss at step 1: **15.63** (finite). Applied in both `forward()` and
  `chunked_forward()`.
- **CSA math fixes** — Two mathematical bugs corrected in `CSAAttention`:
  1. **Block positions**: `_compress_main` now assigns end-of-block positions
     (`i*M + M-1`) matching HCA convention. Previously used start-of-block (`i*M`),
     causing relative RoPE distances to be off by `M-1=3` positions.
  2. **Indexer weights**: `F.softplus` applied to `index_w` before top-k scoring.
     Previously unconstrained (could be negative), causing top-k to potentially
     retrieve irrelevant blocks.
- **HCA apply_rope fix** — `HCAAttention._compress` now passes `config=self.config`
  to `apply_rope`, ensuring YaRN scaling is applied consistently to compressed keys.
- **Aesthetic console logging** — Training output replaced from raw JSON to
  human-readable format with banner, progress %, loss, ppl, tok/s, ETA.
- **Perplexity logging** — `train_ppl` and `val_ppl` added to per-step log and W&B.
- **Active pretraining** — SLURM job running on 2× L40S 48 GB, dense-2b variant,
  finite loss from step 1, W&B project `aether-2b`.

**April 2026 additions (previous sprint)**:
- **TurboQuant / PolarQuant** — 8-bit and 4-bit compressed KV cache.
- **Prefill OOM fix** — `chunked_fast_prefill(mhc_chunk_size=4096)`.
- **YaRN RoPE scaling** — config-driven long-context positional encoding.
- **Speculative decoding** — `SpeculativeDecoder` with α = 0.97 measured.

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
- [x] **CUDA extension build/load path** (`aether_kernels/loader.py`)
  Auto-detects nvcc from CUDA_HOME/bin fallback; PATH is patched before JIT compile.

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

- [x] **Speculative decoding — self-spec draft tuning**
  Full depth × temperature × quant grid swept on 2× RTX 4090 (262K context).
  Best operating point locked in `spec_defaults.py`:
  depth=16, temp=0.6, no draft quant → **6.377 tok/s, 0.861×** at 262K.

- [x] **Dense configuration variant**
  `dense_ffn_intermediate_size=14336` replaces all MoE routing with a single
  SwiGLU per layer. Total parameters: ~2.135 B.

- [x] **CSA math fix — block positions (Bug 1, May 2026)**
  `_compress_main` previously assigned start-of-block positions (`i*M`).
  Fixed to end-of-block (`i*M + M-1`) matching HCA convention.
  Impact: relative RoPE distances were off by `M-1=3` for all CSA layers;
  the first causal observer of block i now correctly encodes relative distance 1.

- [x] **CSA math fix — indexer weights (Bug 2, May 2026)**
  `index_w = self.index_weight(hidden_states)` (unconstrained linear) was
  multiplied directly against per-head similarities. Negative weights would flip
  relevant blocks to negative scores, causing top-k to retrieve wrong blocks.
  Fixed: `F.softplus(index_w)` ensures strictly positive head importance weights.

- [x] **HCA apply_rope fix — missing config (Bug 3, May 2026)**
  `HCAAttention._compress` now passes `config=self.config` to `apply_rope` so
  YaRN scaling is applied consistently to compressed block positions.

- [x] **NaN loss fix — mHC sublayer input normalization (May 2026)**
  `ManifoldConstrainedHyperConnection` now uses `norm_state` (RMSNorm-normalized)
  as sublayer input instead of raw `state`. Raw state grew O(0.02) → O(10²¹)
  in 8 layers via quadratic FFN blow-up. Loss at step 1 is now 15.63 (finite).

- [ ] **Multi-query and grouped-query attention variants**
  Full MHA used for grouped output projection. Non-default settings unvalidated.

- [x] **RoPE scaling for context lengths beyond training length**
  YaRN and linear scaling implemented and validated up to 262K tokens.

---

## 8  Training — active run

- [x] **FSDP multi-GPU training** — 2× L40S 48 GB (SLURM), torchrun + FSDP1
- [x] **Finite loss from step 1** — 15.63 (dense-2b, bf16, seq_len=4096)
- [x] **Perplexity logging** — `train_ppl` and `val_ppl` in per-step log and W&B
- [x] **Aesthetic console output** — banner + `[step/total %]` lines (not JSON)
- [ ] **First checkpoint at step 1000** — in progress
- [ ] **Convergence milestone: loss < 3.0** — not yet reached
- [ ] **Post-training: SFT and DPO** — data downloaded, no trainer yet

---

## 9  Infrastructure / tooling

- [ ] **Distributed checkpoint format (FSDP-safe sharding)**
- [ ] **Weights-only or safetensors export**
- [x] **Config-driven experiment management** — `configs/train.yaml`
- [x] **W&B experiment tracking** — wired in `train_end_to_end.py`, project `aether-2b`
- [ ] **Docker / container definition**
- [ ] **CI/CD pipeline**

---

## 10  Context Length Extension — Staged Plan

This section is the agent's detailed playbook for extending training context from
the current 4K to 262K. Do not start any stage until the stated trigger condition
is met.

---

### Why staged extension?

Training directly at 262K is infeasible for three reasons:
1. **Data scarcity**: documents genuinely longer than 32K are rare; the model
   learns nothing useful from padding-dominated batches.
2. **Memory**: at T=262144 the mHC state `[B, T, n=4, D=1536]` in bf16 is 6.4 GB
   *before* attention activations. `chunked_forward` reduces the mHC peak from
   O(T×n×D) to O(C×n×D), but it is not yet wired into the training loop.
3. **Compute efficiency**: short-context steps are ~50× cheaper per token than
   262K-context steps. Spending 90% of the token budget at 4K and 10% at long
   context is the standard industry recipe (LLaMA-3, Mistral, DeepSeek-V3 all do this).

---

### Stage 0 — Base pretraining  `[ACTIVE]`

| Field | Value |
|-------|-------|
| `seq_len` | 4096 |
| `rope_scaling_type` | `"none"` |
| `mhc_chunk_size` | N/A (standard `forward()`) |
| `batch_size × grad_accum × gpus` | 1 × 8 × 2 = 16 samples → 65,536 tok/step |
| Trigger to advance | `val_loss < 3.0`  AND  `step ≥ 3,000,000` (~20 B tokens) |

**Nothing to change.** Current run (job 4486) covers this stage.

---

### Stage 1 — Medium context (32K)  `[NOT STARTED]`

#### Trigger
`val_loss < 3.0` on the 4K val set **and** at least 3 M optimizer steps completed.

#### Config changes (`configs/train.yaml`)
```yaml
seq_len: 32768
rope_scaling_type: yarn
rope_scaling_factor: 8.0        # 32768 / 4096 = 8×
max_position_embeddings: 4096   # base training length (reference for YaRN ramp)
yarn_beta_fast: 32.0
yarn_beta_slow: 1.0
yarn_mscale: 0.1
mhc_chunk_size: 4096            # NEW field (see code change below)
batch_size: 1
grad_accum: 4                   # effective = 1 × 4 × 2 = 8 samples × 32K = 262K tok/step
warmup_steps: 200               # short re-warmup after ctx jump
```

#### Code changes required (agent: implement before launching Stage 1)

**1. Add `mhc_chunk_size` to `TrainConfig`** in `train_end_to_end.py`:
```python
mhc_chunk_size: int = 0   # 0 = standard forward(); >0 = chunked_forward(chunk_size)
```

**2. Wire `chunked_forward` into the training step** — replace the inner micro-batch forward:
```python
# Before (standard):
out = model(**batch)

# After (chunked when mhc_chunk_size > 0):
if cfg.mhc_chunk_size > 0:
    # model is FSDP-wrapped Aether2BForCausalLM;
    # call the underlying chunked forward via model.module.model.chunked_forward
    # but we still need loss — route through Aether2BForCausalLM.forward() with
    # precomputed hidden states, OR expose a chunked_forward on the top-level model.
    out = model.module.chunked_causal_lm_forward(batch, mhc_chunk_size=cfg.mhc_chunk_size)
else:
    out = model(**batch)
```
The cleanest approach is to add `chunked_causal_lm_forward(batch, mhc_chunk_size)` to
`Aether2BForCausalLM` that calls `self.model.chunked_forward(input_ids, mhc_chunk_size)`
instead of `self.model(input_ids)`, then applies `lm_head` and loss as usual.

**3. Update gradient checkpointing** — at 32K, gradient checkpointing should remain
enabled to keep backward-pass activation memory bounded. No change needed if
`gradient_checkpointing: true` is already set.

**4. Ensure long-context documents are in the training memmap** — the current tokenized
`train.bin` likely has documents truncated at 4096 tokens due to preprocessing. Before
launching Stage 1, re-run preprocessing with `--max-seq-len 32768` or use a
document-level memmap without truncation, then update `data_dir` in the config.

#### Expected memory (2× L40S 48 GB)
```
mHC state   [1, 32768, 4, 1536] bf16 ≈  0.77 GB  (with chunk_size=4096: peak ≈ 0.096 GB)
Attention    _PREFILL_CHUNK=256 inner loops inside HCA/CSA  ← already chunked
Activations  ~8 GB with gradient checkpointing
Total        ~9–12 GB per GPU  ✅ fits comfortably
```

#### Resume from checkpoint
```bash
# Edit configs/train.yaml: set seq_len, rope_scaling_type, mhc_chunk_size as above
# Then restart training — it auto-resumes from checkpoint-latest.pt
srun ... torchrun ... train_end_to_end.py --config configs/train.yaml
```

---

### Stage 2 — Long context (131K)  `[NOT STARTED]`

#### Trigger
`val_loss < 2.5` on 32K val set **and** Stage 1 has run for ≥ 200K optimizer steps.

#### Config changes
```yaml
seq_len: 131072
rope_scaling_factor: 32.0       # 131072 / 4096 = 32×
mhc_chunk_size: 4096
batch_size: 1
grad_accum: 1                   # 1 × 1 × 2 GPUs × 131K = 262K tok/step
warmup_steps: 100
```

#### Memory estimate
```
mHC chunked peak   [1, 4096, 4, 1536] bf16 per chunk ≈ 0.096 GB  ✅
Attention          HCA: O(T/M) compressed blocks = 1024 blocks at M=128  ✅
                   CSA: top-k=256 blocks selected per token  ✅
Activations        ~15 GB with gradient checkpointing
Total              ~16–20 GB per GPU  ✅ fits on L40S 48 GB
```

#### Additional requirement
At T=131072, the HCA inner prefill loop runs `ceil(T / _PREFILL_CHUNK) = 512` iterations.
Python loop overhead becomes measurable (~2–3 s per layer). Consider:
- Setting `use_tiled_prefill_cuda: true` in model config if CUDA kernels are built, OR
- Increasing `_PREFILL_CHUNK` from 256 to 1024 (trades memory for fewer Python iterations).

---

### Stage 3 — Max context (262K)  `[NOT STARTED]`

#### Trigger
`val_loss < 2.3` on 131K val set **and** Stage 2 has run for ≥ 100K optimizer steps.

#### Config changes
```yaml
seq_len: 262144
rope_scaling_factor: 64.0       # 262144 / 4096 = 64×
mhc_chunk_size: 4096
batch_size: 1
grad_accum: 1                   # 1 × 1 × 2 GPUs × 262K = 524K tok/step
warmup_steps: 50
```

#### Memory estimate (2× L40S 48 GB)
```
mHC chunked peak   0.096 GB per chunk  ✅
HCA compressed blocks  2048 blocks (M=128, T=262K)  ✅
Attention activations  ~30 GB per GPU with gradient checkpointing
Total  ~32–38 GB  ✅ fits, but tight — monitor with nvidia-smi
```

If OOM: add a 3rd or 4th GPU (`--gres=gpu:4`), or reduce `_PREFILL_CHUNK` to 128.

#### CUDA kernel activation (recommended at this stage)
Build the CUDA extension and set `use_tiled_prefill_cuda: true` in model config.
This replaces the Python prefill loop with a fused kernel, cutting prefill time by ~3×:
```bash
python scripts/build_cuda_kernels.py --verbose
# Then in configs/train.yaml (model_overrides section, to be added):
# use_tiled_prefill_cuda: true
```

---

### Stage summary

| Stage | seq_len | rope_factor | tok/step | Trigger | Key code change |
|-------|---------|------------|----------|---------|-----------------|
| 0 (active) | 4,096 | 1× (none) | 65K | — | — |
| 1 | 32,768 | 8× (YaRN) | 262K | val_loss < 3.0, step ≥ 3M | add `mhc_chunk_size` to trainer, `chunked_causal_lm_forward` on model |
| 2 | 131,072 | 32× (YaRN) | 262K | val_loss < 2.5, step ≥ 200K post-stage1 | optionally increase `_PREFILL_CHUNK` |
| 3 | 262,144 | 64× (YaRN) | 524K | val_loss < 2.3, step ≥ 100K post-stage2 | CUDA tiled prefill, 4× GPU if needed |

All stages resume from `checkpoint-latest.pt` automatically (`auto_resume: true`).
Only `configs/train.yaml` changes needed between stages.

---

## 11  Infrastructure / tooling

- [ ] **Distributed checkpoint format (FSDP-safe sharding)**
- [ ] **Weights-only or safetensors export**
- [x] **Config-driven experiment management** — `configs/train.yaml`
- [x] **W&B experiment tracking** — wired in `train_end_to_end.py`, project `aether-2b`
- [ ] **Docker / container definition**
- [ ] **CI/CD pipeline**

---

## Priority order — next actions

```
🔴 URGENT

  1. Monitor training — target val_loss < 3.0 (Stage 1 trigger, ~3 M steps)
  2. Standard eval harness (HumanEval + GSM8K via lm-evaluation-harness)
  3. SFT training loop

🟡 IMPORTANT — prepare before Stage 1 ctx extension

  4. Add mhc_chunk_size to TrainConfig + chunked_causal_lm_forward on model
     (see Section 10, Stage 1 — Code changes required)
  5. Re-run preprocessing without 4K truncation so long docs reach train.bin
  6. Perplexity eval script for post-hoc checkpoint comparison

🟢 LATER

  7. Stages 2 and 3 ctx extension (131K → 262K) — after Stage 1 converges
  8. CUDA tiled-prefill for training (needed at Stage 3, 262K)
  9. MQA output projection validation
  10. safetensors export
  11. Dockerfile + CI/CD
  12. Full weight quantisation (FP8 / INT8)
```
```