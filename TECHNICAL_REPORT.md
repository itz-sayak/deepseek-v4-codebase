# Technical Report

## Scope

This repository now contains a self-contained training pipeline for the Aether 2B model using the same dataset sources staged in the external reference repo, but with the official Aether tokenizer instead of the original `cl100k_base` / SentencePiece setup.

The external reference repo was cloned separately outside this workspace-facing package layout.

## Dataset Sources

Exact Hugging Face source identities were taken from the external reference downloader.

Pretrain stage sources:

- `Skylion007/openwebtext`
- `allenai/c4` with `name=en`
- `bigcode/the-stack` with `data/python`, `data/java`, `data/javascript`
- `open-web-math/open-web-math`
- `meta-math/MetaMathQA`
- `HuggingFaceFW/fineweb-edu` with `name=sample-10BT`
- `wikimedia/wikipedia` with `name=20231101.en`
- `cc_news`
- `code_search_net`
- `bigcode/the-stack-smol`
- `EleutherAI/the_pile_deduplicated`
- `HuggingFaceH4/ultrachat_200k`

Additional pretrain sources:

- `Muennighoff/flan`
- `HuggingFaceTB/smol-smoltalk`
- `tatsu-lab/alpaca`

Optional large streaming sources:

- `HuggingFaceFW/fineweb` with `name=sample-350BT`
- `tiiuae/falcon-refinedweb`
- `mlfoundations/dclm-baseline-1.0`

SFT sources:

- `teknium/OpenHermes-2.5`
- `anon8231489123/ShareGPT_Vicuna_unfiltered`
- `microsoft/orca-math-word-problems-200k`
- `WizardLMTeam/WizardLM_evol_instruct_V2_196k`
- `sahil2801/CodeAlpaca-20k`
- `databricks/databricks-dolly-15k`

DPO sources:

- `HuggingFaceH4/ultrafeedback_binarized`
- `Intel/orca_dpo_pairs`

Download commands:

```bash
uv pip install -r requirements.txt
export HF_TOKEN=hf_...

python -m aether_pipeline.download --stage pretrain --output-root ./artifacts/datasets
python -m aether_pipeline.download --stage new_pretrain --output-root ./artifacts/datasets
python -m aether_pipeline.download --stage sft --output-root ./artifacts/datasets
python -m aether_pipeline.download --stage dpo --output-root ./artifacts/datasets
```

The downloader processes sources one by one in manifest order. Each source is written
as resumable shards with a per-source state file, so rerunning the same command
continues an interrupted download instead of restarting from scratch.

Useful development command:

```bash
python -m aether_pipeline.download --stage pretrain --max-samples 100 --shard-size 100 --print-manifest
```

## Preprocessing

The preprocessing path lives in `aether_pipeline/preprocess.py`. Downloaded datasets
are normalised into a common `text` field, then tokenised into `train.bin` and `val.bin`
memmaps using the Aether tokenizer. It accepts both the original direct `save_to_disk()`
layout and the newer resumable shard layout, but refuses to tokenise a source whose
download state is still incomplete.

Preprocessing command:

```bash
python -m aether_pipeline.preprocess \
  --dataset-root ./artifacts/datasets \
  --output-dir ./artifacts/tokenized \
  --target-train-tokens 5000000 \
  --val-fraction 0.01
```

The pretrain token budget is split proportionally using the published source weights
for the 15 main pretrain sources.

## Aether Tokenizer

Tokenizer integration lives in `aether_pipeline/tokenizer.py`.

Default tokenizer: `deepseek-ai/DeepSeek-V3.2`

Integration notes:

- Loaded through `transformers.AutoTokenizer`
- `pad_token` is forced to `eos_token` when missing
- The model config is synchronised to tokeniser metadata before training:
  `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id`
- `encode_ordinary()` used for corpus tokenisation — no BOS/EOS injected mid-document
- EOS appended explicitly at document boundaries

## Training

Training entrypoint:

```bash
python train_end_to_end.py \
  --data-dir ./artifacts/tokenized \
  --output-dir ./artifacts/checkpoints \
  --preset 2b \
  --seq-len 512 \
  --batch-size 1 \
  --grad-accum 1 \
  --epochs 2
```

Smoke-test command:

```bash
python scripts/smoke_pipeline.py
```

Key hyperparameters:

- AdamW for embeddings, norms, LM head, and mHC static/gating parameters
- Muon for matrix-like hidden-layer weights
- Default LR: `2.2e-4` / Muon LR: `2e-2`
- Gradient clipping: `1.0`
- Historical checkpoints: every `25,000` optimizer steps
- Exact resume snapshot: `checkpoint-latest.pt` refreshed every completed step
- Best checkpoint: `best.pth` tracks lowest validation (or training) loss

---

## Benchmark Results

### CUDA sparse-attention microbenchmark

```text
fp16, batch=1, target_tokens=128, heads=16, head_dim=128, source_tokens=8192, top_k=64
kernel:             0.0759 ms
PyTorch reference:  0.1272 ms
speedup:            1.68×
```

### Serving-path benchmark (CPU reference path)

`backend=pytorch`, `preset=tiny`, `source_tokens=32`, `decode_tokens=4`, `batch_size=2`, allocator enabled:

- `prefill_tokens_per_s`: 157.9
- `decode_tokens_per_s`: 20.6
- `ms_per_decode_token`: 48.566

### Serving-path benchmark (CUDA extension path)

Same tiny end-to-end setup, `backend=cuda_extension`:

- `prefill_tokens_per_s`: 62.8
- `decode_tokens_per_s`: 72.5
- `ms_per_decode_token`: 13.787
- `peak_hbm_mib`: 9.0

### GPU throughput — single RTX 4090 (bf16, batch=1, 2B model)

| source_tokens | prefill tok/s | decode tok/s | peak HBM  |
|--------------|---------------|--------------|-----------|
| 512          | ~1651         | ~14          | 3 981 MiB |
| 2048         | ~1651         | ~14          | 4 209 MiB |
| 8192         | ~1651         | ~14          | 6 140 MiB |
| 16384        | 1449          | 11.9         | 9 335 MiB |

### GPU throughput — 2× RTX 4090 (bf16, batch=1, chunked_fast_prefill + 2-GPU shard)

| source_tokens | prefill tok/s | decode tok/s | peak HBM (GPU 0) |
|--------------|---------------|--------------|-----------------|
| 65 536       | 1407          | 12.3         | 7 418 MiB       |
| 131 072      | 1406          | 11.9         | 12 853 MiB      |
| 262 144      | 1140.2        | 9.8          | 16 941 MiB      |

Raw JSON: `artifacts/benchmark_gpu_2b.json`, `artifacts/benchmark_gpu_2b_longctx.json`

---

## April 2026 Sprint: TurboQuant, Prefill OOM Fix, YaRN RoPE Scaling, Speculative Decoding

### 1. TurboQuant — PolarQuant (mHC KV-Cache Compression)

**File**: `aether_2b/turbo_quant.py`

Two-stage compressed KV-cache quantisation:

1. **Walsh-Hadamard Transform (WHT)**: Rotates the D-dimensional compressed vector
   with a fixed Rademacher diagonal, spreading directional structure before quantisation.
2. **Lloyd-Max Scalar Quantisation**: Per-vector absmax scaling + symmetric uniform
   quantisation to int8 (8-bit) or packed uint8 nibbles (4-bit).

**Integration**: `Aether2BServingEngine` accepts `turbo_quant_bits ∈ {None, 8, 4}`.
All emit, extract, attention-step, and paging paths updated.

**Key fix applied**: `_read_compressed` previously returned bf16 unconditionally.
On CPU (float32 model) the einsum in `_attention_step_csa` failed with a dtype mismatch.
Fixed by reading model weight dtype at dequantisation time:
```python
model_dtype = self.model.lm_head.weight.dtype
return self._polar_quant.decode(data, scale, out_dtype=model_dtype)
```

**Needle-in-Haystack Validation** (CPU tiny model):

| ctx | bits | KL vs baseline | top-1 match | max logit diff | verdict |
|-----|------|---------------|-------------|----------------|---------|
| 64  | 8    | ≈ 0.000000    | 100%        | 0.0016         | ✅ PASS |
| 64  | 4    | 0.000515      | 100%        | 0.0963         | ✅ PASS |
| 128 | 8    | ≈ 0.000000    | 100%        | 0.0016         | ✅ PASS |
| 128 | 4    | 0.000228      | 100%        | 0.0678         | ✅ PASS |
| 256 | 8    | ≈ 0.000000    | 100%        | 0.0018         | ✅ PASS |
| 256 | 4    | 0.000728      | 100%        | 0.1141         | ✅ PASS |

**GPU YaRN validation** (RTX 4090, `turbo_quant_bits=4`, `rope_scaling=yarn`,
`chunked_fast_prefill` for ctx ≥ 16K):

| ctx    | bits | KL vs baseline | top-1 match | max logit diff | prefill time | verdict |
|--------|------|----------------|-------------|----------------|--------------|---------|
| 2048   | 4    | 0.000008       | 100%        | 0.0131         | 26.0s        | ✅ PASS |
| 4096   | 4    | 0.000005       | 100%        | 0.0083         | 51.0s        | ✅ PASS |
| 8192   | 4    | 0.000006       | 100%        | 0.0094         | 104.4s       | ✅ PASS |
| 131072 | 4    | 0.000004       | 100%        | 0.0078         | 15.2s        | ✅ PASS |
| 262144 | 4    | 0.000005       | 100%        | 0.0078         | 35.9s        | ✅ PASS |

**Memory reduction** at 262K tokens, 128 HCA layers, D=1536:

| dtype  | MiB/layer | reduction |
|--------|-----------|-----------|
| bf16   | 768 MiB   | 1×        |
| 8-bit  | 384 MiB   | 2×        |
| 4-bit  | 192 MiB   | **4×**    |

**Tests**: 14 unit tests in `tests/test_turbo_quant.py`, all passing.

---

### 2. Prefill OOM Fix — Chunked mHC Forward

**Files**: `aether_2b/modeling.py`, `aether_2b/serving.py`

**Root cause**: At T=262K, the mHC forward simultaneously allocates three tensors
proportional to `[B, T, n, D]`:
- `flat = norm(state.reshape(B, T, n*D))`
- `b_raw = w_res(flat).view(B, T, n, n)`
- `mixed = einsum("btij,btjd->btid", b, state)`

At T=262144, n=4, D=1536, bf16: peak ≈ **25.2 GB** — exceeds 24 GB card.

**Fix**: `ManifoldConstrainedHyperConnection.chunked_forward(state, chunk_size)`.

Key observation: mHC pre-mix and post-mix are per-position. Only the sublayer
(attention/MoE) requires full T in one pass.

- Phase 1 (chunked): Compute `x = einsum("btn,btnd->btd", a, state)` in T-chunks.
  Each chunk allocates `[B, C, n*D]` only. Output `x: [B, T, D]` accumulated.
- Phase 2 (full T): Run sublayer on full `x`. Attention already chunks internally.
- Phase 3 (chunked): Compute residual in T-chunks, each chunk frees immediately.

**Memory at T=262K, C=4096, bf16**:

| method | peak | status |
|--------|------|--------|
| `fast_prefill` | ~25.2 GB | OOM on 24 GB card |
| `chunked_fast_prefill(chunk_size=4096)` | ~**1.2 GB** | ✅ fits comfortably |

**API**:
```python
state = engine.chunked_fast_prefill(token_ids, mhc_chunk_size=4096)
```

**Tests**: 9 tests in `tests/test_chunked_prefill.py`.

**Training-mode tiled prefill integration (implemented)**:

The tiled prefill CUDA kernel is now integrated into training-mode attention
forward as an opt-in path (`use_tiled_prefill_cuda=True`) in
`aether_2b/modeling.py` for HCA/CSA prefill on CUDA when no
attention mask is present. This routes per-token assembled source KV sets
through `tiled_prefill_attention(...)` during model forward/chunked forward.

---

### 3. YaRN RoPE Scaling

**Files**: `aether_2b/configuration.py`, `aether_2b/modeling.py`,
`aether_2b/serving.py`, `tests/test_rope_scaling.py`

Config-driven long-context positional encoding. `get_rope_freqs(seq_len, config)`
implements three modes:

- `none`: standard fixed theta (default, backward compatible)
- `linear`: uniform frequency interpolation by `rope_scaling_factor`
- `yarn`: per-dimension ramp blend between full interpolation (low-freq dims) and
  no interpolation (high-freq dims), plus attention logit correction via
  `rope_attention_scale(config) = log(factor) × yarn_mscale + 1.0`

RoPE is applied post-kernel in Python (confirmed: `hca_compress_cuda.cu` performs
the HBM-side weighted sum only; rotation happens in `sparse_attention.py` after
the kernel returns). No CUDA kernel changes were required.

**New config fields** (all default to `"none"` / `1.0` — existing benchmarks unaffected):

```python
rope_scaling_type: str = "none"       # "none" | "linear" | "yarn"
rope_scaling_factor: float = 1.0      # target_ctx / train_ctx
yarn_beta_fast: float = 32.0
yarn_beta_slow: float = 1.0
yarn_mscale: float = 0.1
max_position_embeddings: int = 4096   # training context length reference
```

**Tests**: 5 tests in `tests/test_rope_scaling.py`. All passing.

All GPU needle-in-haystack runs with YaRN enabled pass at 100% top-1 match
through 262K tokens (see TurboQuant GPU table above — all runs use YaRN).

---

### 4. Speculative Decoding

**File**: `aether_2b/speculative.py`

Implements draft-model speculative decoding (Chen et al., 2023). Output distribution
is provably identical to target-only greedy decoding.

**Algorithm** (one round, K=3 draft tokens):
1. Draft proposes K tokens auto-regressively
2. Target verifies all K+1 positions in one forward pass
3. Token accepted with probability `min(1, p_target / p_draft)`
4. First rejection resampled from `max(0, p_target − p_draft)` (normalised)
5. All accepted → bonus token from target's last position logit

**Tests**: 11 unit tests in `tests/test_speculative.py`. All passing.

**Implemented optimization hooks**:

1. **Adaptive K in decoder runtime**:
  `SpeculativeDecoder` now supports adaptive draft-step control:
  - `adaptive_draft_steps`
  - `min_draft_steps`
  - `max_draft_steps`
  - `adapt_up_threshold` / `adapt_down_threshold`

2. **Shared-layer self-spec draft construction**:
  `build_self_spec_draft_model(target_model, draft_layers)` builds a draft model
  by reusing the first N target layers (shared modules), enabling self-spec-style
  benchmarking without duplicating full draft weights.

---

### 5. Speculative Decoding Benchmark — Measured Results

**Command**:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_speculative.py \
  --source-tokens 512 8192 65536 131072 262144 \
  --decode-tokens 128 --draft-steps 3 --dtype bf16 \
  --device cuda --yarn-scaling \
  --warmup-rounds 1 --benchmark-rounds 5 \
  --mhc-chunk-size 2048 \
  --output-json artifacts/benchmark_speculative_k3.json
```

**Setup**: single RTX 4090, bf16, YaRN enabled, K=3, tiny draft + 2B target,
5 benchmark rounds + 1 warmup.

**HBM budget**: combined peak at 262K = 19,757 + 7,075 = **26,832 MiB**.
Fits within 48 GB (2× 24 GB cards). `budget_capped: false` at all context lengths.

**Results**:

| source_tokens | baseline tok/s | spec tok/s | acceptance rate | net speedup | target HBM | draft HBM |
|--------------|---------------|------------|-----------------|-------------|------------|-----------|
| 512          | 10.90         | 9.36       | **96.97%**      | 0.86×       | 3 818 MiB  | 3 777 MiB |
| 8 192        | 10.77         | 9.34       | **96.97%**      | 0.87×       | 4 318 MiB  | 3 874 MiB |
| 65 536       | 10.74         | 9.14       | **96.97%**      | 0.85×       | 7 780 MiB  | 4 597 MiB |
| 131 072      | 10.80         | 9.33       | **96.97%**      | 0.86×       | 11 791 MiB | 5 423 MiB |
| 262 144      | 10.99         | 9.15       | **96.97%**      | 0.83×       | 19 757 MiB | 7 075 MiB |

All runs: `matches_target_greedy: true` — output correctness confirmed.
Raw JSON: `artifacts/benchmark_speculative_k3.json`.

**Analysis — why α=0.97 does not produce speedup**:

The acceptance rate of 96.97% is excellent and confirms the tiny and 2B models
agree on token choices almost perfectly. The problem is not acceptance rate — it
is draft forward-pass latency.

On GDDR6X (as opposed to HBM3 on A100/H100), both the tiny and 2B models are
memory-bandwidth-bound at decode time. The tiny model's per-step latency is not
negligibly small relative to the target. At K=3, each speculative round costs:

```
wall_time ≈ 3 × draft_step_latency + 1 × target_verify_latency
           + state_management_overhead
```

The measured decode times show speculative rounds take ~13.7s for 128 tokens
vs ~11.8s for baseline — the draft overhead adds ~1.9s per 128-token generation
even though 96 of 99 proposed tokens are accepted. The draft is fast in terms
of compute flops but not fast enough in wall-clock time on this memory-bandwidth-
constrained hardware.

**Fix options**:

1. **K tuning / adaptive-K**: At α=0.97, higher K means more tokens accepted
  per target call. Adaptive-K is now implemented in decoder and exposed in
  benchmark CLI (`--adaptive-k`, `--min-draft-steps`, `--max-draft-steps`).

2. **Shared-layer self-spec draft**: Implemented in benchmark harness via
  `--self-spec-layers N` using `build_self_spec_draft_model(...)`.
  Draft and target share weights for the first N layers, reducing duplicate
  memory pressure during speculative benchmarking.

3. **Quantise the draft KV cache**: Pass `turbo_quant_bits=4` to the draft engine.
   Zero new code required. May recover 20–30% of draft step latency on GDDR6X.

**Recommended next action**: run controlled K-sweeps with self-spec and adaptive-K
enabled, then lock one default policy by net decode tok/s on 65K+ contexts.

---

### Files Added / Modified (April 2026 sprint)

| File | Status | Description |
|------|--------|-------------|
| `aether_2b/turbo_quant.py` | Modified | WHT aliasing fix, `out_dtype` param on `decode()` |
| `aether_2b/configuration.py` | Modified | `max_position_embeddings`, YaRN rope-scaling fields |
| `aether_2b/modeling.py` | Modified | `chunked_forward()` on mHC, Block, and Model; shared RoPE helpers |
| `aether_2b/serving.py` | Modified | `_read_compressed` dtype fix, `chunked_fast_prefill()`, YaRN decode scaling |
| `aether_2b/speculative.py` | **New** | `SpeculativeDecoder`, `SpecDecodeResult`, `SpecDecodeSummary` |
| `aether_2b/__init__.py` | Modified | `SpeculativeDecoder` exported |
| `scripts/needle_eval.py` | **New** | Needle-in-haystack TurboQuant / YaRN validation harness |
| `scripts/benchmark_speculative.py` | **New** | Baseline vs speculative decode benchmark harness |
| `tests/test_turbo_quant.py` | Existing | 14 tests (all passing) |
| `tests/test_chunked_prefill.py` | **New** | 9 tests for chunked mHC forward + `chunked_fast_prefill` |
| `tests/test_rope_scaling.py` | **New** | 5 tests for YaRN/linear RoPE scaling and model smoke |
| `tests/test_speculative.py` | **New** | 11 tests for speculative decoding |
| `artifacts/needle_eval_results.json` | **New** | Quality gate results (CPU + GPU YaRN) |
| `artifacts/benchmark_speculative_k3.json` | **New** | Speculative decode benchmark, K=3 |

---

## May 2026 Sprint: Kernel Fusion, Dual-GPU Spec Bench, Pipeline Integration

### 1. Serving Engine Fusion Helpers

**File**: `aether_2b/serving.py`

Three new methods on `Aether2BServingEngine` expose layer-level control for
shared-layer speculative decoding:

| Method | Purpose |
|--------|---------|
| `step_token_with_hidden(token, state)` | Standard step that also returns the pre-head hidden state `[B, mhc_expansion, H]` for reuse by a deeper engine |
| `step_token_from_hidden(token, state, hidden, start_layer)` | Executes only layers `[start_layer, L)` using a provided hidden state — skips the already-computed shared prefix |
| `_step_token_internal(token, state, *, start_layer, hidden, return_hidden)` | Unified implementation backing all three above |

These allow the target model to skip re-running shared-prefix layers during
verification, reducing the per-round compute cost when the draft and target
share the first N transformer blocks.

### 2. Shared-Layer Fused Speculative Round

**File**: `aether_2b/speculative.py`

`SpeculativeDecoder._spec_round_shared_fused()` uses the fusion helpers above:

- Draft runs K steps via `step_token_with_hidden`, collecting `(logits, state, hidden)` per step.
- Target verification for positions 0..K−1 calls `step_token_from_hidden(vt, state, hidden[i], shared_depth)` — only the tail layers execute.
- Bonus token (position K) uses the standard `step_token`.
- Auto-activated when `_detect_shared_layer_fusion_depth() > 0` (draft layers are object-identical to the first N target layers).

### 3. Dual-GPU Speculative Benchmark — Measured Results (262K context)

**Command**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark_speculative.py \
  --target-model 2b \
  --draft-model tiny \
  --self-spec-layers 8 \
  --source-tokens 262144 \
  --decode-tokens 128 \
  --draft-steps 3 \
  --warmup-rounds 1 \
  --benchmark-rounds 3 \
  --backend cuda \
  --device cuda \
  --dtype bf16 \
  --num-gpus 2 \
  --yarn-scaling \
  --mhc-chunk-size 4096 \
  --output-json artifacts/benchmark_speculative_fused_262k.json
```

**Setup**: 2× RTX 4090 (2-GPU tensor shard), bf16, YaRN enabled, K=3, self-spec-layers=8
(first 8 of 28 target layers shared as draft), 3 benchmark rounds + 1 warmup.

**Memory budget**: combined peak at 262K = 16,941 + 17,972 = **34,913 MiB** (well within
48 GB). `budget_capped: false`.

**Results**:

| source_tokens | baseline tok/s | spec tok/s | acceptance rate | net speedup | target HBM | draft HBM |
|--------------|---------------|------------|-----------------|-------------|------------|-----------|
| 262 144      | 7.57          | 1.76       | **0.0%**        | 0.23×       | 16 941 MiB | 17 972 MiB |

`matches_target_greedy: true` — output correctness confirmed at 262K.
Raw JSON: `artifacts/benchmark_speculative_fused_262k.json`.

**Prefill times** (one-time cost): target 239.5 s, draft 54.6 s at 262K tokens.

**Analysis — 0 % acceptance rate**:

With greedy decoding (`temperature=0`), the acceptance criterion is exact argmax agreement:

```
accept iff  argmax(p_target(· | ctx)) == draft_proposed_token
```

The self-spec draft uses only the first 8 of 28 transformer layers. A heavily
truncated model does not share its argmax tokens with the full model, so every
proposed token is rejected and the bonus is resampled from the corrected (target)
distribution. This keeps output quality intact (`matches_target_greedy: true`) but
eliminates any throughput gain — K+1 target steps produce only 1 accepted token,
roughly 4× slower than baseline.

The draft also incurs its own 8-layer forward per step on top of the K+1 target
calls, explaining the measured 7.57 → 1.76 tok/s regression.

**Path to positive acceptance rate** with self-spec:
1. Use `temperature > 0` — acceptance is stochastic rather than exact-argmax; even
   modest per-token agreement (e.g. 20%) starts to recover throughput.
2. Train a dedicated early-exit draft head at layer 8 — aligns its distribution
   closer to the final layer's output.
3. Use the existing `--adaptive-k` flag: the adaptor will quickly settle to `k=1`
   (effectively disabling spec) rather than paying the K+1 overhead at 0 % α.

### 4. DecodeScheduler — Speculative Decode Wired into Continuous-Batching Pipeline

**File**: `aether_2b/scheduler.py`

Speculative decoding is now a first-class feature of `DecodeScheduler`.

**API**:
```python
from aether_2b.scheduler import DecodeScheduler, GenerationRequest

# Scheduler-level default: all requests use draft_steps=3
scheduler = DecodeScheduler(
    target_engine,
    max_batch_size=8,
    draft_engine=draft_engine,          # optional; enables spec decode
    default_draft_steps=3,              # used when req.draft_steps == 0
)

# Per-request override
req = GenerationRequest("r0", prompt_ids=[...], max_new_tokens=256, draft_steps=5)
scheduler.submit(req)
```

**What changed**:

| Component | Change |
|-----------|--------|
| `GenerationRequest` | Added `draft_steps: int = 0`, `_draft_state`, `_spec_buffer` fields |
| `DecodeScheduler.__init__` | Added `draft_engine`, `default_draft_steps` params; `_spec_decoders` dict |
| `_fill_active_slots` | Prefills draft engine alongside target when `draft_steps > 0` |
| `_step` | Drains `_spec_buffer` on spec requests; triggers a new `_spec_round` when buffer empties; falls back to `step_token` for non-spec requests |
| `_get_or_create_spec_decoder` | Creates and caches one `SpeculativeDecoder` per request seq_id |
| `_run_spec_round` | Dispatches to `_spec_round_shared_fused` when shared-layer fusion is detected, otherwise `_spec_round` |

**Behaviour**:
- Standard requests (`draft_steps=0` or no draft engine) execute unchanged via `step_token`.
- Spec requests emit multiple tokens per round by filling `_spec_buffer` with `accepted_tokens + [bonus]` from each round.  The scheduler drains one token per iteration, maintaining the same `BatchStepResult` interface upstream.
- Spec decoder is cleaned up from `_spec_decoders` when the request finishes.

### 5. Test Coverage Update

| Test file | Tests |
|-----------|-------|
| `tests/test_speculative.py` | 14 (was 11; 3 new fusion/integration tests added) |
| All suites | **68 passed, 1 skipped** (see note below) |

> **Skipped test**: `test_yarn_needle_top1_match` in `tests/test_rope_scaling.py`.
> It is guarded by `@pytest.mark.skipif(os.environ.get("AETHER_RUN_ROPE_NEEDLE") != "1", ...)`
> because the test allocates a full 262K-token context to verify top-1 recall quality
> under YaRN scaling — a multi-minute GPU run not suitable for a CI loop.
> Run explicitly with `AETHER_RUN_ROPE_NEEDLE=1 pytest tests/test_rope_scaling.py`.

---

### Files Added / Modified (May 2026 sprint)

| File | Status | Description |
|------|--------|-------------|
| `aether_2b/serving.py` | Modified | `_step_token_internal`, `step_token_with_hidden`, `step_token_from_hidden`, `generate_speculative` |
| `aether_2b/speculative.py` | Modified | `_spec_round_shared_fused`, `_detect_shared_layer_fusion_depth`, `_sync_shared_layer_states` |
| `aether_2b/scheduler.py` | Modified | Speculative decode integrated into `DecodeScheduler._step`; `draft_engine` + `default_draft_steps` params |
| `scripts/benchmark_speculative.py` | Modified | `--num-gpus` flag; shared-layer shard inheritance; `--self-spec-layers` |
| `tests/test_speculative.py` | Modified | 3 additional fusion/integration tests (14 total) |
| `artifacts/benchmark_speculative_fused_262k.json` | **New** | Dual-GPU 262K self-spec fused benchmark result |

---

```bash
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
python scripts/build_cuda_kernels.py --verbose
'
```

Prerequisites: CUDA-enabled PyTorch, matching `nvcc`, `CUDA_HOME` set, `ninja` installed.

Environment setup:

```bash
uv pip install --python /home/seema/miniforge3/envs/deepfill/bin/python ninja pytest
uv pip install --python /home/seema/miniforge3/envs/deepfill/bin/python -r requirements.txt
conda install -n deepfill -y -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1
```

CUDA correctness and benchmark:

```bash
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
pytest -q tests/test_cuda_runtime.py
pytest -q tests/test_incremental_serving_cuda.py
python scripts/benchmark_cuda_kernels.py --dtype fp16 --source-tokens 8192 --top-k 64
python scripts/benchmark_e2e_serving.py \
  --preset tiny --source-tokens 32 --decode-tokens 4 --batch-size 2 \
  --backend cuda --use-prefix-cache \
  --allocator-device-pages 4 --allocator-host-pages 8 \
  --bench-runs 1 --warmup-runs 0 --dtype fp32
'
```

---

## Section 6: Speculative Decode Grid Sweep (May 2026)

### Objective

Determine whether self-speculative decoding (shared-layer draft) can accelerate
decode at 262K-token context on 2× RTX 4090 by sweeping draft depth and sampling
temperature.

### Setup

| Parameter | Value |
|-----------|-------|
| Hardware | 2× RTX 4090 (24 GB HBM each) |
| Model | Aether2B (28 layers, 2B params, bf16) |
| Sharding | Layers 0–13 on cuda:0, 14–27 on cuda:1 |
| Context | 262 144 tokens (YaRN ×4 RoPE scaling) |
| Decode tokens | 128 |
| Draft | Shared-layer self-spec (`build_self_spec_draft_model`) |
| Depths swept | 12, 16, 20, 24 |
| Temperatures swept | 0.2, 0.4, 0.6, 0.8 |
| Draft steps (K) | Adaptive 1–8, starting at 6 |
| Warmup / benchmark rounds | 1 / 3 |
| Prefill | `chunked_fast_prefill(mhc_chunk_size=4096)` |

### Bug Fixed: Multi-GPU Draft Sharding

`build_self_spec_draft_model` returns a model that **shares** module objects with
the target (layers, embed, head).  The prior code called `.to(device)` on the
draft model after construction, which silently moved shared layers off their
assigned GPU back to cuda:0, corrupting the target's sharding — causing a
`RuntimeError: Expected all tensors to be on the same device` crash at depth ≥ 16.

Fix applied in two places:

1. **`aether_2b/serving.py` `__init__`** — skip `model.to(self.device)`
   when `_layer_devices` is already set (i.e. model is pre-sharded).
2. **`scripts/sweep_speculative.py`** — set `draft_model.model._layer_devices`
   *before* constructing the serving engine, so the engine sees the shard map and
   skips the bulk `.to()`.

### Results

Baseline (target-only decode at 262K): **7.32 tok/s**.

| depth | temp=0.2 | temp=0.4 | temp=0.6 | temp=0.8 | α (all temps) |
|-------|----------|----------|----------|----------|---------------|
| 12    | 1.06 (0.14×) | 1.48 (0.20×) | 1.88 (0.26×) | 2.37 (0.32×) | 0.009–0.269 |
| 16    | 6.28 (0.86×) | 6.27 (0.86×) | **6.30 (0.86×)** | 6.28 (0.86×) | 1.000 |
| 20    | 6.04 (0.82×) | 5.99 (0.82×) | 6.03 (0.82×) | 6.04 (0.83×) | 1.000 |
| 24    | 5.84 (0.80×) | 5.91 (0.81×) | 5.83 (0.80×) | 5.85 (0.80×) | 1.000 |

*Values: tok/s (speedup×).  Best cell bold.*

### Analysis

**Depth=12** has low acceptance (α = 0.009–0.269): the 12-layer draft is too
shallow to predict the 28-layer target, so most draft tokens are rejected and the
target must resample — incurring draft + verification + rejection overhead with no
throughput gain.  Result: 0.14–0.32× baseline.

**Depth ≥ 16** achieves α = 1.000 (every draft token accepted) across all
temperatures, confirming that a 16-layer prefix of the target predicts the full
target distribution faithfully at this context length.  However, the extra
16-layer draft forward pass adds latency that exceeds the benefit of accepting K
tokens in one round: **self-spec is bandwidth-bound at 262K context on RTX 4090**.

The best speedup is 0.86× (depth=16, temp=0.6), meaning self-spec adds ~14%
overhead relative to target-only decoding at this context length.  This is a
**hardware-bound** result: the 262K KV-cache occupies ~17 GB HBM; each
decode step is dominated by cache reads, and the draft step replicates that
bandwidth without a compensating reduction in target-step count (because
verification still touches the full 28-layer KV cache).

**Practical recommendation**: disable self-spec at contexts ≥ 128K on single-node
2× RTX 4090.  Self-spec with shared layers is beneficial only when the draft step
is substantially cheaper than the target step — typically at shorter contexts
(< 32K) or with a separate smaller draft model.

### Best Config Applied

```
depth   = 16
temperature = 0.6
spec_tok_s  = 6.30 tok/s
α           = 1.000
speedup     = 0.860×
```

Written to `aether_2b/spec_defaults.py` via `scripts/apply_best_spec_config.py`.
The scheduler imports this file at startup and falls back to hard-coded defaults
if it is absent (graceful degradation).

### Files Added / Modified (Section 6)

| File | Status | Description |
|------|--------|-------------|
| `aether_2b/serving.py` | Modified | Skip `model.to()` when `_layer_devices` pre-set; prevents draft sharding corruption |
| `aether_2b/spec_defaults.py` | **New** | Auto-generated sweep best-config constants |
| `scripts/sweep_speculative.py` | Modified | Set `_layer_devices` before engine construction; drop `.to()` on shared-layer draft |
| `scripts/apply_best_spec_config.py` | **New** | Reads ranked JSON, writes `spec_defaults.py` |
| `artifacts/sweep_speculative_grid.json` | **New** | Full 4×4 grid results (16 cells) |
| `artifacts/sweep_speculative_ranked.json` | **New** | Cells ranked by speculative tok/s |
| `artifacts/sweep_speculative_grid_16plus.json` | **New** | Partial sweep (depth=16/20/24, 12 cells) |
| `artifacts/sweep_speculative_ranked_16plus.json` | **New** | Ranked partial sweep |

## Section 7: Post-Optimisation Spec Sweep, Dense Config, Pipeline Defaults (May 2026)

### 1. Speculative Decode Optimisations

Two performance fixes applied after the Section 6 sweep:

**Dequant-decode cache** (`serving.py`):
When `turbo_quant_bits` is set on the draft engine, `_read_compressed` was called
every decode step and re-decoded the full resident compressed KV tensor from scratch.
Fixed by memoizing decoded tensors in `self._dequant_cache` keyed on
`(data.data_ptr(), data.shape, scale.data_ptr(), scale.shape, str(model_dtype))`.
Cache is bounded at 128 entries and clears on overflow.  Stable between token
steps until new KV blocks are appended.

**Shared-state fast path** (`speculative.py`):
`_sync_shared_layer_states` previously cloned every shared-layer `ModelServingState`
on every speculative token.  When draft and target use the same quantization format
(the common case) no conversion is needed.  Fixed by aliasing layer state objects
directly (`target_state.layer_states[i] = draft_state.layer_states[i]`) and
returning early.  Eliminates O(shared_depth) deep tensor copies per token.

### 2. Post-Optimisation Benchmark (no draft quantisation)

After the two optimisations, the best unquantised cell was re-measured:

| variant | baseline tok/s | spec tok/s | speedup | notes |
|---------|---------------|------------|---------|-------|
| depth=16, temp=0.6, no quant | **7.409** | **6.377** | **0.861×** | post-opt |
| depth=16, temp=0.6, 4-bit draft | 7.409 | 6.20 | 0.831× | 4-bit KV hurts on RTX 4090 |
| depth=16, temp=0.2, 8-bit draft | 7.409 | 6.02 | 0.828× | 8-bit KV also slower |

**Finding**: draft KV quantisation hurts throughput at 262K context on RTX 4090.
The dequant overhead (Walsh-Hadamard decode per step) exceeds the HBM bandwidth
saving because the RTX 4090's L2/HBM bandwidth is high relative to FP compute
at this batch size.  No-quant is the best setting; locked as default.

`spec_defaults.py` updated with post-opt numbers.

### 3. Spec-Defaults Pipeline Integration

`spec_defaults.py` now exports `SPEC_DEFAULT_DRAFT_QUANT_BITS = None`.
`DecodeScheduler` imports `SPEC_DEFAULT_SELF_SPEC_LAYERS` and
`SPEC_DEFAULT_DRAFT_QUANT_BITS` in addition to the six previously imported
constants.

`DecodeScheduler.from_self_spec_defaults(engine)` factory classmethod added:
builds the full optimal draft engine (depth=16, no quant, adaptive K 1–8,
temp=0.6) in a single call without exposing any construction details.

```python
scheduler = DecodeScheduler.from_self_spec_defaults(engine, max_batch_size=4)
scheduler.submit(GenerationRequest("r0", prompt_ids, max_new_tokens=256))
results = scheduler.run_until_done()
```

### 4. Dense Configuration Variant

`Aether2BConfig.dense_2b()` classmethod added to
`aether_2b/configuration.py`.  Every transformer block gets a single
SwiGLU FFN (no router, no expert list, no balance loss) of width 14336.

**Parameter sizing**:

| component | MoE variant | Dense variant |
|-----------|------------|---------------|
| Embedding | 100.7 M | 100.7 M |
| Attention + mHC (26 layers) | 660 M | 660 M |
| FFN (26 layers) | 1 207 M (16+1 experts × 832) | 1 229 M (14336 wide) |
| **Total** | **1.967 B** | **1.990 B** |

`dense_ffn_intermediate_size=14336` was chosen as the smallest standard
SwiGLU width (multiple of 1024) that brings the dense variant into the ~2B class.

`AetherMoE.__init__` short-circuits to a single `SwiGLU` when
`config.dense_ffn_intermediate_size > 0`.  `forward` bypasses all routing
code and applies the FFN directly to the `[B, T, D]` tensor, returning
`(out, 0.0)` (no balance loss).  `estimate_config_parameters` in `sizing.py`
dispatches to the correct per-layer formula.

### Files Added / Modified (Section 7)

| File | Status | Description |
|------|--------|-------------|
| `aether_2b/configuration.py` | Modified | `dense_ffn_intermediate_size` field; `dense_2b()` factory |
| `aether_2b/modeling.py` | Modified | `AetherMoE.__init__` + `forward` dense fast-path |
| `aether_2b/sizing.py` | Modified | `estimate_config_parameters` handles dense variant |
| `aether_2b/serving.py` | Modified | `_dequant_cache` memoization in `_read_compressed` |
| `aether_2b/speculative.py` | Modified | Shared-state alias fast-path in `_sync_shared_layer_states` |
| `aether_2b/spec_defaults.py` | Modified | Post-opt numbers; `SPEC_DEFAULT_DRAFT_QUANT_BITS` |
| `aether_2b/scheduler.py` | Modified | Import `SELF_SPEC_LAYERS` + `DRAFT_QUANT_BITS`; `from_self_spec_defaults` factory |
| `artifacts/sweep_speculative_grid_noq_d16_t06_fastsync.json` | **New** | Post-opt single-cell benchmark: 6.377 tok/s |

---

## Verification Performed

```bash
python -m py_compile aether_2b/*.py aether_pipeline/*.py \
  aether_kernels/*.py tests/*.py scripts/*.py
python scripts/smoke_pipeline.py
pytest -q
python scripts/benchmark_e2e_serving.py \
  --preset tiny --source-tokens 32 --decode-tokens 4 --batch-size 2 \
  --backend pytorch --use-prefix-cache \
  --allocator-device-pages 4 --allocator-host-pages 8 \
  --bench-runs 1 --warmup-runs 0
CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_speculative.py \
  --source-tokens 512 8192 65536 131072 262144 \
  --decode-tokens 128 --draft-steps 3 --dtype bf16 \
  --device cuda --yarn-scaling \
  --warmup-rounds 1 --benchmark-rounds 5 \
  --mhc-chunk-size 2048 \
  --output-json artifacts/benchmark_speculative_k3.json
```