# DeepSeek-V4-Pro 2B Reference

This repository contains a compact PyTorch implementation of a DeepSeek-V4-Pro-style causal LM, scaled to the 2B parameter class.

## Implemented Paper Features

- Interleaved HCA/CSA hybrid attention after the first two HCA layers
- Token-level KV compression for HCA and overlapped two-branch compression for CSA
- CSA lightning indexer with top-k sparse compressed KV selection
- Shared-key-value multi-query attention, grouped output projection, partial RoPE, YaRN RoPE scaling hooks, sliding-window KV branch, and attention sink logits
- Manifold-Constrained Hyper-Connections (mHC) with dynamic A/B/C maps and Sinkhorn projection of B
- DeepSeekMoE-style shared plus routed experts with `sqrt(softplus(.))` affinity, first-layer hash routing, and sequence balance loss
- MTP depth-1 auxiliary head
- Muon optimizer with the paper's 8+2 hybrid Newton-Schulz coefficients

The source PDF was extracted into `DeepSeek_V4.txt` for auditability. The implementation is in `deepseek_v4_pro_2b/`.

---

## April 2026 Additions

### TurboQuant — PolarQuant (KV-Cache Compression)

`deepseek_v4_pro_2b/turbo_quant.py` — 8-bit and 4-bit compressed mHC KV-cache quantisation:

```python
engine = DeepSeekV4Pro2BServingEngine(model, turbo_quant_bits=8)   # 8-bit (near-lossless)
engine = DeepSeekV4Pro2BServingEngine(model, turbo_quant_bits=4)   # 4-bit (4× smaller)
```

**Quality gate** (needle-in-haystack eval across CPU and GPU contexts):
- 8-bit: KL ≈ 0 vs baseline, 100% top-1 match — **near-lossless**
- 4-bit: KL < 0.001, 100% top-1 match — **production-safe**
- GPU YaRN run (`ctx=2048/4096/8192`, 4-bit, RTX 4090): KL = `8e-6/5e-6/6e-6`, top-1 = `True/True/True`
- GPU YaRN run (`ctx=131072/262144`, 4-bit, RTX 4090): KL = `4e-6/5e-6`, top-1 = `True/True` — **✅ ALL QUALITY GATES PASSED**

```bash
# Run the full quality gate eval
python scripts/needle_eval.py --ctx-lengths 64 128 256 512
```

Memory reduction (decode phase):
- bf16: 768 MiB/layer at 262K ctx | 8-bit: 384 MiB (2×) | 4-bit: 192 MiB (4×)

### Chunked Fast-Prefill (262K+ OOM Fix)

`engine.chunked_fast_prefill(token_ids, mhc_chunk_size=4096)` bounds peak mHC state
memory from O(T×n×D) → O(C×n×D) + O(T×D):

```python
# Standard fast_prefill: OOMs at 262K+ (25.2 GB mHC peak on 2B model)
state = engine.fast_prefill(long_token_ids)

# Memory-safe alternative: 1.2 GB mHC peak at 262K context
state = engine.chunked_fast_prefill(long_token_ids, mhc_chunk_size=4096)
```

The attention sublayer still receives the full T for correct causal context.
Only the per-position mHC pre-mix / post-mix are chunked.

### YaRN RoPE Scaling

`DeepSeekV4Pro2BConfig` now carries `max_position_embeddings` plus
`rope_scaling_type`, `rope_scaling_factor`, `yarn_beta_fast`, `yarn_beta_slow`,
and `yarn_mscale`. The shared `get_rope_freqs(seq_len, config)` helper keeps the
prefill and serving decode RoPE schedule aligned, and `rope_attention_scale(config)`
applies the YaRN logit correction in HCA/CSA attention.

```bash
# Long-context needle diagnostic with YaRN enabled
python scripts/needle_eval.py --ctx-lengths 2048 4096 8192 131072 262144 --turbo-quant-bits 4 --rope-scaling yarn --device cuda
```

### Speculative Decoding

`deepseek_v4_pro_2b/speculative.py` — draft-model-based speculative decoding
(Chen et al., 2023). Identical output distribution to target-only decoding.

```python
from deepseek_v4_pro_2b.speculative import SpeculativeDecoder

decoder = SpeculativeDecoder(
    target_engine,    # large model 
    draft_engine,     # fast draft model
    draft_steps=5,
    temperature=1.0,
)
summary = decoder.generate(prompt_ids, max_new_tokens=256, eos_token_id=2)
print(f"Acceptance rate: {summary.mean_acceptance_rate:.2%}")
print(f"Effective speedup: {summary.effective_speedup:.1f}×")
# Expected: ~4–5× decode throughput at α ≈ 0.8, K = 5
```

The decoder validates that target and draft vocab sizes match before decoding.
It now also supports adaptive draft-step control during generation rounds and
shared-layer self-spec drafts (first N target layers reused as the draft model)
through the benchmark harness.
For real draft benchmarks, use `scripts/benchmark_speculative.py`:

```bash
python scripts/benchmark_speculative.py \
    --target-model 2b \
    --draft-model tiny \
    --source-tokens 65536 131072 262144 \
    --yarn-scaling \
    --output-json /tmp/spec_bench.json

# Self-spec draft (shared first N target layers) + adaptive K
python scripts/benchmark_speculative.py \
    --target-model 2b \
    --draft-model tiny \
    --self-spec-layers 8 \
    --adaptive-k \
    --min-draft-steps 3 \
    --max-draft-steps 8 \
    --source-tokens 512 8192 65536 \
    --dtype bf16 \
    --device cuda \
    --yarn-scaling \
    --output-json /tmp/spec_bench_selfspec.json
```

**Measured speculative decoding** (target=2B, draft=tiny, K=3, bf16, YaRN, RTX 4090):

| ctx_len | baseline tok/s | speculative tok/s | acceptance rate | target peak MiB | draft peak MiB |
|---------|----------------|-------------------|-----------------|-----------------|----------------|
| 512     | 10.90          | 9.36              | 96.97%          | 3 818           | 3 777          |
| 8 192   | 10.77          | 9.34              | 96.97%          | 4 318           | 3 874          |
| 65 536  | 10.74          | 9.14              | 96.97%          | 7 780           | 4 597          |
| 131 072 | 10.80          | 9.33              | 96.97%          | 11 791          | 5 423          |
| 262 144 | 10.99          | 9.15              | 96.97%          | 19 757          | 7 075          |

Raw result JSON: `artifacts/benchmark_speculative_k3.json`.

---

## Long-Context Serving

`deepseek_pipeline/serving.py` implements:

- Hybrid CSA/HCA cache-block sizing based on `lcm(m, m')`
- Disk-backed compressed-prefix reuse
- Full / periodic / zero SWA cache policies
- Restore-plan generation for replaying only the uncached tail
- Backend hook for custom CUDA sparse-attention kernels + PyTorch fallback

The CUDA extension in `deepseek_kernels/` exposes the custom serving kernels used
by this repo:

- fused sparse sink-attention operator
- tiled long-context prefill kernel
- CSA lightning indexer kernel
- HCA compression kernel
- lazy build/load path via `deepseek_kernels/loader.py`

Training-mode forward now has an opt-in tiled-prefill CUDA path in
`deepseek_v4_pro_2b/modeling.py` for HCA/CSA attention when running on CUDA
without an attention mask (`use_tiled_prefill_cuda=True` on config).

Validation and benchmark commands:

```bash
python scripts/build_cuda_kernels.py --verbose
pytest -q tests/test_cuda_runtime.py
pytest -q tests/test_incremental_serving_cuda.py
python scripts/benchmark_cuda_kernels.py --dtype fp16 --source-tokens 8192 --top-k 64
```

The exact incremental decode engine in `deepseek_v4_pro_2b/serving.py`:

- Reuses trained model weights directly
- Maintains exact HCA/CSA per-layer caches (window, partial buffer, compressed blocks)
- Supports `fast_prefill` (5–6× speedup vs step-by-step) + `chunked_fast_prefill` (OOM-safe)
- Routes sparse compressed-KV attention via PyTorch or custom CUDA backend
- Paged allocator for host↔device↔disk KV spill

Serving validation:

```bash
pytest -q tests/test_incremental_serving.py
pytest -q tests/test_turbo_quant.py
pytest -q tests/test_chunked_prefill.py
pytest -q tests/test_speculative.py
```

---

## Quick-Start

```python
from deepseek_v4_pro_2b import DeepSeekV4Pro2BConfig, DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine
from deepseek_v4_pro_2b.speculative import SpeculativeDecoder
from deepseek_pipeline.serving import PytorchAttentionBackend

config = DeepSeekV4Pro2BConfig()
model = DeepSeekV4Pro2BForCausalLM(config).eval()

# Standard prefill + decode
engine = DeepSeekV4Pro2BServingEngine(model, turbo_quant_bits=8)
state = engine.chunked_fast_prefill(prompt_ids, mhc_chunk_size=4096)
logits, state = engine.step_token(next_token, state)

# Speculative decode (2× faster draft + 2B target)
decoder = SpeculativeDecoder(engine, draft_engine, draft_steps=5, temperature=1.0)
summary = decoder.generate(prompt_ids, max_new_tokens=512)
```

The default config is intentionally conservative for a reference implementation.
See `TECHNICAL_REPORT.md` for architecture details, benchmark results, and the April 2026
sprint documentation (TurboQuant, chunked prefill, speculative decoding).

