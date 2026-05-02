# Aether 2B

Aether 2B is a compact PyTorch implementation of a 2B-parameter causal language model built around the architectural ideas in the DeepSeek-V4-Pro technical report. The goal is a single, self-contained reference codebase that is auditable from source PDF to running weights — no black-box dependencies, no vendored binaries. The implementation lives in `aether_2b/`, the data pipeline in `aether_pipeline/`, and the custom CUDA serving kernels in `aether_kernels/`.

---

## Architecture

The model uses a 28-layer transformer with a hybrid attention schedule. The first two layers use standard HCA (Hybrid Compression Attention), after which every layer alternates between HCA and CSA (Compressed Sink Attention). HCA compresses the key-value sequence at the layer boundary using a learned linear projection, while CSA maintains a fixed sink window plus a top-k sparse compressed buffer selected by the CSA lightning indexer. Both branches share partial RoPE embeddings and a unified sliding-window KV branch with attention sink logits.

Each layer's feedforward network is wrapped in a Manifold-Constrained Hyper-Connection (mHC) module. The mHC module maintains a learned A/B/C affinity map and applies a Sinkhorn projection to B at each step, which enforces a doubly-stochastic mixing constraint over the residual stream. The feedforward block itself is an AetherMoE mixture: a small number of shared experts are always activated, while a larger pool of routed experts compete via a `sqrt(softplus(.))` affinity score. The first layer uses hash routing instead of a learned router to break symmetry early in training. A sequence-level balance loss penalises load imbalance across the routed pool.

A pure-dense variant is also supported. Setting `dense_ffn_intermediate_size=14336` in the config replaces all MoE routing with a single SwiGLU of that width, eliminating the router linear, the expert dispatch, and the balance loss entirely. Both variants land at approximately 2.1 B total parameters with 28 layers.

Positional encoding uses RoPE with YaRN scaling. The `get_rope_freqs(seq_len, config)` helper is shared between training forward and the serving decode loop so that the frequency schedule is always consistent. The YaRN logit correction (`rope_attention_scale`) is applied inside HCA and CSA attention and has been validated at context lengths up to 262 K tokens. The model also carries a depth-1 MTP (Multi-Token Prediction) auxiliary head and uses the Muon optimizer with the paper's 8+2 hybrid Newton-Schulz coefficients.

---

## Parameter Counts

| Variant | Total params | FFN per layer | Notes |
|---------|-------------|---------------|-------|
| MoE | 2.111 B | 16 routed + 1 shared experts × 832 width | router + dispatch |
| Dense | 2.135 B | single SwiGLU, width 14336 | no routing overhead |

---

## KV-Cache Quantisation — TurboQuant / PolarQuant

`aether_2b/turbo_quant.py` implements Walsh-Hadamard pre-rotation followed by Lloyd-Max scalar quantisation of the mHC KV cache at 8-bit or 4-bit precision. The quantiser is applied to compressed KV blocks before they are stored and reversed on load. This reduces the decode-phase memory footprint by 2× at 8-bit and 4× at 4-bit with negligible quality loss.

Quality has been validated by a needle-in-haystack top-1 recall evaluation. At 8-bit the KL divergence relative to the unquantised baseline is approximately zero and top-1 match is 100% at all tested context lengths. At 4-bit the KL divergence remains below 0.001 and top-1 match is 100%. On GPU with YaRN enabled, measured KL values at contexts from 2 K to 262 K range between 4 × 10⁻⁶ and 8 × 10⁻⁶. At 262 K tokens the memory footprint of the decode KV cache falls from 768 MiB per layer (bf16) to 384 MiB at 8-bit and 192 MiB at 4-bit.

---

## Chunked Prefill — Long-Context OOM Fix

The standard `fast_prefill` path computes the full mHC pre-mix and post-mix over all T positions in one tensor operation. At T = 262 K with n = 4 hyper-connections and D = 1536 hidden size this produces a 25.2 GB intermediate tensor that exceeds 24 GB VRAM. The `chunked_fast_prefill(token_ids, mhc_chunk_size=4096)` path tiles the mHC operations over non-overlapping chunks of size C, reducing the peak mHC memory from O(T × n × D) to O(C × n × D) + O(T × D). At T = 262 K and C = 4096 the peak mHC allocation drops to 1.2 GB. The attention sublayer still receives the full sequence for correct causal context — only the per-position mHC pre-mix and post-mix are chunked.

---

## Speculative Decoding

`aether_2b/speculative.py` implements Chen et al. (2023) draft-model speculative decoding. The `SpeculativeDecoder` runs K draft steps from a small draft model, then verifies all K proposals against the target model in a single forward pass. The output distribution is provably identical to target-only sampling. Adaptive draft-step control is supported, and the benchmark harness also supports shared-layer self-spec drafts in which the first N layers of the target model are reused as the draft model.

A grid sweep over self-spec depth, sampling temperature, and draft quantisation was run on 2× RTX 4090 at 262 K context and 128 decode tokens. The baseline (target-only) throughput post optimisation is 7.41 tok/s. The best operating point found is depth = 16, temperature = 0.6, no draft quantisation, yielding 6.377 tok/s at a speedup ratio of 0.861×. Acceptance rate at depth 16 is 100% across all temperatures — every draft token is accepted. The throughput loss relative to the target-only baseline comes from the cost of running the 16-layer draft forward pass itself, not from wasted draft proposals. The best configuration is locked in `aether_2b/spec_defaults.py` and loaded in one call via `DecodeScheduler.from_self_spec_defaults(engine)`.

Self-spec grid sweep results (2× RTX 4090, bf16, YaRN, adaptive-K 1–8, 262 K context, 128 decode tokens):

| depth | temp 0.2 | temp 0.4 | temp 0.6 | temp 0.8 |
|-------|----------|----------|----------|----------|
| 12 | 1.06 tok/s (0.14×) | 1.48 (0.20×) | 1.88 (0.26×) | 2.37 (0.32×) |
| 16 | 6.35 (0.86×) | 6.33 (0.85×) | **6.38 (0.861×)** | 6.34 (0.86×) |
| 20 | 6.07 (0.82×) | 6.01 (0.81×) | 6.05 (0.82×) | 6.07 (0.82×) |
| 24 | 5.87 (0.79×) | 5.93 (0.80×) | 5.85 (0.79×) | 5.87 (0.79×) |

Depth 12 shows near-zero acceptance because a heavily truncated model does not reliably reproduce the full target's argmax token. Depth 16 hits 100% acceptance and delivers the best throughput. Depths 20 and 24 accept all tokens but pay more per draft round, so throughput decreases.

---

## Continuous-Batching Decode Scheduler

`aether_2b/scheduler.py` provides `DecodeScheduler`, a continuous-batching loop that multiplexes multiple `GenerationRequest` objects over a single `Aether2BServingEngine`. Each request carries its own draft-step budget, allowing per-request speculative decode configuration. Requests sharing a common prompt prefix are grouped so that the KV cache block for the prefix is populated once and reused. The factory method `DecodeScheduler.from_self_spec_defaults(engine)` wires the sweep-validated operating point from `spec_defaults.py` into the scheduler without any manual configuration.

---

## Long-Context Serving Infrastructure

The serving pipeline in `aether_pipeline/serving.py` implements hybrid CSA/HCA cache-block sizing based on LCM of the compression ratios, disk-backed compressed-prefix reuse via `OnDiskPrefixKVStore`, full/periodic/zero SWA cache eviction policies, and a backend hook that routes sparse compressed-KV attention through either the custom CUDA extension or a pure-PyTorch fallback. The paged KV allocator in `aether_kernels/paged_kv_allocator.py` handles host ↔ device ↔ disk KV spill.

The custom CUDA extension in `aether_kernels/` provides a fused sparse sink-attention operator, a tiled long-context prefill kernel, a CSA lightning indexer kernel, and an HCA compression kernel. The extension is JIT-compiled on first use via `torch.utils.cpp_extension.load`. The loader automatically locates `nvcc` from `CUDA_HOME/bin` and a set of standard system paths so the extension builds correctly regardless of whether `/usr/local/cuda/bin` is on the shell `PATH`.

---

## Training

`train_end_to_end.py` provides a single-script pretraining loop with FSDP multi-GPU support, linear-warmup cosine-decay LR schedule, gradient checkpointing, sequence packing, and Muon + AdamW dual-optimizer split. The script accepts a YAML config file via `--config configs/train.yaml` and supports a `variant` field to switch between the MoE and dense architecture without changing any other hyperparameters. An `ETATracker` provides rolling-window tok/s and hh:mm:ss ETA estimates in the JSON training log. Checkpointing writes only two files per interval: `checkpoint-latest.pt` (always overwritten) and `best.pth` (overwritten only when validation loss improves).

---

## Measured Throughput

CPU / fp32 / tiny model, batch = 1:

| source tokens | step-by-step prefill | fast_prefill | speedup | decode |
|--------------|---------------------|--------------|---------|--------|
| 64 | 202 tok/s | 940 tok/s | 4.6× | ~200 tok/s |
| 512 | 195 tok/s | 1070 tok/s | 5.5× | ~200 tok/s |
| 2048 | 192 tok/s | 1145 tok/s | 6.0× | ~200 tok/s |

GPU / bf16 / 2B model, single RTX 4090:

| source tokens | prefill | decode | peak HBM |
|--------------|---------|--------|----------|
| 512 | ~1651 tok/s | ~14 tok/s | 3981 MiB |
| 8192 | ~1651 tok/s | ~14 tok/s | 6140 MiB |
| 16384 | 1449 tok/s | 11.9 tok/s | 9335 MiB |

GPU / bf16 / 2B model, 2× RTX 4090, chunked_fast_prefill:

| source tokens | prefill | decode | peak HBM (GPU 0) |
|--------------|---------|--------|-----------------|
| 65536 | 1407 tok/s | 12.3 tok/s | 7418 MiB |
| 131072 | 1406 tok/s | 11.9 tok/s | 12853 MiB |
| 262144 | 1140 tok/s | 9.8 tok/s | 16941 MiB |

---

## Quick Start

```python
from aether_2b import Aether2BConfig, Aether2BForCausalLM
from aether_2b.serving import Aether2BServingEngine
from aether_2b.scheduler import DecodeScheduler, GenerationRequest

# MoE variant — 2.111 B params
config = Aether2BConfig()
model = Aether2BForCausalLM(config).eval()

# Dense variant — 2.135 B params, no routing
config_dense = Aether2BConfig.dense_2b()
model_dense = Aether2BForCausalLM(config_dense).eval()

# Serving with 4-bit KV quantisation and long-context safe prefill
engine = Aether2BServingEngine(model, turbo_quant_bits=4)
state = engine.chunked_fast_prefill(prompt_ids, mhc_chunk_size=4096)
logits, state = engine.step_token(next_token, state)

# Continuous batching with optimal self-spec defaults
scheduler = DecodeScheduler.from_self_spec_defaults(engine, max_batch_size=4)
scheduler.submit(GenerationRequest("r0", prompt_ids, max_new_tokens=256))
results = scheduler.run_until_done()
```

```bash
# Install
uv pip install -r requirements.txt

# Build CUDA kernels
python scripts/build_cuda_kernels.py --verbose

# Run full test suite (73 tests, 0 skipped)
pytest tests/ -q

# Train with YAML config
python train_end_to_end.py --config configs/train.yaml --variant moe

# Speculative decode sweep
python scripts/sweep_speculative.py \
    --source-tokens 262144 --decode-tokens 128 \
    --depths 16 20 24 --temperatures 0.2 0.4 0.6 0.8 \
    --num-gpus 2 --yarn-scaling --dtype bf16 --device cuda \
    --output-grid-json artifacts/sweep.json \
    --output-ranked-json artifacts/sweep_ranked.json
```

See `TECHNICAL_REPORT.md` for full architecture details, all benchmark result tables, and the April–May 2026 sprint documentation.
