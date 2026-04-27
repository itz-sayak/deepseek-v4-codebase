# DeepSeek-V4-Pro 2B Reference

This repository contains a compact PyTorch implementation of a DeepSeek-V4-Pro-style causal LM, scaled to the 2B parameter class.

Implemented paper features:

- interleaved HCA/CSA hybrid attention after the first two HCA layers
- token-level KV compression for HCA and overlapped two-branch compression for CSA
- CSA lightning indexer with top-k sparse compressed KV selection
- shared-key-value multi-query attention, grouped output projection, partial RoPE, sliding-window KV branch, and attention sink logits
- Manifold-Constrained Hyper-Connections with dynamic A/B/C maps and Sinkhorn projection of B
- DeepSeekMoE-style shared plus routed experts with `sqrt(softplus(.))` affinity, first-layer hash routing, and sequence balance loss
- MTP depth 1 auxiliary head
- Muon optimizer with the paper's 8+2 hybrid Newton-Schulz coefficients

The source PDF was extracted into `DeepSeek_V4.txt` for auditability. The implementation is in `deepseek_v4_pro_2b/`.

Long-context serving support now also includes `deepseek_pipeline/serving.py`, which implements:

- hybrid CSA/HCA cache-block sizing based on `lcm(m, m')`
- disk-backed compressed-prefix reuse
- full / periodic / zero SWA cache policies
- restore-plan generation for replaying only the uncached tail
- a backend hook for custom CUDA sparse-attention kernels, plus a PyTorch correctness fallback

The CUDA extension sources live in `deepseek_kernels/` and expose a fused sparse sink-attention operator for serving:

```bash
python scripts/build_cuda_kernels.py --verbose
```

Numerical and throughput checks on a CUDA box:

```bash
pytest -q tests/test_cuda_runtime.py
python scripts/benchmark_cuda_kernels.py --dtype fp16 --source-tokens 8192 --top-k 64
```

The repository also now includes an exact incremental decode engine in `deepseek_v4_pro_2b/serving.py`. It:

- reuses the trained model weights directly
- maintains exact HCA/CSA per-layer caches
- supports full / periodic / zero SWA prefix-reuse policies
- can replay the uncached tail and resume generation from the restored state
- routes sparse compressed-KV attention through either the PyTorch reference backend or the custom CUDA backend

Serving validation commands:

```bash
pytest -q tests/test_incremental_serving.py
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
pytest -q tests/test_incremental_serving_cuda.py
'
```

Build prerequisites:

- CUDA-enabled PyTorch
- CUDA toolkit with `nvcc` on `PATH`
- `CUDA_HOME` pointing at that toolkit
- `ninja` installed for fast extension builds

```python
from deepseek_v4_pro_2b import DeepSeekV4Pro2BConfig, DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.sizing import estimate_config_parameters

config = DeepSeekV4Pro2BConfig()
print(estimate_config_parameters(config))
model = DeepSeekV4Pro2BForCausalLM(config)
print(model.estimate_total_parameters())
```

The default config is intentionally conservative for a reference implementation. The repository now includes the on-disk cache-management layer, a custom CUDA sparse-attention kernel, and an exact incremental decode loop that uses them together. I have not measured full end-to-end 1M-token throughput for the complete model stack in this workspace, so this repo should be treated as a validated serving reference rather than as a verified 1M-token benchmark result.
