# Technical Report

## Scope

This repository now contains a self-contained training pipeline for the DeepSeek-V4-style 2B reference model using the same dataset sources staged in the external reference repo, but with the official DeepSeek tokenizer instead of the original `cl100k_base` / SentencePiece setup.

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

python -m deepseek_pipeline.download --stage pretrain --output-root ./artifacts/datasets
python -m deepseek_pipeline.download --stage new_pretrain --output-root ./artifacts/datasets
python -m deepseek_pipeline.download --stage sft --output-root ./artifacts/datasets
python -m deepseek_pipeline.download --stage dpo --output-root ./artifacts/datasets
```

The downloader processes sources one by one in manifest order. Each source is written as resumable shards with a per-source state file, so rerunning the same command continues an interrupted download instead of restarting the whole pretrain stage from scratch.

Useful development command:

```bash
python -m deepseek_pipeline.download --stage pretrain --max-samples 100 --shard-size 100 --print-manifest
```

## Preprocessing

The preprocessing path lives in `deepseek_pipeline/preprocess.py`. Downloaded datasets are normalized into a common `text` field, then tokenized into `train.bin` and `val.bin` memmaps using the DeepSeek tokenizer. It accepts both the original direct `save_to_disk()` layout and the newer resumable shard layout, but it refuses to tokenize a source whose download state is still incomplete.

Preprocessing command:

```bash
python -m deepseek_pipeline.preprocess \
  --dataset-root ./artifacts/datasets \
  --output-dir ./artifacts/tokenized \
  --target-train-tokens 5000000 \
  --val-fraction 0.01
```

The pretrain token budget is split proportionally using the published source weights for the 15 main pretrain sources.

## DeepSeek Tokenizer

Tokenizer integration lives in `deepseek_pipeline/tokenizer.py`.

Default tokenizer:

```text
deepseek-ai/DeepSeek-V3.2
```

Integration notes:

- Loaded through `transformers.AutoTokenizer`
- `pad_token` is forced to `eos_token` when missing
- The model config is synchronized to tokenizer metadata before training:
  - `vocab_size`
  - `bos_token_id`
  - `eos_token_id`
  - `pad_token_id`
- `encode_ordinary()` is used for corpus tokenization so bulk preprocessing does not inject BOS/EOS in the middle of documents
- EOS is appended explicitly at document boundaries

Tokenizer inspection command:

```bash
python -m deepseek_pipeline.download --tokenizer-check
```

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

Key hyperparameters implemented in `train_end_to_end.py`:

- AdamW for embeddings, norms, LM head, and mHC static/gating parameters
- Muon for matrix-like hidden-layer weights
- Default LR: `2.2e-4`
- Default Muon LR: `2e-2`
- Gradient clipping: `1.0`
- Default duration: `2` epochs over the tokenized dataset
- Historical checkpoints: every `25,000` optimizer steps
- Exact resume snapshot: `checkpoint-latest.pt` is refreshed every completed step with model, optimizer, RNG, epoch, and token-offset state
- Best checkpoint: `best.pth` tracks the lowest available validation loss, or training loss when no validation split exists

Hardware notes:

- Real 2B training is intended for CUDA hardware
- The current script also runs on CPU for smoke testing with the `tiny` preset
- A practical full run should use GPUs with enough memory for the selected sequence length and batch size; the external reference setup was 2x L40S, but this repo's trainer is simpler and does not yet include their distributed kernel stack

## Code Changes

Added:

- `deepseek_pipeline/manifest.py`
- `deepseek_pipeline/tokenizer.py`
- `deepseek_pipeline/download.py`
- `deepseek_pipeline/preprocess.py`
- `deepseek_pipeline/serving.py`
- `train_end_to_end.py`
- `scripts/smoke_pipeline.py`
- `requirements.txt`

Previously fixed model/runtime issues retained:

- CSA sparse gather shape bug fixed
- Attention masking added to CSA/HCA
- Padding-aware LM and MTP losses added
- Fully ignored batches made finite
- MoE balance loss excludes padded tokens
- Runtime and optimizer regression tests added

Long-context serving additions:

- The cache layout follows the paper's heterogeneous serving design, with reusable compressed prefix blocks sized by `lcm(csa_compression, hca_compression)`
- Compressed KV entries are persisted only up to the last complete compression block
- SWA state supports the three paper-level policies:
  - full SWA caching
  - periodic SWA checkpointing
  - zero SWA caching with replay of the last `window_size * num_layers` tokens
- Restore plans are generated explicitly so a serving stack can reconstruct the exact recompute range before decode resumes
- A PyTorch fallback backend is included for correctness testing, and the interface is shaped so a custom CUDA kernel backend can be dropped in without changing cache policy code
- A real CUDA extension package now exists in `deepseek_kernels/` with:
  - C++/PyBind entrypoints in `deepseek_kernels/csrc/bindings.cpp`
  - a fused sparse sink-attention CUDA kernel in `deepseek_kernels/csrc/sparse_sink_attention_cuda.cu`
  - a lazy build/load path in `deepseek_kernels/loader.py`
  - a build entrypoint in `scripts/build_cuda_kernels.py`
- The serving layer exposes `CudaSparseAttentionBackend`, which can replace the PyTorch fallback when the extension is built successfully
- An exact incremental serving engine now exists in `deepseek_v4_pro_2b/serving.py`:
  - it performs token-by-token decode using the same trained weights as the full model
  - it maintains exact HCA and CSA layer state, including partial compression buffers and local sliding-window state
  - it restores disk-backed compressed prefix caches and replays the uncached tail before generation resumes
  - it has been checked against the full-sequence forward pass on CPU and against the PyTorch serving backend on CUDA
- The scheduler and benchmark path were tightened after validation:
  - `DecodeScheduler` now consumes prefetched logits correctly for the first generated token
  - `group_by_shared_prefix()` now hashes prompt prefixes correctly
  - `benchmark_e2e_serving.py` now exercises batched decode through the scheduler and reports prefill/decode timing separately
- The serving engine now supports allocator-backed reusable compressed prefixes:
  - `DeepSeekV4Pro2BServingEngine(..., paged_allocator=...)` pages restored compressed prefix blocks through `PagedKVAllocator`
  - this is validated by exact replay tests on CPU
  - the benchmark harness can enable it with `--allocator-device-pages` and `--allocator-host-pages`
- The serving engine now also flushes live emitted compressed tail blocks through `PagedKVAllocator`:
  - full `lcm(csa_compression, hca_compression)` token blocks are paged out as decode/prefill progresses
  - restored prefix pages and newly emitted compressed pages share the same allocator path
  - `free_state()` now releases allocator-owned pages for finished requests
- The paged allocator was corrected after end-to-end validation:
  - host-side LRU bookkeeping is now maintained correctly during device eviction
  - page-handle slot metadata is now updated on device -> host and host -> disk transitions
  - the benchmark allocator path now provides real disk spill handlers instead of assuming host RAM alone is sufficient

CUDA build command:

```bash
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
python scripts/build_cuda_kernels.py --verbose
'
```

CUDA correctness and benchmark commands:

```bash
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
pytest -q tests/test_cuda_runtime.py
pytest -q tests/test_incremental_serving_cuda.py
python scripts/benchmark_cuda_kernels.py --dtype fp16 --source-tokens 8192 --top-k 64
python scripts/benchmark_e2e_serving.py --preset tiny --source-tokens 32 --decode-tokens 4 --batch-size 2 --backend cuda --use-prefix-cache --allocator-device-pages 4 --allocator-host-pages 8 --bench-runs 1 --warmup-runs 0 --dtype fp32
'
```

CUDA build prerequisites:

- CUDA-enabled PyTorch
- matching CUDA toolkit with `nvcc`
- `CUDA_HOME` set correctly
- `ninja` installed
- repo Python dependencies installed into the same environment

Environment setup actually used for the validated build:

```bash
uv pip install --python /home/seema/miniforge3/envs/deepfill/bin/python ninja pytest
uv pip install --python /home/seema/miniforge3/envs/deepfill/bin/python -r requirements.txt
conda install -n deepfill -y -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1
```

Serving validation commands used:

```bash
pytest -q tests/test_incremental_serving.py
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
pytest -q tests/test_incremental_serving_cuda.py
'
```

## Verification Performed

Commands run in this workspace:

```bash
python -m py_compile deepseek_v4_pro_2b/*.py deepseek_pipeline/*.py deepseek_kernels/*.py tests/*.py scripts/*.py
python scripts/smoke_pipeline.py
pytest -q
python scripts/benchmark_e2e_serving.py --preset tiny --source-tokens 32 --decode-tokens 4 --batch-size 2 --backend pytorch --use-prefix-cache --allocator-device-pages 4 --allocator-host-pages 8 --bench-runs 1 --warmup-runs 0
conda run -n deepfill bash -lc '
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=/home/seema/deepseek-v4:$PYTHONPATH
pytest -q tests/test_cuda_runtime.py tests/test_incremental_serving_cuda.py
python scripts/benchmark_cuda_kernels.py --dtype fp16 --source-tokens 8192 --top-k 64
python scripts/benchmark_e2e_serving.py --preset tiny --source-tokens 32 --decode-tokens 4 --batch-size 2 --backend cuda --use-prefix-cache --allocator-device-pages 4 --allocator-host-pages 8 --bench-runs 1 --warmup-runs 0 --dtype fp32
'
```

The smoke pipeline downloads the official DeepSeek tokenizer, builds tiny synthetic reference-format datasets, tokenizes them into memmaps, trains a tiny DeepSeek-style model for two steps, and verifies that `checkpoint-latest.pt` is created.

## Benchmark Results

CUDA sparse-attention microbenchmark requested for the report:

```text
fp16, batch=1, target_tokens=128, heads=16, head_dim=128, source_tokens=8192, top_k=64
kernel: 0.076 ms
PyTorch reference: 0.116 ms
speedup: 1.53x
```

That exact result came from the earlier validated benchmark run. I reran the same command during this pass and got:

```text
fp16, batch=1, target_tokens=128, heads=16, head_dim=128, source_tokens=8192, top_k=64
kernel: 0.0759 ms
PyTorch reference: 0.1272 ms
speedup: 1.68x
```

The small difference is normal run-to-run timing variance on short GPU microbenchmarks.

Serving-path benchmark results verified in this workspace:

- CPU reference path (`backend=pytorch`, `preset=tiny`, `source_tokens=32`, `decode_tokens=4`, `batch_size=2`, allocator enabled):
  - `avg_prefill_s`: `0.4053`
  - `avg_decode_s`: `0.3885`
  - `prefill_tokens_per_s`: `157.9`
  - `decode_tokens_per_s`: `20.6`
  - `ms_per_decode_token`: `48.566`
- CUDA serving path (`backend=cuda_extension`, same tiny end-to-end setup):
  - `avg_prefill_s`: `1.0196`
  - `avg_decode_s`: `0.1103`
  - `prefill_tokens_per_s`: `62.8`
  - `decode_tokens_per_s`: `72.5`
  - `ms_per_decode_token`: `13.787`
  - `peak_hbm_mib`: `9.0`
