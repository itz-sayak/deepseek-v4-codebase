# Command Runbook (Fresh Start, No Data Downloaded)

This file lists the command sequence to bootstrap this repo from scratch.

## 0) Prerequisites

- Python environment available (recommended: conda env `deepfill`)
- `uv` installed
- Hugging Face access token with dataset permission

## 1) Install Dependencies

```bash
cd /home/seema/deepseek-v4
conda activate deepfill
uv pip install -r requirements.txt
```

## 2) Set Hugging Face Auth

```bash
export HF_TOKEN=hf_your_token_here
```

Optional check:

```bash
python -m deepseek_pipeline.download --tokenizer-check
```

## 3) Download Datasets (All Stages)

If this is your first full run, execute all four stages:

```bash
python -m deepseek_pipeline.download --stage pretrain --output-root ./artifacts/datasets
python -m deepseek_pipeline.download --stage new_pretrain --output-root ./artifacts/datasets
python -m deepseek_pipeline.download --stage sft --output-root ./artifacts/datasets
python -m deepseek_pipeline.download --stage dpo --output-root ./artifacts/datasets
```

Small dry-run (quick sanity check):

```bash
python -m deepseek_pipeline.download --stage pretrain --max-samples 100 --shard-size 100 --print-manifest
```

## 4) Preprocess to Tokenized Binaries

```bash
python -m deepseek_pipeline.preprocess \
  --dataset-root ./artifacts/datasets \
  --output-dir ./artifacts/tokenized \
  --target-train-tokens 5000000 \
  --val-fraction 0.01
```

## 5) Smoke Test the Pipeline

```bash
python scripts/smoke_pipeline.py
```

## 6) Train (2B preset)

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

## 7) Optional: Distributed Training (FSDP)

Run with `torchrun` when using multiple GPUs:

```bash
torchrun --nproc_per_node=2 train_end_to_end.py \
  --data-dir ./artifacts/tokenized \
  --output-dir ./artifacts/checkpoints \
  --preset 2b \
  --seq-len 512 \
  --batch-size 1 \
  --grad-accum 1 \
  --epochs 2 \
  --fsdp \
  --gradient-checkpointing
```

## 8) Optional: Build and Validate CUDA Kernels

If your environment does not already have the CUDA toolkit packages used by the
validated build, install them into `deepfill` first and point `CUDA_HOME` at the
active conda prefix:

```bash
conda install -n deepfill -y -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1
export CUDA_HOME=$CONDA_PREFIX
```

```bash
python scripts/build_cuda_kernels.py --verbose
pytest -q tests/test_cuda_runtime.py
pytest -q tests/test_incremental_serving_cuda.py
python scripts/benchmark_cuda_kernels.py --dtype fp16 --source-tokens 8192 --top-k 64
```

## 9) Optional: Serving/Long-Context Validations

```bash
pytest -q tests/test_incremental_serving.py
pytest -q tests/test_turbo_quant.py
pytest -q tests/test_chunked_prefill.py
pytest -q tests/test_speculative.py
python scripts/needle_eval.py --ctx-lengths 64 128 256 512
```

## 10) Optional: End-to-End Serving Benchmark

```bash
python scripts/benchmark_e2e_serving.py \
  --preset 2b \
  --source-tokens 8192 65536 131072 \
  --decode-tokens 128 \
  --batch-size 1 \
  --dtype bf16 \
  --backend cuda \
  --fast-prefill \
  --swa-offload \
  --num-gpus 2
```

## Notes

- Downloads are resumable per source; rerunning the same stage continues unfinished shards.
- Training checkpoints are written to `./artifacts/checkpoints` (`checkpoint-latest.pt`, periodic step snapshots, `best.pth`).
- If CUDA is unavailable in your Python runtime, CUDA benchmark and kernel commands will fail; run CPU smoke/tests instead.