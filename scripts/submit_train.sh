#!/bin/bash
#SBATCH --job-name=aether-2b-pretrain
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sayak.dutta/Aether/artifacts/logs/train_screen.log
#SBATCH --error=/home/sayak.dutta/Aether/artifacts/logs/train_screen.err

set -eo pipefail   # no -u: conda activate scripts reference unbound vars

# ── Environment ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aether

cd /home/sayak.dutta/Aether/aether-2B

# ── GPU pinning: use GPU 0+1 (same NUMA node, NODE topology) not GPU 1+2 (cross-NUMA SYS) ──
export CUDA_VISIBLE_DEVICES=0,1

# ── NCCL tuning for single-node ────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
# Do NOT set NCCL_SOCKET_IFNAME=lo — that forces slow TCP loopback.
# Unset lets NCCL auto-select CUDA P2P / SHM for same-NUMA PCIe GPUs.
export OMP_NUM_THREADS=4

echo "[train] Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "[train] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"

# ── Launch ─────────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node=2 \
    train_end_to_end.py \
    --config configs/train.yaml

echo "[train] Job finished at $(date)"

# Auto-resubmit so training continues across 24h walltime slices.
# The trainer will auto-resume from checkpoint-latest.pt.
echo "[train] Resubmitting for next slice..."
sbatch "$(realpath "$0")"
