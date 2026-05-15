#!/usr/bin/env python3
"""Tokenize downloaded dataset shards using the DeepSeek-V3 tokenizer.

Three phases:
  1. tokenize  – each shard dir gets a _tokens.bin sidecar (resumable)
  2. merge-sources – per-shard sidecars concatenated to sources/stage__source.bin
  3. combine  – all source bins concatenated to train.bin + val.bin

Usage:
    python scripts/tokenize_dataset.py
    python scripts/tokenize_dataset.py --workers 12 --val-ratio 0.005
    python scripts/tokenize_dataset.py --merge-only   # skip phase 1
    python scripts/tokenize_dataset.py --no-combine   # skip phase 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import groupby
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (per spec)
# ---------------------------------------------------------------------------

# DeepSeek-V3 actual special token IDs (verified from AutoTokenizer)
BOS_ID: int = 0
EOS_ID: int = 1
TOKENIZER_NAME: str = "deepseek-ai/DeepSeek-V3"
DATA_ROOT: Path = Path("/home/sayak.dutta/Aether/data")
OUTPUT_ROOT: Path = Path("/home/sayak.dutta/Aether/data/tokenized")
STAGES: Tuple[str, ...] = ("pretrain", "new_pretrain", "large_pretrain")
SHARD_TOKENS_NAME: str = "_tokens.bin"

# ---------------------------------------------------------------------------
# Worker (each subprocess loads tokenizer once via initializer)
# ---------------------------------------------------------------------------

_TOKENIZER = None  # module-global inside each worker process


def _init_worker(tokenizer_name: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer  # noqa: PLC0415
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token


def _tokenize_shard(shard_path: str) -> Tuple[str, int]:
    """Tokenize one shard.  Write <shard>/_tokens.bin.  Return (shard_path, n_tokens)."""
    tokens_bin = os.path.join(shard_path, SHARD_TOKENS_NAME)
    # Resume: skip if sidecar already exists and is non-empty
    if os.path.exists(tokens_bin):
        n = os.path.getsize(tokens_bin) // 4
        if n > 0:
            return shard_path, n

    from datasets import load_from_disk  # noqa: PLC0415
    try:
        ds = load_from_disk(shard_path)
    except Exception as exc:
        print(f"[warn] failed to load {shard_path}: {exc}", flush=True)
        return shard_path, 0

    # Encode one text at a time to keep peak memory low (avoids building
    # a full batch tensor when a shard contains very long documents).
    ids: List[int] = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        token_ids = _TOKENIZER.encode(t, add_special_tokens=False)
        ids.append(BOS_ID)
        ids.extend(token_ids)
        ids.append(EOS_ID)

    if not ids:
        return shard_path, 0

    arr = np.array(ids, dtype=np.uint32)
    tmp_bin = tokens_bin + ".tmp"
    arr.tofile(tmp_bin)
    os.replace(tmp_bin, tokens_bin)
    return shard_path, len(arr)


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def _discover_shards(data_root: Path, stages: Tuple[str, ...]) -> List[Tuple[str, str, Path]]:
    """Return sorted list of (stage, source_name, shard_path).

    Avoids per-shard stat() calls on NFS: any directory whose name starts with
    'shard-' inside a source's _shards/ dir is treated as a complete shard.
    The tokenizer worker silently skips unreadable shards via the try/except in
    _tokenize_shard, so this is safe.
    """
    result: List[Tuple[str, str, Path]] = []
    for stage in stages:
        stage_dir = data_root / stage
        if not stage_dir.exists():
            continue
        for source_dir in sorted(stage_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            shards_dir = source_dir / "_shards"
            if not shards_dir.exists():
                continue
            # Use os.scandir for speed — avoids extra stat per entry on most FSes.
            with os.scandir(str(shards_dir)) as it:
                entries = sorted(
                    (e for e in it if e.name.startswith("shard-") and e.is_dir(follow_symlinks=False)),
                    key=lambda e: e.name,
                )
            for entry in entries:
                result.append((stage, source_dir.name, Path(entry.path)))
    return result


# ---------------------------------------------------------------------------
# Phase 1 – tokenize
# ---------------------------------------------------------------------------

def phase_tokenize(
    shards: List[Tuple[str, str, Path]],
    workers: int,
    tokenizer_name: str,
) -> Dict[Tuple[str, str], int]:
    """Tokenize all shards in parallel.  Returns {(stage, source): total_tokens}."""
    ctx = get_context("spawn")
    shard_paths = [str(s) for _, _, s in shards]
    source_keys = [(stage, src) for stage, src, _ in shards]

    print(f"[phase1] tokenizing {len(shard_paths):,} shards  workers={workers}", flush=True)
    t0 = time.time()
    done = 0
    source_tokens: Dict[Tuple[str, str], int] = {}

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(tokenizer_name,),
    ) as pool:
        for (stage, source), (_shard_path, n_tok) in zip(
            source_keys,
            pool.imap(_tokenize_shard, shard_paths, chunksize=4),
        ):
            key = (stage, source)
            source_tokens[key] = source_tokens.get(key, 0) + n_tok
            done += 1
            if done % 1000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = len(shard_paths) - done
                eta_s = int(remaining / rate) if rate > 0 else 0
                h, rem = divmod(eta_s, 3600)
                m, s = divmod(rem, 60)
                print(
                    f"[phase1] {done:,}/{len(shard_paths):,} shards "
                    f"rate={rate:.1f}/s  eta={h:02d}:{m:02d}:{s:02d}",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[phase1] done. {done:,} shards in {elapsed:.0f}s ({done/elapsed:.1f} shards/s)", flush=True)
    return source_tokens


# ---------------------------------------------------------------------------
# Phase 2 – merge shards into per-source .bin files
# ---------------------------------------------------------------------------

def phase_merge_sources(
    shards: List[Tuple[str, str, Path]],
    output_root: Path,
    source_tokens: Optional[Dict[Tuple[str, str], int]],
) -> None:
    """Concatenate per-shard _tokens.bin into sources/stage__source.bin."""
    sources_dir = output_root / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    def _key(item: Tuple[str, str, Path]) -> Tuple[str, str]:
        return (item[0], item[1])

    print("[phase2] merging shards into per-source .bin files", flush=True)

    for (stage, source), group_iter in groupby(sorted(shards, key=_key), key=_key):
        group = list(group_iter)
        out_path = sources_dir / f"{stage}__{source}.bin"
        expected = (source_tokens or {}).get((stage, source), None)

        if out_path.exists() and expected is not None:
            existing = out_path.stat().st_size // 4
            if existing == expected and expected > 0:
                print(f"[phase2] skip {stage}/{source}  ({existing:,} tokens already)", flush=True)
                continue

        t0 = time.time()
        written = 0
        with open(out_path, "wb") as fout:
            for _, _, shard_path in group:
                tokens_bin = shard_path / SHARD_TOKENS_NAME
                if tokens_bin.exists() and tokens_bin.stat().st_size > 0:
                    with open(tokens_bin, "rb") as fin:
                        while True:
                            chunk = fin.read(8 * 1024 * 1024)  # 8 MB
                            if not chunk:
                                break
                            fout.write(chunk)
                            written += len(chunk) // 4

        elapsed = time.time() - t0
        print(f"[phase2] {stage}/{source}  {written:,} tokens  {elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Phase 3 – combine all source bins into train.bin + val.bin
# ---------------------------------------------------------------------------

def phase_combine(output_root: Path, val_ratio: float) -> None:
    """Concatenate source .bin files into train.bin; split tail off as val.bin."""
    sources_dir = output_root / "sources"
    source_files = sorted(sources_dir.glob("*.bin"))
    if not source_files:
        print("[phase3] no source .bin files found – skipping", flush=True)
        return

    print(f"[phase3] combining {len(source_files)} source files into train.bin", flush=True)
    train_bin = output_root / "train.bin"
    manifest: Dict[str, object] = {}
    total_tokens = 0

    with open(train_bin, "wb") as fout:
        for src_file in source_files:
            n = src_file.stat().st_size // 4
            manifest[src_file.stem] = n
            total_tokens += n
            print(f"[phase3]   {src_file.name}: {n:,} tokens", flush=True)
            with open(src_file, "rb") as fin:
                while True:
                    chunk = fin.read(256 * 1024 * 1024)  # 256 MB
                    if not chunk:
                        break
                    fout.write(chunk)

    val_tokens = int(total_tokens * val_ratio)
    train_tokens = total_tokens - val_tokens

    if val_tokens > 0:
        print(f"[phase3] splitting last {val_tokens:,} tokens -> val.bin", flush=True)
        data = np.memmap(str(train_bin), dtype=np.uint32, mode="r")
        val_arr = np.array(data[train_tokens:])
        val_arr.tofile(str(output_root / "val.bin"))
        del val_arr, data
        with open(str(train_bin), "r+b") as f:
            f.truncate(train_tokens * 4)

    manifest["_total_train_tokens"] = train_tokens
    manifest["_total_val_tokens"] = val_tokens
    manifest["_sources"] = len(source_files)
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[phase3] train.bin : {train_tokens:,} tokens  ({train_tokens*4/1e9:.1f} GB)", flush=True)
    print(f"[phase3] val.bin   : {val_tokens:,} tokens", flush=True)
    print(f"[phase3] manifest  : {output_root / 'manifest.json'}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize dataset shards with DeepSeek-V3 tokenizer")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="Parallel tokenizer processes (default: min(8, cpu_count))")
    parser.add_argument("--val-ratio", type=float, default=0.005,
                        help="Fraction of tokens reserved for val.bin (default: 0.005)")
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME,
                        help=f"HuggingFace tokenizer ID (default: {TOKENIZER_NAME})")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip phase 1 tokenization; use existing _tokens.bin sidecars")
    parser.add_argument("--no-combine", action="store_true",
                        help="Skip phase 3 (do not create train.bin / val.bin)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stages = tuple(args.stages)
    shards = _discover_shards(data_root, stages)
    print(f"[tokenize] discovered {len(shards):,} shards  stages={list(stages)}", flush=True)
    print(f"[tokenize] tokenizer={args.tokenizer}  BOS={BOS_ID}  EOS={EOS_ID}", flush=True)
    print(f"[tokenize] output_root={output_root}", flush=True)

    if not shards:
        print("[tokenize] no shards found – nothing to do", flush=True)
        return

    source_tokens: Optional[Dict[Tuple[str, str], int]] = None

    if not args.merge_only:
        source_tokens = phase_tokenize(shards, args.workers, args.tokenizer)
    else:
        # Reconstruct counts from existing sidecars
        source_tokens = {}
        for stage, source, shard_path in shards:
            tb = shard_path / SHARD_TOKENS_NAME
            if tb.exists():
                key = (stage, source)
                source_tokens[key] = source_tokens.get(key, 0) + tb.stat().st_size // 4
        print(f"[tokenize] merge-only: found {sum(source_tokens.values()):,} existing tokens", flush=True)

    phase_merge_sources(shards, output_root, source_tokens)

    if not args.no_combine:
        phase_combine(output_root, args.val_ratio)

    print("[tokenize] all phases complete.", flush=True)


if __name__ == "__main__":
    main()
