from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm

from .manifest import PRETRAIN_SOURCES, SOURCE_INDEX
from .tokenizer import load_deepseek_tokenizer


RESUME_STATE_NAME = "_resume_state.json"
SHARD_DIR_NAME = "_shards"


def load_preprocessed_source(dataset_dir: str):
    direct_dataset_info = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.exists(direct_dataset_info):
        return load_from_disk(dataset_dir)

    state_path = os.path.join(dataset_dir, RESUME_STATE_NAME)
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        if not state.get("completed"):
            raise RuntimeError(f"Dataset download is incomplete: {dataset_dir}")

    shard_root = os.path.join(dataset_dir, SHARD_DIR_NAME)
    if os.path.isdir(shard_root):
        shard_dirs = sorted(
            os.path.join(shard_root, name)
            for name in os.listdir(shard_root)
            if name.startswith("shard-")
        )
        if not shard_dirs:
            raise RuntimeError(f"No dataset shards found in {dataset_dir}")
        shards = [load_from_disk(path) for path in shard_dirs]
        if len(shards) == 1:
            return shards[0]
        return concatenate_datasets(shards)

    raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")


def _iter_texts(dataset_dir: str, start: int = 0, stop: Optional[int] = None) -> Iterable[str]:
    dataset = load_preprocessed_source(dataset_dir)
    end = len(dataset) if stop is None else min(stop, len(dataset))
    for idx in range(start, end):
        item = dataset[idx]
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def _tokenize_texts(
    texts: Iterable[str],
    tokenizer_name: str,
    tokenizer_cache_dir: Optional[str],
    output_path: str,
    token_budget: Optional[int],
) -> int:
    tok = load_deepseek_tokenizer(name_or_path=tokenizer_name, cache_dir=tokenizer_cache_dir)
    total = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "ab") as handle:
        for text in texts:
            ids = tok.encode_ordinary(text)
            if not ids:
                continue
            ids.append(tok.eos_token_id)
            if token_budget is not None and total + len(ids) > token_budget:
                remaining = token_budget - total
                if remaining <= 0:
                    break
                ids = ids[:remaining]
            np.asarray(ids, dtype=np.uint32).tofile(handle)
            total += len(ids)
            if token_budget is not None and total >= token_budget:
                break
    return total


def build_pretrain_bins(
    dataset_root: str,
    output_dir: str,
    tokenizer_name: str,
    tokenizer_cache_dir: Optional[str],
    target_train_tokens: int,
    val_fraction: float,
    sources: Optional[List[str]] = None,
) -> None:
    chosen = [SOURCE_INDEX[name] for name in sources] if sources else PRETRAIN_SOURCES
    total_weight = sum(spec.weight for spec in chosen)
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    for path in (train_path, val_path):
        if os.path.exists(path):
            os.remove(path)
    os.makedirs(output_dir, exist_ok=True)

    for spec in chosen:
        src_dir = os.path.join(dataset_root, spec.stage, spec.name)
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Missing dataset directory: {src_dir}")
        source_tokens = int(target_train_tokens * (spec.weight / total_weight))
        val_tokens = int(source_tokens * val_fraction)
        train_tokens = source_tokens - val_tokens
        dataset = load_preprocessed_source(src_dir)
        if len(dataset) == 0:
            raise RuntimeError(f"No texts found in {src_dir}")
        split_index = max(1, int(len(dataset) * (1.0 - val_fraction)))
        train_texts = tqdm(_iter_texts(src_dir, 0, split_index), desc=f"tokenize:{spec.name}:train", unit=" docs")
        val_texts = tqdm(_iter_texts(src_dir, split_index, None), desc=f"tokenize:{spec.name}:val", unit=" docs")
        wrote_train = _tokenize_texts(train_texts, tokenizer_name, tokenizer_cache_dir, train_path, train_tokens)
        wrote_val = _tokenize_texts(val_texts, tokenizer_name, tokenizer_cache_dir, val_path, val_tokens if val_tokens > 0 else 0)
        print(f"{spec.name}: wrote {wrote_train:,} train tokens and {wrote_val:,} val tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize downloaded datasets into train.bin/val.bin using the DeepSeek tokenizer.")
    parser.add_argument("--dataset-root", default="./artifacts/datasets")
    parser.add_argument("--output-dir", default="./artifacts/tokenized")
    parser.add_argument("--tokenizer-name", default=os.environ.get("DEEPSEEK_TOKENIZER_NAME", "deepseek-ai/DeepSeek-V3.2"))
    parser.add_argument("--tokenizer-cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--target-train-tokens", type=int, default=5_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()

    build_pretrain_bins(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        target_train_tokens=args.target_train_tokens,
        val_fraction=args.val_fraction,
        sources=args.source or None,
    )


if __name__ == "__main__":
    main()
