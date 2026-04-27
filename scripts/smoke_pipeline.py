from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "smoke"
DATASETS = ARTIFACTS / "datasets" / "pretrain"
TOKENIZED = ARTIFACTS / "tokenized"
CHECKPOINTS = ARTIFACTS / "checkpoints"


def write_dataset(name: str, texts):
    path = DATASETS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([{"text": text, "source": name} for text in texts]).save_to_disk(str(path))


def run(cmd):
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)

    write_dataset("openwebtext", ["Reference web text example.", "Another document about models and training."])
    write_dataset("c4_en", ["Common crawl sample text for testing.", "This dataset is tiny but valid."])
    write_dataset("code_python", ["def add(a, b):\n    return a + b", "class Tiny:\n    pass"])
    write_dataset("code_java", ["public class Hello { public static void main(String[] args) {} }"])
    write_dataset("code_javascript", ["function square(x) { return x * x; }"])
    write_dataset("openwebmath", ["Let x^2 + 1 = 0. Then x = i or -i."])
    write_dataset("metamathqa", ["### Instruction:\nSolve 2+2.\n\n### Response:\n4"])
    write_dataset("fineweb_edu", ["Educational text about linear algebra and optimization."])
    write_dataset("wikipedia", ["Wikipedia-like encyclopedic text about transformers."])
    write_dataset("cc_news", ["News article discussing AI systems and model training."])
    write_dataset("code_search_net", ["Find the function and describe its behavior."])
    write_dataset("code_github", ["#!/bin/bash\necho smoke"])
    write_dataset("redpajama_books", ["Long-form narrative prose about research and engineering."])
    write_dataset("arxiv_math", ["We prove the theorem using an induction argument over n."])
    write_dataset("stackexchange", ["User: How do I tokenize text?\n\nAssistant: Use the official tokenizer."])

    run(
        [
            sys.executable,
            "-m",
            "deepseek_pipeline.preprocess",
            "--dataset-root",
            str(ARTIFACTS / "datasets"),
            "--output-dir",
            str(TOKENIZED),
            "--target-train-tokens",
            "4096",
            "--val-fraction",
            "0.1",
        ]
    )
    run(
        [
            sys.executable,
            "train_end_to_end.py",
            "--data-dir",
            str(TOKENIZED),
            "--output-dir",
            str(CHECKPOINTS),
            "--preset",
            "tiny",
            "--seq-len",
            "32",
            "--max-steps",
            "2",
            "--eval-interval",
            "1",
            "--save-interval",
            "2",
        ]
    )

    latest = CHECKPOINTS / "checkpoint-latest.pt"
    best = CHECKPOINTS / "best.pth"
    if not latest.exists():
        raise SystemExit("checkpoint-latest.pt was not created")
    if not best.exists():
        raise SystemExit("best.pth was not created")

    first_run = torch.load(latest, map_location="cpu", weights_only=False)
    if first_run.get("step") != 2:
        raise SystemExit(f"expected resume checkpoint at step 2, found {first_run.get('step')}")

    run(
        [
            sys.executable,
            "train_end_to_end.py",
            "--data-dir",
            str(TOKENIZED),
            "--output-dir",
            str(CHECKPOINTS),
            "--preset",
            "tiny",
            "--seq-len",
            "32",
            "--max-steps",
            "3",
            "--eval-interval",
            "1",
            "--save-interval",
            "2",
        ]
    )

    resumed = torch.load(latest, map_location="cpu", weights_only=False)
    if resumed.get("step") != 3:
        raise SystemExit(f"expected resumed training to reach step 3, found {resumed.get('step')}")
    if "next_token_offset" not in resumed:
        raise SystemExit("resume checkpoint is missing next_token_offset")
    print(f"SMOKE_OK {latest}")


if __name__ == "__main__":
    main()
