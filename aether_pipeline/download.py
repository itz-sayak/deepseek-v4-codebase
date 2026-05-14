from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import Dataset, load_dataset
from huggingface_hub import HfFileSystem, list_repo_files
import zstandard

from .manifest import DPO_SOURCES, LARGE_PRETRAIN_SOURCES, NEW_PRETRAIN_SOURCES, PRETRAIN_SOURCES, SFT_SOURCES, SOURCE_INDEX, SourceSpec
from .tokenizer import load_aether_tokenizer


RESUME_STATE_NAME = "_resume_state.json"
SHARD_DIR_NAME = "_shards"


def _maybe_login_from_env() -> None:
    return None


def _atomic_write_json(path: str, payload: Dict[str, object]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _load_resume_state(output_dir: str) -> Dict[str, object]:
    state_path = os.path.join(output_dir, RESUME_STATE_NAME)
    if not os.path.exists(state_path):
        return {}
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _shard_root(output_dir: str) -> str:
    return os.path.join(output_dir, SHARD_DIR_NAME)


def _is_direct_saved_dataset(path: str) -> bool:
    return os.path.exists(os.path.join(path, "dataset_info.json"))


def _write_resume_state(
    output_dir: str,
    spec: SourceSpec,
    *,
    raw_items_seen: int,
    kept_records: int,
    next_shard_index: int,
    max_samples: Optional[int],
    shard_size: int,
    completed: bool,
    source_cursor: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload: Dict[str, object] = {
        "source": spec.name,
        "stage": spec.stage,
        "hf_name": spec.hf_name,
        "raw_items_seen": raw_items_seen,
        "kept_records": kept_records,
        "next_shard_index": next_shard_index,
        "max_samples": max_samples,
        "shard_size": shard_size,
        "completed": completed,
    }
    if source_cursor is not None:
        payload["source_cursor"] = source_cursor
    _atomic_write_json(os.path.join(output_dir, RESUME_STATE_NAME), payload)


def _flush_shard(output_dir: str, shard_index: int, records: List[Dict[str, object]]) -> str:
    shard_path = os.path.join(_shard_root(output_dir), f"shard-{shard_index:06d}")
    tmp_path = f"{shard_path}.tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    Dataset.from_list(records).save_to_disk(tmp_path)
    os.replace(tmp_path, shard_path)
    return shard_path


def _iter_dclm_items(
    spec: SourceSpec,
    start_raw_index: int = 0,
    start_cursor: Optional[Dict[str, object]] = None,
) -> Iterator[tuple[int, Dict[str, object], Optional[Dict[str, object]]]]:
    _maybe_login_from_env()
    token = os.environ.get("HF_TOKEN") or None
    repo_files = sorted(path for path in list_repo_files(spec.hf_name, repo_type="dataset", token=token) if path.endswith(".jsonl.zst"))
    fs = HfFileSystem(token=token)
    cursor_repo_file = None
    cursor_line_number = 0
    if start_cursor:
        cursor_repo_file = str(start_cursor.get("repo_file") or "") or None
        cursor_line_number = int(start_cursor.get("line_number") or 0)

    raw_index = start_raw_index if cursor_repo_file else 0

    for repo_file in repo_files:
        if cursor_repo_file and repo_file < cursor_repo_file:
            continue
        with fs.open(f"datasets/{spec.hf_name}/{repo_file}", "rb") as handle:
            with zstandard.ZstdDecompressor().stream_reader(handle) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                for line_number, line in enumerate(text_stream, start=1):
                    if not line.strip():
                        continue
                    if cursor_repo_file == repo_file and line_number <= cursor_line_number:
                        continue
                    current_raw_index = raw_index
                    raw_index += 1
                    if current_raw_index < start_raw_index:
                        continue
                    cursor = {"repo_file": repo_file, "line_number": line_number}
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"[skip] malformed json file={repo_file} line={line_number}: {exc}", flush=True)
                        yield current_raw_index, {}, cursor
                        continue
                    yield current_raw_index, item, cursor


def _iter_source_items(
    spec: SourceSpec,
    start_raw_index: int = 0,
    start_cursor: Optional[Dict[str, object]] = None,
) -> Iterator[tuple[int, Dict[str, object], Optional[Dict[str, object]]]]:
    if spec.name == "dclm_baseline":
        yield from _iter_dclm_items(spec, start_raw_index=start_raw_index, start_cursor=start_cursor)
        return

    _maybe_login_from_env()
    ds = load_dataset(spec.hf_name, streaming=True, **spec.hf_kwargs())
    for raw_index, item in enumerate(ds):
        if raw_index < start_raw_index:
            continue
        yield raw_index, item, None


def _save_records_incrementally(spec: SourceSpec, output_dir: str, max_samples: Optional[int], shard_size: int) -> str:
    state = _load_resume_state(output_dir)
    if state:
        previous_max_samples = state.get("max_samples")
        previous_shard_size = state.get("shard_size")
        if previous_max_samples != max_samples or previous_shard_size != shard_size:
            raise ValueError(
                f"Resume config mismatch for {spec.name}: expected max_samples={previous_max_samples} and shard_size={previous_shard_size}. "
                "Use --force to restart this source with different settings."
            )
    raw_items_seen = int(state.get("raw_items_seen", 0))
    kept_records = int(state.get("kept_records", 0))
    next_shard_index = int(state.get("next_shard_index", 0))
    source_cursor = state.get("source_cursor") if isinstance(state.get("source_cursor"), dict) else None
    if max_samples is not None and kept_records >= max_samples:
        _write_resume_state(
            output_dir,
            spec,
            raw_items_seen=raw_items_seen,
            kept_records=kept_records,
            next_shard_index=next_shard_index,
            max_samples=max_samples,
            shard_size=shard_size,
            completed=True,
            source_cursor=source_cursor,
        )
        return output_dir

    buffer: List[Dict[str, object]] = []
    current_cursor = source_cursor
    last_state_write_raw = raw_items_seen

    def write_state(*, completed: bool) -> None:
        nonlocal last_state_write_raw
        _write_resume_state(
            output_dir,
            spec,
            raw_items_seen=raw_items_seen,
            kept_records=kept_records,
            next_shard_index=next_shard_index,
            max_samples=max_samples,
            shard_size=shard_size,
            completed=completed,
            source_cursor=current_cursor,
        )
        last_state_write_raw = raw_items_seen

    def flush_buffer() -> None:
        nonlocal next_shard_index, buffer
        if not buffer:
            return
        _flush_shard(output_dir, next_shard_index, buffer)
        next_shard_index += 1
        buffer = []
        write_state(completed=False)

    try:
        for raw_index, item, item_cursor in _iter_source_items(spec, start_raw_index=raw_items_seen, start_cursor=source_cursor):
            raw_items_seen = raw_index + 1
            current_cursor = item_cursor or current_cursor
            # Persist cursor progress during long catch-up scans so retries resume quickly.
            if current_cursor is not None and raw_items_seen - last_state_write_raw >= 10000:
                write_state(completed=False)
            record = normalize_record(spec, item)
            if record is None:
                continue
            record["source"] = spec.name
            buffer.append(record)
            kept_records += 1
            if len(buffer) >= shard_size:
                flush_buffer()
            if max_samples is not None and kept_records >= max_samples:
                break
    except BaseException:
        flush_buffer()
        raise

    flush_buffer()
    write_state(completed=True)
    return output_dir


def _strip(text: Optional[str]) -> str:
    return (text or "").strip()


def _format_instruction(instruction: str, response: str, inp: str = "") -> Optional[Dict[str, str]]:
    instruction = _strip(instruction)
    response = _strip(response)
    inp = _strip(inp)
    if not instruction or not response:
        return None
    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{response}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    return {"text": text}


def normalize_record(spec: SourceSpec, item: Dict[str, object]) -> Optional[Dict[str, object]]:
    kind = spec.text_kind
    if kind in {"text", "content"}:
        value = _strip(item.get(kind)) or _strip(item.get("text")) or _strip(item.get("content"))
        return {"text": value} if value else None
    if kind == "the_stack":
        value = _strip(item.get("content"))
        return {"text": value, "language": _strip(item.get("lang"))} if value else None
    if kind == "the_stack_smol":
        value = _strip(item.get("content"))
        return {"text": value, "language": _strip(item.get("lang"))} if value else None
    if kind == "metamathqa":
        query = _strip(item.get("query") or item.get("question") or item.get("problem"))
        response = _strip(item.get("response") or item.get("answer") or item.get("solution"))
        return _format_instruction(query, response)
    if kind == "wikipedia":
        value = _strip(item.get("text"))
        title = _strip(item.get("title"))
        return {"text": f"{title}\n\n{value}" if title else value} if value else None
    if kind == "code_search_net":
        code = _strip(item.get("func_code_string"))
        doc = _strip(item.get("func_documentation_string"))
        lang = _strip(item.get("language"))
        if not code:
            return None
        return {"text": f"{doc}\n\n{code}" if doc else code, "language": lang}
    if kind == "ultrachat":
        messages = item.get("messages") or []
        parts = []
        for message in messages:
            role = _strip(message.get("role")).capitalize()
            content = _strip(message.get("content"))
            if role and content:
                parts.append(f"{role}: {content}")
        text = "\n\n".join(parts)
        return {"text": text} if text else None
    if kind == "flan":
        return _format_instruction(_strip(item.get("inputs")), _strip(item.get("targets")))
    if kind == "smoltalk":
        messages = item.get("messages") or []
        blocks = []
        for idx in range(len(messages) - 1):
            first = messages[idx]
            second = messages[idx + 1]
            if _strip(first.get("role")).lower() == "user" and _strip(second.get("role")).lower() == "assistant":
                block = _format_instruction(_strip(first.get("content")), _strip(second.get("content")))
                if block:
                    blocks.append(block["text"])
        return {"text": "\n\n".join(blocks)} if blocks else None
    if kind == "alpaca":
        return _format_instruction(_strip(item.get("instruction")), _strip(item.get("output")), _strip(item.get("input")))
    if kind == "sft_messages":
        conversations = item.get("conversations") or item.get("messages") or []
        parts = []
        for message in conversations:
            role = _strip(message.get("from") or message.get("role")).capitalize()
            content = _strip(message.get("value") or message.get("content"))
            if role and content:
                parts.append(f"{role}: {content}")
        return {"text": "\n\n".join(parts)} if parts else None
    if kind == "sharegpt":
        conversations = item.get("conversations") or []
        parts = []
        for message in conversations:
            role = _strip(message.get("from")).capitalize()
            content = _strip(message.get("value"))
            if role and content:
                parts.append(f"{role}: {content}")
        return {"text": "\n\n".join(parts)} if parts else None
    if kind == "orca_math":
        return _format_instruction(_strip(item.get("question")), _strip(item.get("answer")))
    if kind == "wizardlm":
        return _format_instruction(_strip(item.get("instruction")), _strip(item.get("output")))
    if kind == "dolly":
        return _format_instruction(_strip(item.get("instruction")), _strip(item.get("response")), _strip(item.get("context")))
    if kind == "preference":
        prompt = _strip(item.get("prompt") or item.get("instruction"))
        chosen = _strip(item.get("chosen") or item.get("response_j"))
        rejected = _strip(item.get("rejected") or item.get("response_k"))
        if not prompt or not chosen or not rejected:
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    raise ValueError(f"Unsupported text_kind: {kind}")


def save_source(spec: SourceSpec, output_root: str, max_samples: Optional[int], force: bool, shard_size: int) -> str:
    output_dir = os.path.join(output_root, spec.stage, spec.name)
    if os.path.exists(output_dir):
        if not force:
            state = _load_resume_state(output_dir)
            if _is_direct_saved_dataset(output_dir) or state.get("completed"):
                return output_dir
        else:
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return _save_records_incrementally(spec, output_dir, max_samples=max_samples, shard_size=shard_size)


def stage_sources(stage: str) -> Iterable[SourceSpec]:
    if stage == "pretrain":
        return PRETRAIN_SOURCES
    if stage == "new_pretrain":
        return NEW_PRETRAIN_SOURCES
    if stage == "large_pretrain":
        return LARGE_PRETRAIN_SOURCES
    if stage == "sft":
        return SFT_SOURCES
    if stage == "dpo":
        return DPO_SOURCES
    if stage == "all":
        return PRETRAIN_SOURCES + NEW_PRETRAIN_SOURCES + SFT_SOURCES + DPO_SOURCES
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the reference datasets into this repo's format.")
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "new_pretrain", "large_pretrain", "sft", "dpo", "all"])
    parser.add_argument("--output-root", default="./artifacts/datasets")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-size", type=int, default=1000, help="Rows per on-disk shard for resumable downloads.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-manifest", action="store_true")
    parser.add_argument("--tokenizer-check", action="store_true", help="Download and print Aether tokenizer metadata.")
    args = parser.parse_args()

    if args.print_manifest:
        for spec in stage_sources(args.stage):
            print(f"{spec.stage:12s} {spec.name:20s} {spec.hf_name} {spec.hf_kwargs()}")
        return

    if args.tokenizer_check:
        tok = load_aether_tokenizer()
        print(
            {
                "name_or_path": tok.name_or_path,
                "vocab_size": tok.vocab_size,
                "bos_token_id": tok.bos_token_id,
                "eos_token_id": tok.eos_token_id,
                "pad_token_id": tok.pad_token_id,
            }
        )

    selected = [SOURCE_INDEX[name] for name in args.source] if args.source else list(stage_sources(args.stage))
    for index, spec in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] downloading {spec.stage}/{spec.name}")
        path = save_source(spec, args.output_root, args.max_samples, args.force, args.shard_size)
        print(f"saved {spec.name} -> {path}")


if __name__ == "__main__":
    main()
