#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="/home/sayak.dutta/Aether/data"
SOURCE_NAME="dclm_baseline"
SOURCE_PATH="mlfoundations/dclm-baseline-1.0"
DEFAULT_EXPECTED_BYTES="7196108338436"
TARGET_DIR="$DATA_ROOT/large_pretrain/$SOURCE_NAME"
LOG_FILE="$DATA_ROOT/${SOURCE_NAME}_download.log"
ENV_FILE="$REPO_DIR/.env"
RESUME_STATE_FILE="$TARGET_DIR/_resume_state.json"

if [[ -f "$ENV_FILE" ]]; then
    export $(grep -v '^\s*#' "$ENV_FILE" | xargs)
fi

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTHONUNBUFFERED=1
MAX_RETRIES="${MAX_RETRIES:-20}"
NO_GROWTH_RESTART_SECS="${NO_GROWTH_RESTART_SECS:-900}"

if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hf_transfer") else 1)
PY
then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
fi

proc_cpu_seconds() {
    local pid="$1"
    local cputime
    cputime="$(ps -p "$pid" -o cputime= 2>/dev/null | awk '{print $1}')"
    if [[ -z "$cputime" ]]; then
        echo 0
        return
    fi

    IFS=':' read -r first second third <<< "$cputime"
    if [[ -n "$third" ]]; then
        echo $((10#$first * 3600 + 10#$second * 60 + 10#$third))
    elif [[ -n "$second" ]]; then
        echo $((10#$first * 60 + 10#$second))
    else
        echo $((10#$first))
    fi
}

mkdir -p "$DATA_ROOT"
mkdir -p "$TARGET_DIR"

cd "$REPO_DIR"

EXPECTED_BYTES="$(python - <<'PY'
from datasets import load_dataset_builder

expected = 0

try:
    builder = load_dataset_builder("mlfoundations/dclm-baseline-1.0")
    info = builder.info
    expected = int(info.download_size or 0)
except Exception:
    expected = 0

print(expected)
PY
)"

if [[ -z "${EXPECTED_BYTES}" ]]; then
    EXPECTED_BYTES=0
fi

if [[ "$EXPECTED_BYTES" -le 0 ]]; then
    EXPECTED_BYTES="$DEFAULT_EXPECTED_BYTES"
fi

if [[ "$EXPECTED_BYTES" -gt 0 ]]; then
    EXPECTED_HR="$(numfmt --to=iec-i --suffix=B "$EXPECTED_BYTES")"
    echo "[meta] HF estimated total size for $SOURCE_PATH: $EXPECTED_HR ($EXPECTED_BYTES bytes)"
else
    echo "[meta] Could not read HF download_size for $SOURCE_PATH; ETA will use live speed only."
fi

echo "[run] Starting source=$SOURCE_NAME output_root=$DATA_ROOT"
echo "[run] Log file: $LOG_FILE"

attempt=1
while true; do
    echo "[run] attempt=$attempt/$MAX_RETRIES"

    python -m aether_pipeline.download \
        --source "$SOURCE_NAME" \
        --output-root "$DATA_ROOT" \
        > >(tee -a "$LOG_FILE") \
        2> >(tee -a "$LOG_FILE" >&2) &

    DOWNLOAD_PID=$!

    START_TS="$(date +%s)"
    PREV_TS="$START_TS"
    PREV_BYTES="$(du -sb "$TARGET_DIR" 2>/dev/null | awk '{print $1}')"
    PREV_BYTES="${PREV_BYTES:-0}"
    PREV_CPU_SECS="$(proc_cpu_seconds "$DOWNLOAD_PID")"
    NO_GROWTH_SECS=0

    while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
        sleep 20

        NOW_TS="$(date +%s)"
        CUR_BYTES="$(du -sb "$TARGET_DIR" 2>/dev/null | awk '{print $1}')"
        CUR_BYTES="${CUR_BYTES:-0}"
        CUR_CPU_SECS="$(proc_cpu_seconds "$DOWNLOAD_PID")"

        DELTA_T=$((NOW_TS - PREV_TS))
        DELTA_B=$((CUR_BYTES - PREV_BYTES))
        DELTA_CPU=$((CUR_CPU_SECS - PREV_CPU_SECS))
        ELAPSED=$((NOW_TS - START_TS))

        if [[ "$DELTA_T" -le 0 ]]; then
            DELTA_T=1
        fi

        RATE_BPS=$((DELTA_B / DELTA_T))
        if [[ "$RATE_BPS" -lt 0 ]]; then
            RATE_BPS=0
        fi

        if [[ "$DELTA_B" -gt 0 || "$DELTA_CPU" -gt 0 ]]; then
            NO_GROWTH_SECS=0
        else
            NO_GROWTH_SECS=$((NO_GROWTH_SECS + DELTA_T))
        fi

        CUR_HR="$(numfmt --to=iec-i --suffix=B "$CUR_BYTES")"
        RATE_HR="$(numfmt --to=iec-i --suffix=B "$RATE_BPS")"

        RAW_ITEMS="n/a"
        KEPT_RECORDS="n/a"
        SHARD_INDEX="n/a"
        CURSOR_FILE="n/a"
        CURSOR_LINE="n/a"
        if [[ -f "$RESUME_STATE_FILE" ]]; then
            read -r RAW_ITEMS KEPT_RECORDS SHARD_INDEX CURSOR_FILE CURSOR_LINE < <(
                python - "$RESUME_STATE_FILE" <<'PY'
import json
import sys

try:
    state = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    print("n/a n/a n/a n/a n/a")
    raise SystemExit(0)

cursor = state.get("source_cursor")
if isinstance(cursor, dict):
    repo_file = str(cursor.get("repo_file") or "n/a")
    line_number = str(cursor.get("line_number") or "n/a")
else:
    repo_file = "n/a"
    line_number = "n/a"

print(
    f"{state.get('raw_items_seen', 'n/a')} "
    f"{state.get('kept_records', 'n/a')} "
    f"{state.get('next_shard_index', 'n/a')} "
    f"{repo_file} "
    f"{line_number}"
)
PY
            )
        fi

        ETA_STR="n/a"
        PCT_STR="n/a"
        if [[ "$EXPECTED_BYTES" -gt 0 ]]; then
            if [[ "$CUR_BYTES" -gt "$EXPECTED_BYTES" ]]; then
                PCT_STR="100.0"
            else
                PCT_STR="$(awk -v c="$CUR_BYTES" -v t="$EXPECTED_BYTES" 'BEGIN { printf "%.1f", (c*100.0)/t }')"
            fi

            if [[ "$RATE_BPS" -gt 0 && "$CUR_BYTES" -lt "$EXPECTED_BYTES" ]]; then
                REMAIN=$((EXPECTED_BYTES - CUR_BYTES))
                ETA_S=$((REMAIN / RATE_BPS))
                ETA_STR="$(printf '%02d:%02d:%02d' $((ETA_S/3600)) $(((ETA_S%3600)/60)) $((ETA_S%60)))"
            fi
        fi

        ELAPSED_STR="$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)))"

        echo "[progress] elapsed=$ELAPSED_STR size=$CUR_HR rate=$RATE_HR/s percent=$PCT_STR eta=$ETA_STR"
        echo "[state] raw_items_seen=$RAW_ITEMS kept_records=$KEPT_RECORDS next_shard_index=$SHARD_INDEX cursor_file=$CURSOR_FILE cursor_line=$CURSOR_LINE"
        echo "[log] $(tail -n 1 "$LOG_FILE" 2>/dev/null || true)"

        if [[ "$NO_GROWTH_SECS" -ge "$NO_GROWTH_RESTART_SECS" ]]; then
            echo "[warn] no target-dir growth for ${NO_GROWTH_SECS}s; restarting this attempt"
            kill "$DOWNLOAD_PID" 2>/dev/null || true
            break
        fi

        PREV_TS="$NOW_TS"
        PREV_BYTES="$CUR_BYTES"
        PREV_CPU_SECS="$CUR_CPU_SECS"
    done

    set +e
    wait "$DOWNLOAD_PID"
    EXIT_CODE=$?
    set -e

    FINAL_BYTES="$(du -sb "$TARGET_DIR" 2>/dev/null | awk '{print $1}')"
    FINAL_BYTES="${FINAL_BYTES:-0}"
    FINAL_HR="$(numfmt --to=iec-i --suffix=B "$FINAL_BYTES")"

    if [[ "$EXIT_CODE" -eq 0 ]]; then
        echo "[done] source=$SOURCE_NAME exit_code=0 final_size=$FINAL_HR ($FINAL_BYTES bytes)"
        echo "[done] log=$LOG_FILE"
        exit 0
    fi

    echo "[warn] downloader exited with code=$EXIT_CODE at size=$FINAL_HR ($FINAL_BYTES bytes)"

    if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
        echo "[error] max retries reached ($MAX_RETRIES); stopping"
        echo "[error] inspect log: $LOG_FILE"
        exit "$EXIT_CODE"
    fi

    backoff=$((attempt * 15))
    if [[ "$backoff" -gt 300 ]]; then
        backoff=300
    fi
    echo "[retry] sleeping ${backoff}s before retry"
    sleep "$backoff"
    attempt=$((attempt + 1))
done
