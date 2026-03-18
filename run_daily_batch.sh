#!/bin/bash

set -euo pipefail

log_info() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $1"
}

APP_HOME="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$APP_HOME/.venv311/bin/python}"
LOG_DIR="${LOG_DIR:-$APP_HOME/logs}"
HF_HOME_DIR="${HF_HOME_DIR:-$APP_HOME/cache/huggingface}"
ENV_FILE="${ENV_FILE:-$APP_HOME/.env.batch}"
TARGET_COLUMN="${TARGET_COLUMN:-상품명}"
STEP="${STEP:-all}"

INPUT_BASE_PATH="${INPUT_BASE_PATH:-s3://zone-1-raw-data/kicc/pos/master/shop_item_daily_update}"
OUTPUT_BASE_PATH="${OUTPUT_BASE_PATH:-s3://zone-1-raw-data/kicc/pos/master/categorized_shop_item_daily_update}"

RUN_DATE="${RUN_DATE:-$(date -d 'yesterday' '+%Y-%m-%d')}"
YEAR="$(date -d "$RUN_DATE" '+%Y')"
MONTH="$(date -d "$RUN_DATE" '+%m')"
DAY="$(date -d "$RUN_DATE" '+%d')"

INPUT_PATH="${INPUT_BASE_PATH}/year=${YEAR}/month=${MONTH}/day=${DAY}/"
OUTPUT_PATH="${OUTPUT_BASE_PATH}"

mkdir -p "$LOG_DIR" "$HF_HOME_DIR"
LOG_FILE="${LOG_DIR}/daily_job_$(date -d "$RUN_DATE" '+%Y-%m-%d').log"

{
  echo "$(date '+%Y-%m-%d %H:%M:%S') ▶️ 배치 시작"
  log_info "APP_HOME=$APP_HOME"
  log_info "PYTHON_BIN=$PYTHON_BIN"
  log_info "INPUT_PATH=$INPUT_PATH"
  log_info "OUTPUT_PATH=$OUTPUT_PATH"
  log_info "TARGET_COLUMN=$TARGET_COLUMN"
  log_info "STEP=$STEP"

  if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] Python 실행 파일을 찾을 수 없습니다: $PYTHON_BIN"
    exit 1
  fi

  if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
    log_info "Loaded env file: $ENV_FILE"
  fi

  HF_HOME="$HF_HOME_DIR" "$PYTHON_BIN" -u "$APP_HOME/main.py" --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH" --target_column "$TARGET_COLUMN" --step "$STEP"

  echo "$(date '+%Y-%m-%d %H:%M:%S') ✅ 배치 종료"
} 2>&1 | tee -a "$LOG_FILE"
