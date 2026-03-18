#!/bin/bash


ENV_NAME="my_env"  # Conda env info
INPUT_PATH="./input/sample_input.csv"  # input data path info
OUTPUT_PATH="./output/sample_output.csv"  # output result data path info
TARGET_COLUMN="item_name"	# item name column
STEP=${1:-all}   # pipeline step ['all', 'preprocess', 'model', 'postprocess']
# ======================================

# 1. Conda activate
echo "[INFO] Conda 정보­: $ENV_NAME"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# 2. Python activate
echo "[INFO] 실행 모듈: $STEP"
echo "[INFO] main.py start...."
python main.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH" --target_column "$TARGET_COLUMN" --step "$STEP"

# 3.result
echo "[INFO] The end classify"
