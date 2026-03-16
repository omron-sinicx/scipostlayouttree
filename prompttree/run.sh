#!/bin/bash
set -e

# ----------------------------
# API keys
# ----------------------------
source ./.openai_apikey
source ./.google_apikey
source ./.anthropic_apikey

# ----------------------------
# experiment config
# ----------------------------
# MODEL="gpt-5.2-2025-12-11"
MODEL="gpt-5-2025-08-07"
# MODEL="gpt-5-mini-2025-08-07"
# MODEL="gemini-3-pro-preview"
# MODEL="gemini-3-flash-preview"
# MODEL="claude-haiku-4-5"

NUM_ROUNDS=1
K_FEWSHOT=2
MAX_RETRIES=3
SEED=0

OUTPUT_DIR="output"
RUN_NAME="${MODEL}_r${NUM_ROUNDS}_k${K_FEWSHOT}_s${SEED}"

# ----------------------------
# run
# ----------------------------
python src/main.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --num-rounds "$NUM_ROUNDS" \
  --k-fewshot "$K_FEWSHOT" \
  --max-retries "$MAX_RETRIES" \
  --seed "$SEED" \
