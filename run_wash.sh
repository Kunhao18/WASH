#!/usr/bin/env bash
#
# End-to-end WASH pipeline: shared vocab -> generation -> native detection.
#
# Edit configs/wash.json first and point every model_path at a real model.
# Then run:
#
#     bash run_wash.sh
#
# Optional environment overrides:
#     CONFIG=configs/wash.json   PROMPTS=prompts/sample.jsonl
#     RESULT_DIR=results         NUM_SAMPLES=8
#     MAX_TOKENS=256             FPR=0.05
#     EXECUTION=parallel         # parallel (one GPU/model) | sequential (single GPU, offload)
#
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-configs/wash.json}"
PROMPTS="${PROMPTS:-prompts/sample.jsonl}"
RESULT_DIR="${RESULT_DIR:-results}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"      # 0 = all prompts in the file
MAX_TOKENS="${MAX_TOKENS:-256}"
FPR="${FPR:-0.05}"
EXECUTION="${EXECUTION:-parallel}"   # parallel | sequential

mkdir -p "$RESULT_DIR"

echo "==================================================================="
echo "[1/3] Building shared vocabulary"
echo "==================================================================="
VOCAB_PATH="$(python -c "import json;print(json.load(open('$CONFIG'))['vocab_config_path'])")"
if [[ -f "$VOCAB_PATH" ]]; then
    echo "Shared vocab already exists at $VOCAB_PATH (skipping). Delete it to rebuild."
else
    python scripts/build_vocab.py --config "$CONFIG"
fi

echo
echo "==================================================================="
echo "[2/3] Generating completions (raw / watermark / wash)"
echo "==================================================================="
# raw       = unwatermarked anchor model (negatives, for FPR calibration)
# watermark = anchor model + its watermark, no attack (positives)
# wash      = the WASH attack (multi-model ensemble blend, removes watermark)
for MODE in raw watermark wash; do
    python scripts/wash_generate.py \
        --config "$CONFIG" \
        --mode "$MODE" \
        --prompts "$PROMPTS" \
        --output "$RESULT_DIR/$MODE.jsonl" \
        --num-samples "$NUM_SAMPLES" \
        --max-tokens "$MAX_TOKENS" \
        --execution "$EXECUTION"
done

echo
echo "==================================================================="
echo "[3/3] Native watermark detection (TPR@${FPR})"
echo "==================================================================="
python scripts/wash_detect.py \
    --config "$CONFIG" \
    --inputs "$RESULT_DIR/raw.jsonl" "$RESULT_DIR/watermark.jsonl" "$RESULT_DIR/wash.jsonl" \
    --fpr "$FPR"

echo
echo "Pipeline complete. Per-sample completions are in $RESULT_DIR/."
