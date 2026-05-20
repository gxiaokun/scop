#!/usr/bin/env bash
set -euo pipefail


BASE="."
RAW_DATA_SOURCE="${BASE}/unified_kg_cron_questions_all.json"
GRAPH="${BASE}/TimelineCronQ.pkl"

INTERMEDIATE_DIR="${BASE}/intermediate_data"
FINAL_DIR="${BASE}/final_data"

mkdir -p "${INTERMEDIATE_DIR}"
mkdir -p "${FINAL_DIR}"


echo "============================================================"
echo "[1/5] Repair duration-related question text"
echo "============================================================"
STEP1_INPUT="${INTERMEDIATE_DIR}/step1_raw.json"
cp "${RAW_DATA_SOURCE}" "${STEP1_INPUT}"

python 1_repair_duration.py \
  --input "${STEP1_INPUT}"


echo "============================================================"
echo "[2/5] Repair timeline gold answers and events"
echo "============================================================"
STEP2_INPUT="${INTERMEDIATE_DIR}/step1_raw.question_text_fixed.json"
STEP2_OUTPUT="${INTERMEDIATE_DIR}/step2_gold_repaired.json"

python 2_repair_timeline_gold.py \
  --input "${STEP2_INPUT}" \
  --graph "${GRAPH}" \
  --output "${STEP2_OUTPUT}" \
  --workers 32


echo "============================================================"
echo "[3/5] Remove duplicate questions / merge duplicated cases"
echo "============================================================"
STEP3_TARGET="${INTERMEDIATE_DIR}/step3_leakage_cleaned.json"
cp "${STEP2_OUTPUT}" "${STEP3_TARGET}"

python 3_repair_leakage.py \
  --input "${STEP3_TARGET}"


echo "============================================================"
echo "[4/5] Final formatting, balancing, and resplitting"
echo "============================================================"
STEP4_INPUT="${STEP3_TARGET}"

python 4_final_format.py \
  --input "${STEP4_INPUT}" \
  --output_dir "${FINAL_DIR}"


echo "============================================================"
echo "[5/5] LLM question repair for all splits"
echo "============================================================"

SPLITS=("train" "validation" "test")

for SPLIT in "${SPLITS[@]}"; do
    echo ">>> Starting LLM repair for: ${SPLIT}.json"
    
    STEP5_INPUT="${FINAL_DIR}/${SPLIT}.json"
    STEP5_OUTPUT="${FINAL_DIR}/${SPLIT}_repaired.json"
    
    python 5_repair_question.py \
      --input "${STEP5_INPUT}" \
      --output "${STEP5_OUTPUT}" \
      --api-key "xxx" \
      --base-url "xxx" \
      --model "gemini-3-flash" \
      --max-workers 16 
      
    echo ">>> Finished LLM repair for: ${SPLIT}.json -> ${SPLIT}_repaired.json"
done

echo "============================================================"
echo "Pipeline finished."
echo "Intermediate files & reports saved to: ${INTERMEDIATE_DIR}"
echo "Final dataset directory: ${FINAL_DIR}"
echo "============================================================"