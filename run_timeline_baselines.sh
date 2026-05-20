#!/usr/bin/env bash

set -uo pipefail

# =========================
# 0. Load Environment Configuration
# =========================

CONFIG_ENV=${CONFIG_ENV:-config.env}

if [[ ! -f "${CONFIG_ENV}" ]]; then
  echo "[ERROR] Configuration file not found: ${CONFIG_ENV}"
  echo "Please execute in the project root directory, or specify the config file via CONFIG_ENV=/path/to/config.env."
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_ENV}"
set +a

# =========================
# 1. Basic Configuration
# =========================

PYTHON_BIN=${PYTHON_BIN:-python}
RUN_FILE=${RUN_FILE:-timeline_baseline.py}

TEST_SIZE=${TEST_SIZE:-2500}
MAX_WORKERS=${MAX_WORKERS:-6}

FINAL_K=${FINAL_K:-20}
FILTER_RETRIEVE_K=${FILTER_RETRIEVE_K:-50}
MAX_STEPS=${MAX_STEPS:-5}

BASE_LOG_DIR=${LOG_DIR:-${EXPERIMENT_OUTPUT_DIR:-./test_run_results}/corn_baseline/baseline_logs}

# Currently, the main function in cr_base.py reads RUNS_DIR, not EXPERIMENT_OUTPUT_DIR.
# If RUNS_DIR is not explicitly specified externally, it defaults to run_results/baseline.
# BASELINE_OUTPUT_DIR=${BASELINE_OUTPUT_DIR:-${EXPERIMENT_OUTPUT_DIR:-./run_results}}
# export RUNS_DIR=${RUNS_DIR:-${BASELINE_OUTPUT_DIR}}
# mkdir -p "${RUNS_DIR}"

RUN_ID=$(date +%Y%m%d_%H%M%S)_pid$$

RUN_ROOT="${BASE_LOG_DIR}/runs"
SUMMARY_ROOT="${BASE_LOG_DIR}/summaries"

RUN_DIR="${RUN_ROOT}/${RUN_ID}"
MODE_LOG_DIR="${RUN_DIR}/mode_logs"

mkdir -p "${MODE_LOG_DIR}"
mkdir -p "${SUMMARY_ROOT}"

MASTER_LOG="${RUN_DIR}/master.log"
ALL_LIVE_LOG="${RUN_DIR}/all_live.log"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
SUMMARY_SNAPSHOT="${SUMMARY_ROOT}/summary_${RUN_ID}.txt"
DONE_FILE="${RUN_DIR}/ALL_DONE"

# Basic Input Validation
if [[ ! -f "${RUN_FILE}" ]]; then
  echo "[ERROR] Python entry script not found: ${RUN_FILE}"
  exit 1
fi

required_envs=(
  CHAT_MODEL
  OPENAI_BASE_URL
  OPENAI_API_KEY
  EMBED_MODEL_PATH
  DATASET_DIR
  BASE_STORE_DIR
)

for env_name in "${required_envs[@]}"; do
  if [[ -z "${!env_name:-}" ]]; then
    echo "[ERROR] Missing required environment variable in config.env: ${env_name}"
    exit 1
  fi
done

# Create symlinks for easy access to the latest run
ln -sfn "runs/${RUN_ID}" "${BASE_LOG_DIR}/latest_run"
ln -sfn "runs/${RUN_ID}/master.log" "${BASE_LOG_DIR}/latest_master.log"
ln -sfn "runs/${RUN_ID}/all_live.log" "${BASE_LOG_DIR}/latest_all_live.log"
ln -sfn "runs/${RUN_ID}/summary.txt" "${BASE_LOG_DIR}/latest_summary.txt"

# Output of the script itself goes to both the master log and the terminal
exec > >(tee -a "${MASTER_LOG}") 2>&1

# Reduce Python buffering for easier real-time monitoring
export PYTHONUNBUFFERED=1

# Write summary to both the current run directory and the historical summaries directory
write_summary() {
  tee -a "${SUMMARY_FILE}" "${SUMMARY_SNAPSHOT}"
}

# =========================
# 2. Modes to Run
# =========================

# By default, run all baselines at once.
# To temporarily run only specific modes, execute:
# MODE_LIST="react ircot" bash run_baseline_batch_updated.sh
MODE_LIST=${MODE_LIST:-"rag rag_filter react react_filter ircot ircot_filter hyde_rag query2doc_rag hyde_rag_filter query2doc_rag_filter"}
# MODE_LIST=${MODE_LIST:-"llm cot rag rag_filter react react_filter ircot ircot_filter"}
read -r -a MODES <<< "${MODE_LIST}"

if [[ ${#MODES[@]} -eq 0 ]]; then
  echo "[ERROR] MODE_LIST is empty, no modes to execute."
  exit 1
fi

touch "${ALL_LIVE_LOG}"
touch "${SUMMARY_FILE}"
touch "${SUMMARY_SNAPSHOT}"

# =========================
# 3. Print Startup Information
# =========================

{
  echo "===== Baseline Batch Run ====="
  echo "Run ID: ${RUN_ID}"
  echo "PWD: $(pwd)"
  echo "CONFIG_ENV=${CONFIG_ENV}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "RUN_FILE=${RUN_FILE}"
  echo "CHAT_MODEL=${CHAT_MODEL}"
  echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
  echo "EMBED_MODEL_PATH=${EMBED_MODEL_PATH}"
  echo "RUNS_DIR=${RUNS_DIR}"
  echo "TEST_SIZE=${TEST_SIZE}"
  echo "FINAL_K=${FINAL_K}"
  echo "FILTER_RETRIEVE_K=${FILTER_RETRIEVE_K}"
  echo "MAX_STEPS=${MAX_STEPS}"
  echo "MAX_WORKERS=${MAX_WORKERS}"
  echo "MODES=${MODES[*]}"
  echo "RUN_DIR=${RUN_DIR}"
  echo ""
} | write_summary

# Save a copy of the config for future reproduction; will not log API keys.
cat > "${RUN_DIR}/run_config.env" <<EOF_INNER
RUN_ID=${RUN_ID}
CONFIG_ENV=${CONFIG_ENV}
PYTHON_BIN=${PYTHON_BIN}
RUN_FILE=${RUN_FILE}
CHAT_MODEL=${CHAT_MODEL}
OPENAI_BASE_URL=${OPENAI_BASE_URL}
EMBED_MODEL_PATH=${EMBED_MODEL_PATH}
RUNS_DIR=${RUNS_DIR}
TEST_SIZE=${TEST_SIZE}
FINAL_K=${FINAL_K}
FILTER_RETRIEVE_K=${FILTER_RETRIEVE_K}
MAX_STEPS=${MAX_STEPS}
MAX_WORKERS=${MAX_WORKERS}
MODE_LIST=${MODES[*]}
EOF_INNER

# =========================
# 4. Run All Baselines Sequentially
# =========================

for MODE in "${MODES[@]}"; do
  LOG_FILE="${MODE_LOG_DIR}/${MODE}.log"

  {
    echo "========================================"
    echo "Start mode: ${MODE}"
    echo "Log file: ${LOG_FILE}"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
  } | write_summary

  echo "[${MODE}] ===== START $(date '+%Y-%m-%d %H:%M:%S') =====" >> "${ALL_LIVE_LOG}"

  "${PYTHON_BIN}" "${RUN_FILE}" \
    --mode "${MODE}" \
    --final_k "${FINAL_K}" \
    --filter_retrieve_k "${FILTER_RETRIEVE_K}" \
    --max_steps "${MAX_STEPS}" \
    --max_workers "${MAX_WORKERS}" \
    --test_size "${TEST_SIZE}" \
    2>&1 \
    | tee -a "${LOG_FILE}" \
    | sed -u "s/^/[${MODE}] /" \
    | tee -a "${ALL_LIVE_LOG}"

  EXIT_CODE=${PIPESTATUS[0]}

  echo "[${MODE}] ===== END $(date '+%Y-%m-%d %H:%M:%S') | EXIT_CODE=${EXIT_CODE} =====" >> "${ALL_LIVE_LOG}"

  {
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Exit code: ${EXIT_CODE}"
  } | write_summary

  if [[ "${EXIT_CODE}" -ne 0 ]]; then
    {
      echo "Mode ${MODE} failed. Check log: ${LOG_FILE}"
      echo ""
    } | write_summary
    continue
  fi

  {
    echo "Metrics for ${MODE}:"
    grep -E "Overall:|Simple|Medium|Complex" "${LOG_FILE}" || echo "No metrics found in ${LOG_FILE}"
    echo ""
  } | write_summary
done

# =========================
# 5. Completion Marker
# =========================

{
  echo "===== All baseline runs finished ====="
  echo "Finish time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Run directory: ${RUN_DIR}"
  echo "Summary saved to: ${SUMMARY_FILE}"
  echo "Summary snapshot saved to: ${SUMMARY_SNAPSHOT}"
  echo "Unified live log saved to: ${ALL_LIVE_LOG}"
  echo "Master log saved to: ${MASTER_LOG}"
  echo "Baseline result root: ${RUNS_DIR}"
} | write_summary

touch "${DONE_FILE}"
ln -sfn "runs/${RUN_ID}/ALL_DONE" "${BASE_LOG_DIR}/LATEST_ALL_DONE"

echo ""
echo "DONE_FILE=${DONE_FILE}"
echo "Use these commands:"
echo "  cat ${BASE_LOG_DIR}/latest_summary.txt"
echo "  tail -f ${BASE_LOG_DIR}/latest_all_live.log"
echo "  ls ${BASE_LOG_DIR}/runs"
echo "  find ${RUNS_DIR} -maxdepth 2 -type f | sort"