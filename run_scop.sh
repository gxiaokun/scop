#!/usr/bin/env bash

set -uo pipefail

# ================= Base Path & Environment Loading =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/config.env}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] Configuration file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# config.env currently uses KEY=VALUE format; automatically exported to Python child processes via set -a.
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

# Pre-check for critical configurations: avoids exposing missing configs midway through Python execution.
: "${CHAT_MODEL:?CHAT_MODEL is not set, please check config.env}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL is not set, please check config.env}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set, please check config.env}"
: "${DATASET_DIR:?DATASET_DIR is not set, please check config.env}"
: "${BASE_STORE_DIR:?BASE_STORE_DIR is not set, please check config.env}"

# ================= Execution Parameters =================
# MultiTQ or TimelineCronQR
DATASET="${DATASET:-MultiTQ}"
TEST_SIZE="${TEST_SIZE:-60000}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MAX_TIMEOUTS="${MAX_TIMEOUTS:-20}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_FILE="${RUN_FILE:-scop_main.py}"
RUNS="${RUNS:-1}"
ABLATION_TOPK="${ABLATION_TOPK:-10}"

# By default, a single run executes all four ablation branches: full / no_triple / no_align / no_constraint.
# To run only specific branches, override before the command, for example:
# ABLATION_TYPES="full no_align" bash run_ablation_updated.sh
ABLATION_TYPES="${ABLATION_TYPES:-full}"
# ABLATION_TYPES="${ABLATION_TYPES:-full no_triple no_align no_constraint}"
read -r -a ABLATION_TYPE_LIST <<< "${ABLATION_TYPES}"

if [[ ! -f "${RUN_FILE}" ]]; then
    echo "[ERROR] Python entry script not found: ${RUN_FILE}" >&2
    exit 1
fi

# ================= Log Directory Management =================
MODEL_NAME="${CHAT_MODEL}"
# Prevent unexpected directory expansion if the model name contains '/' in the future.
MODEL_TAG="${MODEL_NAME//\//_}"

# BASE_LOG_DIR="${LOG_ROOT:-ablation_batch_logs}"
BASE_LOG_DIR="${LOG_ROOT:-${EXPERIMENT_OUTPUT_DIR:-./test_run_results}/ablation_logs}"
DATASET_LOG_DIR="${BASE_LOG_DIR}/${DATASET}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL_TAG}"
RUN_DIR="${DATASET_LOG_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUMMARY_FILE="${RUN_DIR}/summary_${MODEL_TAG}.txt"
MASTER_LOG="${RUN_DIR}/master.log"

clear || true
cat <<EOF_HEADER
============================================================
Starting Ablation Experiment Pipeline | ID: ${RUN_ID}
Current Dataset: ${DATASET}
Current Model: ${MODEL_NAME}
OpenAI-Compatible Endpoint: ${OPENAI_BASE_URL}
Planned Runs: ${RUNS}
Branches per Run: ${ABLATION_TYPES}
Test Set Size: ${TEST_SIZE}
Max Workers: ${MAX_WORKERS}
Log Directory: ${RUN_DIR}
============================================================
EOF_HEADER

{
    echo "Ablation Study Summary | Run ID: ${RUN_ID} | Dataset: ${DATASET} | LLM: ${MODEL_NAME}"
    echo "Endpoint: ${OPENAI_BASE_URL}"
    echo "Runs: ${RUNS} | Ablation Types: ${ABLATION_TYPES} | Test Size: ${TEST_SIZE} | Max Workers: ${MAX_WORKERS}"
    echo "============================================================"
} > "${SUMMARY_FILE}"

run_one() {
    local TYPE="$1"
    local SIZE="$2"
    local LOG_FILE="${RUN_DIR}/${TYPE}.log"
    local START_TIME
    START_TIME="$(date "+%Y-%m-%d %H:%M:%S")"

    echo ""
    echo ">>> [Stage Started] ${TYPE} (Size: ${SIZE})"
    echo "    Dataset: ${DATASET} | Start Time: ${START_TIME}"
    echo "    Model: ${MODEL_NAME}"

    {
        echo ""
        echo "============================================================"
        echo "Detailed Process for Ablation Branch [${TYPE}] (Live Record)"
        echo "============================================================"
    } >> "${SUMMARY_FILE}"

    export PYTHONUNBUFFERED=1

    "${PYTHON_BIN}" "${RUN_FILE}" \
        --dataset "${DATASET}" \
        --test_size "${SIZE}" \
        --max_workers "${MAX_WORKERS}" \
        --max_timeouts "${MAX_TIMEOUTS}" \
        --runs 1 \
        --ablation_type "${TYPE}" \
        --ablation_topk "${ABLATION_TOPK}" 2>&1 \
        | tee /dev/fd/2 \
        | while IFS= read -r line; do
            # Filter out progress bar noise, keep other logs.
            if echo "${line}" | grep -v -E "it/s|%" > /dev/null; then
                echo "${line}" >> "${LOG_FILE}"
                echo "${line}" >> "${MASTER_LOG}"

                # Real-time capture of key metrics and run info to write into summary.
                if echo "${line}" | grep -iE "====== run|│ Metric|│ Hit|metric|overall|multiple|single|entity|time" > /dev/null; then
                    if ! echo "${line}" | grep -E "### Table|### 表格|>>>>>> FINAL" > /dev/null; then
                        echo "${line}" >> "${SUMMARY_FILE}"
                    fi
                fi
            fi
        done

    local EXIT_CODE="${PIPESTATUS[0]}"

    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        {
            echo ""
            echo "============================================================"
            echo "Final Statistical Summary for Ablation Branch [${TYPE}]"
            sed -n '/>>>>>> FINAL ABLATION STATS/,/>>>>>> END OF STATS/p' "${LOG_FILE}"
            echo "============================================================"
            echo ""
        } >> "${SUMMARY_FILE}"
    else
        {
            echo ""
            echo "============================================================"
            echo "Ablation Branch [${TYPE}] Execution Failed | Exit Code: ${EXIT_CODE}"
            echo "Please check the log: ${LOG_FILE}"
            echo "============================================================"
            echo ""
        } >> "${SUMMARY_FILE}"

        echo "[ERROR] Branch ${TYPE} execution failed, exit code: ${EXIT_CODE}" >&2
        echo "[ERROR] Corresponding log: ${LOG_FILE}" >&2
    fi
}

# ================= Execution Queue =================
for (( i=1; i<=RUNS; i++ )); do
    echo ""
    echo "############################################################"
    echo "  Starting experiment run ${i} / ${RUNS}"
    echo "############################################################"

    for TYPE in "${ABLATION_TYPE_LIST[@]}"; do
        run_one "${TYPE}" "${TEST_SIZE}"
    done
done

echo ""
echo "All experiment runs completed."
echo "Result summary compiled at: ${SUMMARY_FILE}"
echo "Complete master log located at: ${MASTER_LOG}"