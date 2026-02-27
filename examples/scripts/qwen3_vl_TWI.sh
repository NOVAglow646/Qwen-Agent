#!/usr/bin/env bash
set -euo pipefail


# Optional: set and activate your conda env before running this script.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate qwen35_vllm

# ===== Config (can be overridden by environment variables) =====
MODEL_PATH="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3-VL-235B-A22B-Instruct"
SERVED_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
ENGINE_STARTUP_TIMEOUT="${ENGINE_STARTUP_TIMEOUT:-1800}"

# ===== Environment =====
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LOCAL_OAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LOCAL_OAI_API_KEY="EMPTY"
export LOCAL_MODEL_NAME="${SERVED_MODEL_NAME}"
export VLLM_ENGINE_READY_TIMEOUT_S="${ENGINE_STARTUP_TIMEOUT}"

LOG_DIR="examples/scripts/logs"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_qwen3_vl_twi_$(date +%Y%m%d_%H%M%S).log"

# ===== Build compatibility args =====
EXTRA_ARGS=()
if vllm serve --help 2>&1 | grep -q -- "--engine-startup-timeout-seconds"; then
  EXTRA_ARGS+=("--engine-startup-timeout-seconds" "${ENGINE_STARTUP_TIMEOUT}")
elif vllm serve --help 2>&1 | grep -q -- "--engine-startup-timeout"; then
  EXTRA_ARGS+=("--engine-startup-timeout" "${ENGINE_STARTUP_TIMEOUT}")
fi

# ===== Start vLLM (background) =====
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  "${EXTRA_ARGS[@]}" >"${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM started, pid=${VLLM_PID}, log=${VLLM_LOG}"

cleanup() {
  echo "[INFO] Stopping vLLM pid=${VLLM_PID}"
  kill "${VLLM_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ===== Wait for readiness =====
MAX_WAIT_SEC="${ENGINE_STARTUP_TIMEOUT}"
SLEEP_SEC="2"
ELAPSED="0"

until python - <<'PY'
import os
import sys
import urllib.request

url = os.environ['LOCAL_OAI_BASE_URL'].rstrip('/') + '/models'
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        sys.exit(0 if 200 <= resp.status < 300 else 1)
except Exception:
    sys.exit(1)
PY
do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[ERROR] vLLM exited before ready. Last 120 log lines:"
    tail -n 120 "${VLLM_LOG}" || true
    exit 1
  fi

  if [ "${ELAPSED}" -ge "${MAX_WAIT_SEC}" ]; then
    echo "[ERROR] Timeout waiting for vLLM readiness after ${MAX_WAIT_SEC}s. Last 120 log lines:"
    tail -n 120 "${VLLM_LOG}" || true
    exit 1
  fi

  sleep "${SLEEP_SEC}"
  ELAPSED=$((ELAPSED + SLEEP_SEC))
done

echo "[INFO] vLLM is ready at ${LOCAL_OAI_BASE_URL}"

# ===== Run Qwen3-VL TWI demo =====
python examples/assistant_qwen3vl_TWI_local.py
