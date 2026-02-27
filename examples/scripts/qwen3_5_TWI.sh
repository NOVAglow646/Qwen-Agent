#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
MODEL_PATH="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3.5-397B-A17B"
SERVED_MODEL_NAME="Qwen3.5-397B-A17B"
PORT="8000"
TP_SIZE="8"
GPU_IDS="0,1,2,3,4,5,6,7"
MAX_MODEL_LEN="32768"
GPU_MEM_UTIL="0.92"

# ===== Environment =====
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LOCAL_OAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LOCAL_OAI_API_KEY="EMPTY"
export LOCAL_MODEL_NAME="${SERVED_MODEL_NAME}"

LOG_DIR="examples/scripts/logs"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_qwen3_5_twi_$(date +%Y%m%d_%H%M%S).log"

# ===== Start vLLM (background) =====
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code >"${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM started, pid=${VLLM_PID}, log=${VLLM_LOG}"

cleanup() {
  echo "[INFO] Stopping vLLM pid=${VLLM_PID}"
  kill "${VLLM_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ===== Wait for readiness =====
MAX_WAIT_SEC="600"
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

# ===== Run TWI demo =====
python examples/assistant_qwen3_5_TWI_local.py
