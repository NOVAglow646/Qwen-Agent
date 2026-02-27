#!/usr/bin/env bash
set -euo pipefail

conda activate qwen35_vllm

# ===== Config =====
MODEL_PATH="Qwen/Qwen3.5-397B-A17B-FP8"
SERVED_MODEL_NAME="Qwen3.5-397B-A17B-FP8"
PORT="8000"
TP_SIZE="8"
GPU_IDS="0,1,2,3,4,5,6,7"
MAX_MODEL_LEN="32768"
GPU_MEM_UTIL="0.92"
ENGINE_STARTUP_TIMEOUT="${ENGINE_STARTUP_TIMEOUT:-1800}"

# ===== Environment =====
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export LOCAL_OAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LOCAL_OAI_API_KEY="EMPTY"
export LOCAL_MODEL_NAME="${SERVED_MODEL_NAME}"

LOG_DIR="examples/scripts/logs"
mkdir -p "${LOG_DIR}"
VLLM_LOG="${LOG_DIR}/vllm_qwen3_5_twi_$(date +%Y%m%d_%H%M%S).log"

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
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
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
