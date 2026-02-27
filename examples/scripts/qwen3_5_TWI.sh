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

# ===== Start vLLM (background) =====
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code &

VLLM_PID=$!
echo "[INFO] vLLM started, pid=${VLLM_PID}"

cleanup() {
  echo "[INFO] Stopping vLLM pid=${VLLM_PID}"
  kill "${VLLM_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Optional: wait a bit for server warmup
sleep 8

# ===== Run TWI demo =====
python examples/assistant_qwen3_5_TWI_local.py
