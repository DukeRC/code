#!/bin/bash
# vLLM Server Startup Script
# Usage:
#   1. Set HF_HOME to control where Hugging Face cache is stored.
#      export HF_HOME=/path/to/hf_cache
#   2. Set the HF_TOKEN environmental variable in a .env file in the root of the directory.
#   3. Run the script:
#   ./vllm_server.sh [ADDITIONAL_VLLM_ARGS...]
#Example:
#   ./vllm_server.sh --max-model-len 8192

set -euo pipefail

# Configuration
PORT=8000
MODEL_NAME="Qwen/Qwen2-7B-Instruct"
SERVED_MODEL_NAME="local-vllm"
API_KEY="your-secret-key"

# Setup Hugging Face parameters
export HF_HOME="${HF_HOME:-/work/${USER}/.huggingface}"

# vLLM's JIT caches default to $HOME; keep them off the home quota too
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/work/${USER}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/${USER}/.cache/vllm}"
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

# Determine tensor parallel size based on CUDA_VISIBLE_DEVICES
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]]; then
  IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  TENSOR_PARALLEL_SIZE="${#gpu_ids[@]}"
else
  TENSOR_PARALLEL_SIZE="1"
fi

# Launch vLLM server
vllm serve \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_NAME" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --api-key "$API_KEY" \
  --trust-remote-code \
  "$@" \
  > "vllm_server.log" 2>&1 &

# Wait for the vLLM server to become ready by polling its models endpoint,
# and exit if the server process dies during startup.
SERVER_PID=$!
until curl -fsS -H "Authorization: Bearer ${API_KEY}" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Error: vLLM exited during startup. See vllm_server.log"
    exit 1
  fi
  sleep 2
done

NODE_FQDN="$(hostname -f)"
echo ""
echo "vLLM is serving at: http://${NODE_FQDN}:${PORT}"
echo "Model: ${MODEL_NAME}"
echo "Model Alias: ${SERVED_MODEL_NAME}"
echo "Tensor parallel: ${TENSOR_PARALLEL_SIZE}x GPU"
if (( $# > 0 )); then
  echo "Extra vLLM args: $*"
fi
echo ""
echo "Export on client shell:"
echo "export VLLM_HOST=http://${NODE_FQDN}:${PORT}"
echo "export VLLM_API_KEY=${API_KEY}"
echo ""
echo "Stop server:"
echo "kill $SERVER_PID && pkill -f VLLM::"
