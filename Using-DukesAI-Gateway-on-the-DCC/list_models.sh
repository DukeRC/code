#!/usr/bin/env bash
#
# A script to query all available models through Duke's AI Gateway.
# Courtesy of Drew Stinnett:https://ai.colab.duke.edu/colab-ai-blog/all-blogs/getting-started-with-dukes-ai-gateway-a-developers-guide
#
# Usage:
# Get your API token from: https://dashboard.ai.duke.edu/api-keys
# Set your API token in a .env file in this directory with the following content:
# LITELLM_TOKEN="your_api_token_here"
# or export the environment variable directly in your shell. Then run,
# ./list_models.sh

set -e

# This is the base URL for all operations
LITELLM_URL="https://litellm.oit.duke.edu/v1"

# Load environment variables from .env file if it exists
if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

if [[ -z "$LITELLM_TOKEN" ]]; then
  echo "Error: LITELLM_TOKEN is not set. Please set it to your LiteLLM API token." 1>&2
  exit 1
fi

# Query the API to list all available models
echo "Listing all models available in LiteLLM..."
curl -X GET "${LITELLM_URL}/models" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LITELLM_TOKEN}" | jq -r .data[].id | sort
