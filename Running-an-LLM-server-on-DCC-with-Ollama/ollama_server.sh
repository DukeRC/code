#!/bin/bash

# Configuration
MODEL_PATH="/work/${USER}/ollama/models"
PORT=11434

# Unset variables to avoid conflicts
unset ROCR_VISIBLE_DEVICES

# Model path
export OLLAMA_MODELS="$MODEL_PATH"

# Bind to all interfaces on that node, on port 11434
export OLLAMA_HOST="0.0.0.0:$PORT"

# Start Ollama server in the background
ollama serve &

echo "🦙 Ollama is now serving at http://$(hostname -f):$PORT"
echo "Run the following command on the client shell to connect to the Ollama server:"
echo "export OLLAMA_HOST=http://$(hostname -f):$PORT"
