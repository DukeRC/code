#!/bin/bash
#SBATCH -J ollama_batch_job
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=100G
#SBATCH -t 00:10:00

# Start server
# conda activate is undefined in a batch shell until conda.sh has been sourced
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ollama-env
module load ollama/latest
ollama-serve.sh > server.log

# Extract and export OLLAMA_HOST
eval "$(grep '^export OLLAMA_HOST=' server.log)"
echo $OLLAMA_HOST

# Run inference
python ollama_client.py > output.txt
