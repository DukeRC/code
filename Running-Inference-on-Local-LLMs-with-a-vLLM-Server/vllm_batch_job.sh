#!/bin/bash
#SBATCH -J vllm_batch_job
#SBATCH -p gpu-common
#SBATCH -A rescomp
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --mem=100G
#SBATCH -t 00:30:00

cd $SLURM_SUBMIT_DIR

# conda activate is undefined in a batch shell until conda.sh has been sourced
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-env

# Start the server; the script returns once it is ready to accept requests
./vllm_server.sh > server.log

# Extract the connection details printed by the server
eval "$(grep '^export VLLM_HOST=' server.log)"
eval "$(grep '^export VLLM_API_KEY=' server.log)"

# Run inference
./vllm_client.py > output.txt

# Stop the server
pkill -f VLLM::
