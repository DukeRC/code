#!/bin/bash
#SBATCH -J vllm-local          # Job name
#SBATCH -p gpu-common          # Partition name
#SBATCH -A rescomp             # Account
#SBATCH --gres=gpu:5000_ada:1  # One RTX 5000 Ada GPU
#SBATCH --mem=100G             # Memory
#SBATCH -t 1:00:00             # Time limit hrs:min:sec
#SBATCH -o vllm-%j.out         # Standard output and error log

cd $SLURM_SUBMIT_DIR

# conda activate is undefined in a batch shell until conda.sh has been sourced
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-env

# Batch jobs do not read ~/.bashrc, so set the cache locations here
export HF_HOME="/work/${USER}/.huggingface"
export FLASHINFER_WORKSPACE_BASE="/work/${USER}"
export VLLM_CACHE_ROOT="/work/${USER}/.cache/vllm"

./vllm_local.py
