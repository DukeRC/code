#!/bin/bash
#SBATCH -J ollama_batch_job
#SBATCH -p scavenger-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=100G
#SBATCH -t 00:10:00

module load ollama/latest
ollama-serve.sh

conda activate ollama-env
python ollama_client.py > output.txt
