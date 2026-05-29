#!/bin/bash
#SBATCH -p scavenger
#SBATCH -A rescomp
#SBATCH -t 00:05:00
#SBATCH --mem=2G
#SBATCH -c 4
#SBATCH -J openai-gateway-demo
#SBATCH -o openai-gateway-demo-%j.out

source ~/.bashrc
conda activate openai-gateway

cd $SLURM_SUBMIT_DIR
./openai_example.py "Why is the sky blue?"
