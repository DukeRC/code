#!/bin/bash
#SBATCH -p scavenger
#SBATCH -A rescomp
#SBATCH -t 01:00:00
#SBATCH --mem=2G
#SBATCH -c 1
#SBATCH -J openai-gateway-csv
#SBATCH -o openai-gateway-csv-%j.out

source ~/.bashrc
conda activate openai-gateway

cd $SLURM_SUBMIT_DIR
./openai_csv.py prompts.csv responses.csv
