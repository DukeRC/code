#!/bin/bash
#SBATCH -p scavenger
#SBATCH -t 00:10:00
#SBATCH --mem=2G
#SBATCH -c 1
#SBATCH -J openai-gateway-csv
#SBATCH -o openai-gateway-csv-%j.out

source ~/.bashrc
conda activate openai-gateway

cd $SLURM_SUBMIT_DIR
./openai_csv.py prompts.csv responses.csv
