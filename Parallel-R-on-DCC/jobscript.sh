#!/bin/bash
#SBATCH -J parallel-R          # Job name
#SBATCH -p scavenger           # Partition name
#SBATCH -c 8                   # Cores for the task
#SBATCH --mem=64G              # Memory
#SBATCH -t 04:00:00            # Walltime limit (hh:mm:ss)

module load R-openblas/4.6.1
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

Rscript my_analysis.R
