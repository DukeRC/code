#!/bin/bash
#SBATCH -J R_benchmark         # Job name
#SBATCH -p scavenger           # Partition name
#SBATCH -c 32                   # Cores for the task
#SBATCH --mem=64G              # Memory
#SBATCH -t 04:00:00            # Walltime limit (hh:mm:ss)

module load R-openblas/4.6.1
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

Rscript R_benchmark.R 6000 3 1,2,4,8,16,32
