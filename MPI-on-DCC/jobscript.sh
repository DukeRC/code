#!/bin/bash
#SBATCH -J mpi_ring_test      # Job name
#SBATCH -p scavenger          # Partition
#SBATCH -N 2                  # Total # of nodes
#SBATCH --ntasks-per-node 4   # Tasks per node

# Initialization
module load mpich/5.0.1
cd $SLURM_SUBMIT_DIR

# Ring test: neighbor exchange and collective reduction across all ranks.
mpif90 mpi_ring_test.f90 -o mpi_ring_test
mpirun -n $SLURM_NTASKS ./mpi_ring_test > mpi_ring_test_output.txt

# Hello world: reports rank, communicator size, and hostname only.
# To run this instead, comment out the two lines above and uncomment the two below.
# mpif90 mpi_hello.f90 -o mpi_hello
# mpirun -n $SLURM_NTASKS ./mpi_hello > mpi_hello_output.txt
