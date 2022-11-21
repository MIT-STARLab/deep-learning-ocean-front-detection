#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name fine_tune
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --exclusive

python fine_tune.py