#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name train
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --exclusive

python train.py