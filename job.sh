#!/bin/bash

#SBATCH --account=jdeshmuk_1278
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

module purge
module load apptainer

apptainer exec --nv --writable-tmpfs simplerenv.sif python example.py
