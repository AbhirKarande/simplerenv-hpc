#!/bin/bash
  
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00

module purge
module load apptainer

apptainer exec --nv --writable-tmpfs simplerenv.sif /bin/bash -c ". /opt/miniconda/etc/profile.d/conda.sh && conda activate simpler_env && python3 new_example.py"