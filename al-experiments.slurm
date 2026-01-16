#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --job-name=al-experiments-silicon-interviews
#SBATCH --partition=clara
#SBATCH --gpus=v100:4

module load CUDA/12.4.0

nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free --format=csv -l 10 > gpu_log_$SLURM_JOB_ID.txt &

/software/jupyterhub/lab02/conda/bin/python al-experiments-base.py

echo "Job ended at $(date)"
