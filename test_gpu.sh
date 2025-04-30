#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB

echo "Running on $(hostname)"
nvidia-smi
