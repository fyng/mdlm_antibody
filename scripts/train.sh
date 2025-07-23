#!/bin/bash
#SBATCH --job-name=mdlm
#SBATCH --partition=componc_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=5:00:00
#SBATCH --output=log/train.out
#SBATCH --error=log/train.err

cd /data1/tanseyw/projects/feiyang/diffusion_RG/mdlm_antibody
source ~/.bashrc
mamba activate mdlm

python train.py --config configs/antibody_config.yaml