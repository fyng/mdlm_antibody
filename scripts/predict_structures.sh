#!/bin/bash
#SBATCH --job-name=boltz
#SBATCH --partition=componc_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=0:20:00
#SBATCH --output=log/boltz.out
#SBATCH --error=log/boltz.err

cd /data1/tanseyw/projects/feiyang/diffusion_RG/mdlm_antibody
source ~/.bashrc
mamba activate boltz

boltz predict output/yaml/ --use_msa_server --cache ./boltz --out_dir ./output/boltz/