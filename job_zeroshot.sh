#!/bin/bash
#SBATCH --job-name=rlvr_zeroshot
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/zeroshot_%j.log

module load python/3.11
python evaluate_browsergym_rlvr.py \
    --policies zero_shot \
    --model-steps 20 \
    --out artifacts/eval_zeroshot
