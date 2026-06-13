#!/bin/bash
#SBATCH --job-name=rlvr_grpo
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=logs/grpo_%j.log

module load python/3.11
python train_grpo.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --epochs 10 \
    --rollouts-per-task 4 \
    --out artifacts/grpo_checkpoint
