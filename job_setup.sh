#!/bin/bash
#SBATCH --job-name=rlvr_setup
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.log

module load python/3.11
pip install -r requirements.txt
playwright install chromium
python -c "
from transformers import Qwen2VLForConditionalGeneration
Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
print('model download complete')
"
