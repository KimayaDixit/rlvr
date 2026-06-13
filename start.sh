#!/bin/bash
#SBATCH --job-name=rlvr_grpo
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=2023300049@spit.ac.in

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Load miniconda to get Python 3.13
module load miniconda3/25.5.1

# Create a fresh venv — isolates from system site-packages with old TRL
python -m venv grpo_env --clear
source grpo_env/bin/activate

# Verify we are using the venv Python, not system Python
which python
python --version

# Upgrade pip first
pip install --upgrade pip -q

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Install exact versions that include GRPOTrainer
pip install "transformers==4.47.0" \
            "trl==0.12.0" \
            "peft==0.14.0" \
            "accelerate==1.2.0" \
            "bitsandbytes==0.45.0" \
            "datasets==3.2.0" \
            "einops" \
            "qwen-vl-utils" \
            "huggingface_hub" -q

# Verify GRPOConfig is importable — if this fails, training will not start
python -c "from trl import GRPOConfig, GRPOTrainer; print('GRPOTrainer import OK')"
if [ $? -ne 0 ]; then
    echo "ERROR: GRPOTrainer import failed. Check TRL installation."
    exit 1
fi

# Verify GPU is visible to PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo "Dependencies installed and verified at: $(date)"

# Run training — all output goes to run.log
python grpo_train.py > run.log 2>&1

echo "Training finished at: $(date)"
