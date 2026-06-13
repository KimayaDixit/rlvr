"""
Social-RLVR GRPO Training Script
Model: Qwen/Qwen2.5-VL-7B-Instruct
Hardware: 2x NVIDIA RTX ADA 6000 (48GB VRAM each)
Method: QLoRA 4-bit + GRPO weight updates

Usage on GPU server:
    python grpo_train.py

Or via SLURM:
    sbatch start.sh
"""

import json
import os
import re
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "training_data.json"
OUTPUT_DIR = "./grpo_rlvr_output"
LORA_SAVE_PATH = "./grpo_rlvr_lora"
SEED = 42

# ---------------------------------------------------------------------------
# Step 1: Verify GPU
# ---------------------------------------------------------------------------

print("=" * 60)
print("Social-RLVR GRPO Training")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: No GPU detected. This script requires a GPU.")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB VRAM")

print()

# ---------------------------------------------------------------------------
# Step 2: Load training data
# ---------------------------------------------------------------------------

print(f"Loading training data from {DATA_PATH}...")
if not Path(DATA_PATH).exists():
    print(f"ERROR: {DATA_PATH} not found.")
    print("Upload it via: scp training_data.json username@gpu-domain:~/rlvr_project/")
    sys.exit(1)

with open(DATA_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

rewards = [s["reward"] for s in raw_data]
print(f"Total samples: {len(raw_data)}")
print(f"  reward=1.0 (successful): {rewards.count(1.0)}")
print(f"  reward=0.0 (failed):     {rewards.count(0.0)}")
print(f"Tasks covered: {set(s['task_id'].split('.')[-1] for s in raw_data)}")
print()

# ---------------------------------------------------------------------------
# Step 3: Build successful action lookup for reward function
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(
    r"^(click|fill|select_option|noop|scroll|press|keyboard_press)\(.*\)$"
)

successful_actions = {}
for item in raw_data:
    if item["reward"] > 0 and item["action"] != "noop(100)":
        key = item["prompt"][:200]
        successful_actions[key] = item["action"]

print(f"Known successful actions in dataset: {len(successful_actions)}")

# ---------------------------------------------------------------------------
# Step 4: Format dataset for GRPO
# ---------------------------------------------------------------------------

def prepare_grpo_samples(raw_data):
    samples = []
    for item in raw_data:
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a web browser agent. "
                    "Given a task and page elements, output exactly one action as JSON. "
                    "Example: {\"action\": \"fill(\\\"16\\\", \\\"TRV-8429-IN\\\")\"}"
                ),
            },
            {
                "role": "user",
                "content": item["prompt"],
            },
        ]
        completion = (
            item["response"]
            if item["response"]
            else f'{{"action": "{item["action"]}"}}'
        )
        samples.append({
            "prompt": prompt,
            "completion": completion,
            "reward": item["reward"],
            "task_id": item["task_id"],
        })
    return samples


grpo_samples = prepare_grpo_samples(raw_data)
dataset = Dataset.from_list(grpo_samples)
split = dataset.train_test_split(test_size=0.1, seed=SEED)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples:  {len(eval_dataset)}")
print()

# ---------------------------------------------------------------------------
# Step 5: Define reward function
# ---------------------------------------------------------------------------

def parse_action_from_response(response: str) -> str | None:
    try:
        data = json.loads(response)
        if isinstance(data, list):
            data = data[0] if data else {}
        action = str(data.get("action", "")).strip()
        if ACTION_RE.match(action):
            return action
    except Exception:
        pass
    # unescaped inner quotes fallback
    m = re.search(r'"action"\s*:\s*"(.+)"[\s}]*$', response, re.DOTALL)
    if m:
        action = m.group(1).strip().rstrip('"').rstrip("}").rstrip('"').strip()
        if ACTION_RE.match(action):
            return action
    # direct regex scan
    m = re.search(r"(click|fill|select_option|noop)\([^`]{1,100}\)", response)
    if m and ACTION_RE.match(m.group(0)):
        return m.group(0)
    return None


def reward_function(
    completions: list[str],
    prompts: list[list[dict]],
    **kwargs,
) -> list[float]:
    """
    GRPO reward function.

    Scoring:
      +0.3  valid JSON with 'action' key
      +0.3  action matches allowed format (click/fill/select_option/etc)
      +0.4  action matches a known successful action from the training data
    Max: 1.0
    """
    rewards = []
    for completion, prompt_msgs in zip(completions, prompts):
        reward = 0.0

        # extract user content for lookup
        user_content = ""
        for msg in reversed(prompt_msgs):
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        # reward 1: valid JSON
        try:
            parsed = json.loads(completion)
            if isinstance(parsed, dict) and "action" in parsed:
                reward += 0.3
        except Exception:
            if "{" in completion and "action" in completion:
                reward += 0.1

        # reward 2: valid action format
        action = parse_action_from_response(completion)
        if action is not None and action != "noop(100)":
            reward += 0.3

        # reward 3: matches known successful action
        prompt_key = user_content[:200]
        if prompt_key in successful_actions:
            if action == successful_actions[prompt_key]:
                reward += 0.4

        rewards.append(reward)

    return rewards


# quick sanity check
_test = reward_function(
    ['{"action": "fill(\\"16\\", \\"TRV-8429-IN\\")"}', "I will click", "noop(100)"],
    [[{"role": "user", "content": "test"}]] * 3,
)
print(f"Reward function sanity check: {_test}")
assert _test[0] >= 0.6
assert _test[1] == 0.0
print("Reward function OK.")
print()

# ---------------------------------------------------------------------------
# Step 6: Load tokenizer
# ---------------------------------------------------------------------------

print(f"Loading tokenizer for {MODEL_NAME}...")
try:
    # Qwen2.5-VL uses AutoProcessor for vision+text
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = processor.tokenizer
except Exception:
    # fallback to tokenizer only
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
print()

# ---------------------------------------------------------------------------
# Step 7: Load model in 4-bit QLoRA
# ---------------------------------------------------------------------------

print(f"Loading {MODEL_NAME} in 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # RTX ADA supports bf16
    bnb_4bit_use_double_quant=True,
)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",          # spreads across both GPUs automatically
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded.")
print(f"Parameters: {model.num_parameters():,}")
for i in range(num_gpus):
    print(f"  GPU {i} VRAM used: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
print()

# ---------------------------------------------------------------------------
# Step 8: Apply LoRA
# ---------------------------------------------------------------------------

print("Applying LoRA adapters...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,               # higher rank — you have VRAM to spare
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print()

# ---------------------------------------------------------------------------
# Step 9: Configure GRPO training
# ---------------------------------------------------------------------------

print("Configuring GRPOTrainer...")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,

    # training duration
    num_train_epochs=3,
    per_device_train_batch_size=4,      # 4 per GPU, 2 GPUs = effective 8
    gradient_accumulation_steps=4,      # effective batch = 32
    
    # GRPO specific
    num_generations=8,                  # completions sampled per prompt for comparison
    max_new_tokens=256,
    temperature=0.8,

    # learning rate
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # memory
    gradient_checkpointing=True,
    bf16=True,                          # RTX ADA 6000 supports bf16
    optim="paged_adamw_8bit",

    # logging and saving
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    evaluation_strategy="steps",
    save_total_limit=3,                 # keep only last 3 checkpoints

    # sequence length
    max_prompt_length=1024,

    # misc
    seed=SEED,
    report_to="none",
    dataloader_num_workers=4,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_function,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print("GRPOTrainer configured.")
print(f"Training samples: {len(train_dataset)}")
print(f"Eval samples:     {len(eval_dataset)}")
print()

# ---------------------------------------------------------------------------
# Step 10: Train
# ---------------------------------------------------------------------------

print("Starting GRPO training...")
print("Progress will be logged to run.log and slurm output.")
print()

trainer.train()

# ---------------------------------------------------------------------------
# Step 11: Save
# ---------------------------------------------------------------------------

print()
print(f"Saving LoRA weights to {LORA_SAVE_PATH}...")
trainer.save_model(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)

print("Files saved:")
for f in sorted(Path(LORA_SAVE_PATH).iterdir()):
    print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

print()
print("=" * 60)
print("Training complete.")
print(f"LoRA weights saved to: {LORA_SAVE_PATH}")
print("Download via:")
print(f"  scp -r username@gpu-domain:~/rlvr_project/{LORA_SAVE_PATH} ./")
print("=" * 60)
