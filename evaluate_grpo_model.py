"""
STEP 3 - Run this on your LOCAL machine after downloading grpo_rlvr_lora.zip from Colab.

This script:
1. Loads the base Qwen2.5-0.5B model
2. Applies your trained LoRA weights on top
3. Serves it as a drop-in replacement for Ollama
4. Evaluates it against the original zero-shot model

Usage:
    # First extract grpo_rlvr_lora.zip into your project root
    # Then install: pip install transformers peft torch --break-system-packages
    
    .\.conda\python.exe scripts\evaluate_grpo_model.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(".playwright").resolve()))

LORA_PATH = Path("grpo_rlvr_lora")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TASK_IDS = [
    "browsergym/social_rlvr.report.extract_tracking_code",
    "browsergym/social_rlvr.gallery.aesthetic_travel_to_meera",
    "browsergym/social_rlvr.orders.priority_followup",
    "browsergym/social_rlvr.schedule.design_review_shared_slot",
]
MODEL_STEPS = 8


ACTION_RE = re.compile(
    r"^(click|fill|select_option|noop|scroll|press|keyboard_press)\(.*\)$"
)


def load_grpo_model():
    """Load base model + LoRA weights."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "transformers", "peft", "torch", "accelerate",
            "--break-system-packages", "-q"
        ])
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # load in float32 on CPU (no GPU on your machine)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )

    if LORA_PATH.exists():
        print(f"Applying LoRA weights from {LORA_PATH}")
        model = PeftModel.from_pretrained(base_model, str(LORA_PATH))
        model = model.merge_and_unload()  # merge LoRA into base weights
        print("LoRA weights merged successfully.")
    else:
        print(f"WARNING: {LORA_PATH} not found. Running base model without LoRA.")
        model = base_model

    model.eval()
    return model, tokenizer


def generate_action(model, tokenizer, prompt_text: str) -> str:
    """Generate one action from the trained model."""
    import torch

    messages = [
        {
            "role": "system",
            "content": "You are a web browser agent. Given a task and page elements, output exactly one action as JSON."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def parse_action(raw: str) -> str | None:
    """Parse action from model output."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            data = data[0] if data else {}
        action = str(data.get("action", "")).strip()
        if ACTION_RE.match(action):
            return action
    except Exception:
        pass

    m = re.search(r'"action"\s*:\s*"(.+)"[\s}]*$', raw, re.DOTALL)
    if m:
        action = m.group(1).strip().rstrip('"').rstrip("}").rstrip('"').strip()
        if ACTION_RE.match(action):
            return action

    direct = re.search(
        r"(click|fill|select_option|noop|scroll|press|keyboard_press)\([^`]{1,100}\)",
        raw,
    )
    if direct and ACTION_RE.match(direct.group(0)):
        return direct.group(0)

    return None


class GRPOPolicy:
    """Policy that uses the locally loaded GRPO-trained model."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.name = "grpo_trained_qwen_0_5b"
        self.task_id = ""
        self.step = 0

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.step = 0

    def act(self, env, obs: dict) -> str:
        self.step += 1
        from social_rlvr_web.observation import axtree_to_text, dom_to_text

        goal = obs.get("goal") or obs.get("instruction", "")
        last_error = obs.get("last_action_error", "")
        axtree = axtree_to_text(obs.get("axtree_object"), max_lines=80)
        if not axtree.strip():
            axtree = obs.get("accessibility_tree", "")
        dom = dom_to_text(obs.get("dom_object"), max_lines=60)

        prompt = f"""Goal: {goal}

Valid elements:
{axtree[:800]}

Last error: {last_error}

Output one action as JSON: {{"action": "fill(\\"bid\\", \\"value\\")"}}"""

        raw = generate_action(self.model, self.tokenizer, prompt)
        print(f"  [GRPO] raw: {raw[:100]}")
        action = parse_action(raw) or "noop(100)"
        print(f"  [GRPO] action: {action}")
        return action


def run_episode(task_id: str, policy, max_steps: int) -> dict:
    import gymnasium as gym
    import social_rlvr_web.browsergym_tasks  # noqa

    env = gym.make(task_id, headless=True, slow_mo=0, pre_observation_delay=0.05)
    policy.reset(task_id)
    trajectory = []
    reward = 0.0
    terminated = False
    truncated = False
    info = {}

    try:
        obs, _ = env.reset()
        for step in range(1, max_steps + 1):
            action = policy.act(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            task_info = info.get("task_info", {})
            trajectory.append({
                "step": step,
                "action": action,
                "reward": reward,
                "success": task_info.get("success", False),
                "verifier_message": task_info.get("verifier_message", ""),
            })
            if terminated or truncated:
                break
    finally:
        env.close()

    task_info = info.get("task_info", {})
    return {
        "policy": policy.name,
        "task_id": task_id,
        "success": bool(task_info.get("success", False)),
        "reward": float(reward),
        "steps": len(trajectory),
        "verifier_message": task_info.get("verifier_message", ""),
        "trajectory": trajectory,
    }


def main():
    model, tokenizer = load_grpo_model()
    policy = GRPOPolicy(model, tokenizer)

    results = []
    for task_id in TASK_IDS:
        print(f"\n{'='*50}")
        print(f"Task: {task_id.split('.')[-1]}")
        result = run_episode(task_id, policy, MODEL_STEPS)
        results.append(result)
        print(f"Success: {result['success']} | Steps: {result['steps']} | {result['verifier_message']}")

    print(f"\n{'='*50}")
    print("GRPO MODEL RESULTS:")
    print(f"{'='*50}")
    success_rate = sum(r["success"] for r in results) / len(results)
    print(f"Success rate: {success_rate:.1%} ({sum(r['success'] for r in results)}/{len(results)})")
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task_id'].split('.')[-1]}: {r['verifier_message']}")

    out = Path("artifacts/eval_grpo_local")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}/results.json")


if __name__ == "__main__":
    main()
