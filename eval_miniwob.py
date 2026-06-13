"""
eval_miniwob.py — Zero-shot evaluation script for MiniWoB++ tasks.

Usage (Ollama backend, quick test):
    python eval_miniwob.py --backend ollama --model qwen2.5vl:3b --tasks click-button,login-user --episodes 5

Usage (HuggingFace backend, research pipeline):
    python eval_miniwob.py --backend hf --model Qwen/Qwen2.5-VL-3B-Instruct --tasks click-button,login-user --episodes 10

Usage (load trained LoRA checkpoint for post-GRPO eval):
    python eval_miniwob.py --backend hf --model Qwen/Qwen2.5-VL-3B-Instruct --lora-checkpoint ./checkpoints/grpo_qwen25_3b --tasks click-button --episodes 10

Requirements:
    pip install browsergym-miniwob playwright pillow requests
    pip install transformers torch accelerate peft qwen-vl-utils   # for --backend hf
    python -m playwright install chromium

Environment:
    MINIWOB_URL=http://127.0.0.1:8000/   # local HTML server for MiniWoB assets
                                          # serve with: python -m http.server 8000 --directory miniwob-plusplus/miniwob/html
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import browsergym.core
import browsergym.miniwob

# ─────────────────────────────────────────────────────────────────────────────
# Curated MiniWoB task list
# Selected for: binary rewards, 10-40% expected zero-shot SR, visual grounding
# needed, similar structure to WebArena (forms, clicks, text input)
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1 — simple single-action tasks (good for sanity checking setup)
TIER1_TASKS = [
    "click-button",
    "click-button-sequence",
    "click-checkboxes",
    "click-color",
    "click-dialog",
    "click-link",
    "click-option",
    "click-tab",
    "focus-text",
    "focus-text-2",
]

# Tier 2 — multi-step tasks (target zone: ~10-40% zero-shot SR)
TIER2_TASKS = [
    "login-user",
    "enter-date",
    "enter-text",
    "enter-time",
    "choose-list",
    "click-checkboxes-soft",
    "click-shades",
    "click-tab-2",
    "navigate-tree",
    "search-engine",
]

# Tier 3 — harder tasks (for generalization eval, not training)
TIER3_TASKS = [
    "book-flight",
    "email-inbox",
    "find-word",
    "read-table",
    "social-media",
    "use-autocomplete",
    "use-spinner",
]

# Default evaluation set: Tier 2 (learnable zone for GRPO)
DEFAULT_TASKS = TIER2_TASKS

# Map task name → known optimal step count (None = use --max-steps)
OPTIMAL_STEPS: dict[str, Optional[int]] = {
    "click-button": 1,
    "click-button-sequence": 3,
    "click-checkboxes": 2,
    "click-color": 1,
    "click-dialog": 1,
    "click-link": 1,
    "click-option": 1,
    "click-tab": 1,
    "focus-text": 1,
    "login-user": 3,
    "enter-date": 2,
    "enter-text": 2,
    "choose-list": 2,
    "search-engine": 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Observation serialization
# ─────────────────────────────────────────────────────────────────────────────

ACTION_FORMAT = """
Return your action as a single JSON object:
  {"action": "click(\\"bid\\")"}          — click an element
  {"action": "fill(\\"bid\\", \\"text\\")"} — type into a text field
  {"action": "select_option(\\"bid\\", \\"value\\")"}  — choose a dropdown option
  {"action": "check(\\"bid\\")"}          — tick a checkbox
  {"action": "noop(500)"}               — wait (use if page is loading)

Use the exact bid values from the accessibility tree below.
Do not explain. Return only the JSON object.
""".strip()


def build_text_prompt(obs: dict) -> str:
    goal = obs.get("goal", obs.get("utterance", "Complete the task."))
    axtree = obs.get("axtree_txt", "(no accessibility tree)")
    last_error = obs.get("last_action_error", "")
    error_section = f"\nLast action error: {last_error}" if last_error else ""
    return (
        f"Task: {goal}\n\n"
        f"Accessibility tree:\n{axtree[:3000]}"  # cap to avoid token overflow
        f"{error_section}\n\n"
        f"{ACTION_FORMAT}"
    )


def screenshot_to_pil(obs: dict):
    """Convert BrowserGym screenshot (numpy array) to PIL Image."""
    import numpy as np
    from PIL import Image

    screenshot = obs.get("screenshot")
    if screenshot is None:
        return None
    if isinstance(screenshot, bytes):
        return Image.open(io.BytesIO(screenshot)).convert("RGB")
    # numpy array (H, W, 3)
    arr = np.asarray(screenshot, dtype=np.uint8)
    return Image.fromarray(arr)


def parse_action(raw: str) -> str:
    """Extract a BrowserGym action string from model output."""
    # Try to parse JSON object
    for match in re.finditer(r'\{[^{}]+\}', raw, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if "action" in obj:
                return obj["action"].strip()
        except json.JSONDecodeError:
            continue
    # Fallback: look for known action patterns directly
    for pattern in [r'(click\("[^"]+"\))', r'(fill\("[^"]+",\s*"[^"]*"\))',
                    r'(select_option\("[^"]+",\s*"[^"]*"\))', r'(check\("[^"]+"\))',
                    r'(noop\(\d*\))']:
        match = re.search(pattern, raw)
        if match:
            return match.group(1)
    return "noop(500)"


# ─────────────────────────────────────────────────────────────────────────────
# Policy: Ollama (quick testing)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OllamaPolicy:
    """
    Zero-shot policy via Ollama.
    Good for quick local testing. Cannot be used for GRPO training.
    """
    model: str = "qwen2.5vl:3b"
    host: str = "http://127.0.0.1:11434"
    use_images: bool = True
    name: str = field(init=False)
    _task_id: str = field(default="", init=False)

    def __post_init__(self):
        self.name = f"zero_shot_ollama_{self.model.replace(':', '_').replace('.', '_')}"

    def reset(self, task_id: str) -> None:
        self._task_id = task_id

    def act(self, obs: dict) -> str:
        import requests

        prompt = build_text_prompt(obs)
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}

        if self.use_images:
            import base64
            import numpy as np
            screenshot = obs.get("screenshot")
            if screenshot is not None:
                if isinstance(screenshot, bytes):
                    img_b64 = base64.b64encode(screenshot).decode()
                else:
                    from PIL import Image
                    pil = Image.fromarray(np.asarray(screenshot, dtype=np.uint8))
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                payload["images"] = [img_b64]

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return parse_action(raw)
        except Exception as exc:
            raise RuntimeError(f"Ollama error: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Policy: HuggingFace Qwen2.5-VL (research pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HuggingFaceVLPolicy:
    """
    Zero-shot policy using Qwen2.5-VL via HuggingFace transformers.

    This is the correct backend for the research pipeline:
    - Same code path for zero-shot eval, post-SFT eval, and post-GRPO eval
    - Supports loading LoRA checkpoints (lora_checkpoint arg)
    - Generates gradients during training (unlike Ollama)

    Args:
        model_id:         HuggingFace model ID, e.g. "Qwen/Qwen2.5-VL-3B-Instruct"
        lora_checkpoint:  Path to LoRA adapter dir (None = no adapter = zero-shot)
        use_images:       Whether to pass screenshots to the model
        max_new_tokens:   Max tokens to generate per action
        temperature:      Sampling temperature (0.0 = greedy)
        load_in_4bit:     Use bitsandbytes 4-bit quantization (saves ~8GB VRAM)
    """
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    lora_checkpoint: Optional[Path] = None
    use_images: bool = True
    max_new_tokens: int = 128
    temperature: float = 0.0
    load_in_4bit: bool = True
    name: str = field(init=False)
    _model: object = field(default=None, init=False, repr=False)
    _processor: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        suffix = "zero_shot" if self.lora_checkpoint is None else "lora"
        self.name = f"{suffix}_hf_{self.model_id.split('/')[-1].lower().replace('-', '_')}"

    def _load(self):
        """Lazy-load model + processor on first call."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import Qwen2_5_VLForConditionalGeneration

        print(f"[HFPolicy] Loading {self.model_id} (4bit={self.load_in_4bit}) ...")
        t0 = time.time()

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )

        if self.lora_checkpoint is not None:
            from peft import PeftModel
            print(f"[HFPolicy] Loading LoRA from {self.lora_checkpoint} ...")
            model = PeftModel.from_pretrained(model, str(self.lora_checkpoint))
            model = model.merge_and_unload()  # merge for faster inference

        model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = model
        print(f"[HFPolicy] Model loaded in {time.time() - t0:.1f}s")

    def reset(self, task_id: str) -> None:
        self._load()

    def act(self, obs: dict) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        goal = obs.get("goal", obs.get("utterance", "Complete the task."))
        axtree = obs.get("axtree_txt", "")
        last_error = obs.get("last_action_error", "")

        text_content = (
            f"Task: {goal}\n\n"
            f"Accessibility tree:\n{axtree[:3000]}"
        )
        if last_error:
            text_content += f"\nLast action error: {last_error}"
        text_content += f"\n\n{ACTION_FORMAT}"

        content = []
        if self.use_images:
            pil_img = screenshot_to_pil(obs)
            if pil_img is not None:
                content.append({"type": "image", "image": pil_img})

        content.append({"type": "text", "text": text_content})

        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        # Strip the input tokens from the output
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:]
        raw = self._processor.decode(output_ids, skip_special_tokens=True)
        return parse_action(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    policy,
    max_steps: int,
    episode_idx: int = 0,
) -> dict:
    """
    Run one episode of a MiniWoB task with the given policy.

    Returns a result dict containing: task_id, policy, success, reward,
    steps, rrr, trajectory (list of per-step dicts).
    """
    env = gym.make(
        task_id,
        headless=True,
        slow_mo=0,
    )

    task_name = task_id.replace("browsergym/miniwob.", "")
    policy.reset(task_id)

    trajectory: list[dict] = []
    reward = 0.0
    terminated = False
    truncated = False
    info: dict = {}

    try:
        obs, reset_info = env.reset()

        for step_idx in range(1, max_steps + 1):
            step_start = time.time()

            try:
                action = policy.act(obs)
            except RuntimeError as exc:
                # Model/Ollama error — log and abort episode
                trajectory.append({
                    "step": step_idx,
                    "action": "<model_error>",
                    "reward": 0.0,
                    "success": False,
                    "verifier_message": str(exc),
                    "last_action_error": str(exc),
                    "elapsed_ms": int((time.time() - step_start) * 1000),
                })
                break

            obs, reward, terminated, truncated, info = env.step(action)
            task_info = info.get("task_info", {})
            success_step = bool(task_info.get("success", False))

            trajectory.append({
                "step": step_idx,
                "action": action,
                "reward": float(reward),
                "success": success_step,
                "verifier_message": task_info.get("verifier_message", ""),
                "last_action_error": obs.get("last_action_error", ""),
                "elapsed_ms": int((time.time() - step_start) * 1000),
            })

            if terminated or truncated:
                break

    finally:
        env.close()

    task_info = info.get("task_info", {})
    success = bool(task_info.get("success", False))

    # MiniWoB rewards are 0.0–1.0 (some tasks give partial credit)
    # For binary SR reporting we threshold at 0.5
    binary_success = success or float(reward) >= 0.5

    steps = len(trajectory)
    optimal = OPTIMAL_STEPS.get(task_name)
    rrr = round((optimal / steps), 4) if (binary_success and steps and optimal) else 0.0

    return {
        "policy": policy.name,
        "task_id": task_id,
        "task_name": task_name,
        "episode": episode_idx,
        "success": binary_success,
        "raw_reward": float(reward),
        "steps": steps,
        "optimal_steps": optimal,
        "rrr": rrr,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "verifier_message": task_info.get("verifier_message", ""),
        "trajectory": trajectory,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def summarize(results: list[dict]) -> list[dict]:
    """Aggregate results by (policy, task_name)."""
    groups: dict[tuple, list] = {}
    for r in results:
        key = (r["policy"], r["task_name"])
        groups.setdefault(key, []).append(r)

    rows = []
    for (policy_name, task_name), group in sorted(groups.items()):
        n = len(group)
        sr = sum(r["success"] for r in group) / n
        rows.append({
            "policy": policy_name,
            "task": task_name,
            "episodes": n,
            "success_rate": round(sr, 4),
            "mean_reward": round(sum(r["raw_reward"] for r in group) / n, 4),
            "mean_steps": round(sum(r["steps"] for r in group) / n, 2),
            "mean_rrr": round(sum(r["rrr"] for r in group) / n, 4),
        })

    # Also add an overall row per policy
    for policy_name in sorted({r["policy"] for r in results}):
        group = [r for r in results if r["policy"] == policy_name]
        n = len(group)
        sr = sum(r["success"] for r in group) / n
        rows.append({
            "policy": policy_name,
            "task": "ALL",
            "episodes": n,
            "success_rate": round(sr, 4),
            "mean_reward": round(sum(r["raw_reward"] for r in group) / n, 4),
            "mean_steps": round(sum(r["steps"] for r in group) / n, 2),
            "mean_rrr": round(sum(r["rrr"] for r in group) / n, 4),
        })

    return rows


def print_summary(rows: list[dict]) -> None:
    headers = ["policy", "task", "episodes", "success_rate", "mean_reward", "mean_steps", "mean_rrr"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    sep = "-+-".join("-" * widths[h] for h in headers)
    header_row = " | ".join(h.ljust(widths[h]) for h in headers)
    print(header_row)
    print(sep)

    prev_policy = None
    for row in rows:
        if prev_policy and row["policy"] != prev_policy:
            print(sep)
        prev_policy = row["policy"]
        is_total = row["task"] == "ALL"
        line = " | ".join(str(row[h]).ljust(widths[h]) for h in headers)
        print(("**" if is_total else "  ") + line.lstrip())


def write_artifacts(results: list[dict], summary: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Episode-level CSV (no trajectory column)
    episode_fields = [
        "policy", "task_id", "task_name", "episode",
        "success", "raw_reward", "steps", "optimal_steps", "rrr",
        "terminated", "truncated", "verifier_message",
    ]
    with (out_dir / "episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in episode_fields})

    # Summary CSV
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # Full trajectories JSONL (includes per-step actions + rewards)
    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Per-task SR breakdown (useful for picking training tasks)
    task_rows = [r for r in summary if r["task"] != "ALL"]
    with (out_dir / "per_task_sr.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)

    print(f"\nArtifacts written to {out_dir.resolve()}/")
    print(f"  episode_results.csv   — per-episode outcomes")
    print(f"  summary.csv           — aggregated SR / reward / steps")
    print(f"  per_task_sr.csv       — per-task breakdown")
    print(f"  trajectories.jsonl    — full step-by-step logs")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot MiniWoB++ evaluation for Qwen2.5-VL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task selection
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated MiniWoB task names (without 'browsergym/miniwob.' prefix). "
             "Use 'tier1', 'tier2', or 'tier3' for preset lists.",
    )
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per task (use ≥5 for reliable SR estimates).")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Max browser actions per episode.")

    # Backend
    parser.add_argument(
        "--backend", choices=["ollama", "hf"], default="hf",
        help="Model backend. 'ollama' for quick local testing. "
             "'hf' for research pipeline (supports LoRA checkpoints).",
    )

    # Ollama options
    parser.add_argument("--ollama-model", default="qwen2.5vl:3b")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))

    # HuggingFace options
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace model ID.")
    parser.add_argument("--lora-checkpoint", type=Path, default=None,
                        help="Path to LoRA adapter directory (for post-SFT or post-GRPO eval).")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (needs more VRAM, but more accurate).")
    parser.add_argument("--no-images", action="store_true",
                        help="Text-only mode: don't pass screenshots to model.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0.0 = greedy (deterministic).")
    parser.add_argument("--max-new-tokens", type=int, default=128)

    # Output
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "miniwob_zero_shot",
                        help="Output directory for CSV and JSONL results.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step logging.")

    args = parser.parse_args()

    # Resolve task list
    preset_map = {"tier1": TIER1_TASKS, "tier2": TIER2_TASKS, "tier3": TIER3_TASKS}
    if args.tasks in preset_map:
        task_names = preset_map[args.tasks]
    else:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    task_ids = [f"browsergym/miniwob.{name}" for name in task_names]

    # Set Playwright browser path if not already set
    playwright_path = Path(".playwright").resolve()
    if playwright_path.exists():
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(playwright_path))

    # Set MiniWoB URL if not set
    if "MINIWOB_URL" not in os.environ:
        os.environ["MINIWOB_URL"] = "http://127.0.0.1:8000/"
        print("[Warning] MINIWOB_URL not set. Defaulting to http://127.0.0.1:8000/")
        print("          Start MiniWoB HTML server with:")
        print("          python -m http.server 8000 --directory miniwob-plusplus/miniwob/html")

    # Build policy
    if args.backend == "ollama":
        policy = OllamaPolicy(
            model=args.ollama_model,
            host=args.ollama_host,
            use_images=not args.no_images,
        )
    else:
        policy = HuggingFaceVLPolicy(
            model_id=args.hf_model,
            lora_checkpoint=args.lora_checkpoint,
            use_images=not args.no_images,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            load_in_4bit=not args.no_4bit,
        )

    print(f"\n{'='*60}")
    print(f"Policy:   {policy.name}")
    print(f"Backend:  {args.backend}")
    print(f"Tasks:    {len(task_ids)}")
    print(f"Episodes: {args.episodes} per task  ({len(task_ids) * args.episodes} total)")
    print(f"Max steps: {args.max_steps}")
    print(f"Output:   {args.out}")
    print(f"{'='*60}\n")

    # Run evaluation
    results: list[dict] = []
    total_episodes = len(task_ids) * args.episodes

    for task_id in task_ids:
        task_name = task_id.replace("browsergym/miniwob.", "")
        task_successes = 0

        for ep_idx in range(args.episodes):
            ep_num = ep_idx + 1
            if not args.quiet:
                print(f"[{len(results)+1:3d}/{total_episodes}] {task_name} ep={ep_num} ...", end=" ", flush=True)

            t0 = time.time()
            result = run_episode(task_id, policy, max_steps=args.max_steps, episode_idx=ep_idx)
            elapsed = time.time() - t0

            results.append(result)
            task_successes += result["success"]

            if not args.quiet:
                status = "✓" if result["success"] else "✗"
                print(
                    f"{status}  reward={result['raw_reward']:.2f}  "
                    f"steps={result['steps']}  rrr={result['rrr']}  "
                    f"({elapsed:.1f}s)"
                )

        sr = task_successes / args.episodes
        print(f"  → {task_name}: SR={sr:.0%} ({task_successes}/{args.episodes})")

    # Summarize and write
    summary = summarize(results)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print_summary(summary)

    write_artifacts(results, summary, args.out)


if __name__ == "__main__":
    main()
