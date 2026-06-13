"""
STEP 1 - Run this on your LOCAL machine first.
Collects (prompt, action, reward) samples from your task server
and saves them as a dataset JSON file to upload to Colab.

Run:
    .\.conda\python.exe scripts\collect_grpo_dataset.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(".playwright").resolve()))

import gymnasium as gym
import social_rlvr_web.browsergym_tasks  # noqa: F401
from social_rlvr_web.model_policy import OllamaVisionPolicy
from social_rlvr_web.observation import axtree_to_text, dom_to_text, screenshot_to_base64_png


TASK_IDS = [
    "browsergym/social_rlvr.report.extract_tracking_code",
    "browsergym/social_rlvr.gallery.aesthetic_travel_to_meera",
    "browsergym/social_rlvr.orders.priority_followup",
    "browsergym/social_rlvr.schedule.design_review_shared_slot",
    "browsergym/social_rlvr.messages.last_five_new_year",
]

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:latest")
REPEATS_PER_TASK = 5       # how many rollouts per task
MODEL_STEPS = 8            # max steps per episode
OUT_FILE = Path("artifacts/grpo_dataset/training_data.json")


def build_prompt_text(obs: dict) -> str:
    """Build the text prompt exactly as model_policy does."""
    goal = obs.get("goal") or obs.get("instruction", "")
    last_error = obs.get("last_action_error", "")

    axtree = axtree_to_text(obs.get("axtree_object"), max_lines=80)
    if not axtree.strip():
        axtree = obs.get("accessibility_tree", "")

    dom = dom_to_text(obs.get("dom_object"), max_lines=100)

    # build valid bid lines
    valid_bids = _valid_bid_lines(obs.get("axtree_object"))

    return f"""IMPORTANT: You must respond with ONLY a single JSON object. No markdown, no bullets, no explanation.
Example of the ONLY valid response format: {{"action": "click(\\"57\\")"}}

You are controlling a BrowserGym web task.

Goal:
{goal}

Return exactly one next action as JSON:
{{"action": "click(\\"bid\\")"}}

Allowed action formats:
- click("17")
- fill("16", "text")
- select_option("12", "visible option text")
- noop(100)

Rules:
- Use only numeric bid values from the valid element IDs below.
- Never output literal placeholder strings.
- Prefer one concrete UI action per turn.
- For select dropdowns, use select_option on the COMBOBOX bid not the option bid.
- For text fields, use fill. For checkboxes and buttons, use click.

Last action error: {last_error}

Valid element IDs:
{valid_bids}

Accessibility tree:
{axtree}

DOM summary:
{dom}""".strip()


def _valid_bid_lines(axtree) -> str:
    if not isinstance(axtree, dict):
        return ""
    INTERACTIVE = {"textbox", "button", "combobox", "menuitem", "link",
                   "checkbox", "radio", "spinbutton", "switch"}
    seen = set()
    lines = []
    for node in axtree.get("nodes", []):
        if not isinstance(node, dict) or "browsergym_id" not in node:
            continue
        role = node.get("role", {})
        name = node.get("name", {})
        role_v = role.get("value", "") if isinstance(role, dict) else str(role)
        name_v = (name.get("value", "") if isinstance(name, dict) else str(name)).strip()
        bid = node["browsergym_id"]
        if bid in seen:
            continue
        if role_v in INTERACTIVE:
            lines.append(f'- bid="{bid}" role={role_v} name="{name_v}"')
            seen.add(bid)
    return "\n".join(lines[:80])


def collect_episode(task_id: str, policy: OllamaVisionPolicy) -> list[dict]:
    """Run one episode and collect (prompt, response, reward) for every step."""
    samples = []
    env = gym.make(task_id, headless=True, slow_mo=0, pre_observation_delay=0.05)
    policy.reset(task_id)
    try:
        obs, _ = env.reset()
        for step in range(1, MODEL_STEPS + 1):
            prompt = build_prompt_text(obs)

            # get raw model response directly
            import requests as req
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.8, "num_ctx": 2048, "num_predict": 512},
            }
            # include screenshot
            try:
                payload["images"] = [screenshot_to_base64_png(obs["screenshot"], max_width=320)]
            except Exception:
                pass

            try:
                resp = req.post("http://127.0.0.1:11434/api/generate",
                                json=payload, timeout=120)
                raw_response = resp.json().get("response", "noop(100)")
            except Exception as e:
                raw_response = "noop(100)"
                print(f"  Model call failed: {e}")

            # parse action
            action = policy._parse_action(raw_response) or "noop(100)"

            # step environment
            obs, reward, terminated, truncated, info = env.step(action)

            samples.append({
                "task_id": task_id,
                "step": step,
                "prompt": prompt,
                "response": raw_response,
                "action": action,
                "reward": float(reward),
                "success": bool(info.get("task_info", {}).get("success", False)),
                "verifier_message": info.get("task_info", {}).get("verifier_message", ""),
            })

            print(f"  step {step}: {action[:60]} | reward={reward}")

            if terminated or truncated:
                break
    finally:
        env.close()
    return samples


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    policy = OllamaVisionPolicy(model=OLLAMA_MODEL, use_images=True)
    all_samples = []

    for task_id in TASK_IDS:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        for repeat in range(REPEATS_PER_TASK):
            print(f"  Repeat {repeat + 1}/{REPEATS_PER_TASK}")
            try:
                samples = collect_episode(task_id, policy)
                all_samples.extend(samples)
                successes = sum(1 for s in samples if s["reward"] > 0)
                print(f"  Episode done: {len(samples)} steps, {successes} rewarded steps")
            except Exception as e:
                print(f"  Episode failed: {e}")

    OUT_FILE.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")
    print(f"\nCollected {len(all_samples)} samples")
    print(f"Saved to {OUT_FILE.resolve()}")

    # print summary
    from collections import Counter
    task_counts = Counter(s["task_id"].split(".")[-1] for s in all_samples)
    reward_counts = Counter(s["reward"] for s in all_samples)
    print(f"\nSamples per task: {dict(task_counts)}")
    print(f"Reward distribution: {dict(reward_counts)}")


if __name__ == "__main__":
    main()
