from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol

import gymnasium as gym

import social_rlvr_web.browsergym_tasks  # noqa: F401
from social_rlvr_web.rlvr_policy import LearnedTrajectoryPolicy, RLVRPolicyError


TASK_IDS = [
    "browsergym/social_rlvr.report.extract_tracking_code",
    "browsergym/social_rlvr.gallery.aesthetic_travel_to_meera",
    "browsergym/social_rlvr.messages.last_five_new_year",
    "browsergym/social_rlvr.orders.priority_followup",
    "browsergym/social_rlvr.schedule.design_review_shared_slot",
]

OPTIMAL_STEPS = {
    "browsergym/social_rlvr.report.extract_tracking_code": 2,
    "browsergym/social_rlvr.gallery.aesthetic_travel_to_meera": 2,
    "browsergym/social_rlvr.messages.last_five_new_year": 15,
    "browsergym/social_rlvr.orders.priority_followup": 4,
    "browsergym/social_rlvr.schedule.design_review_shared_slot": 5,
}

# ── BrowserGym action space ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a browser automation agent. You interact with web pages by
outputting exactly one action per turn using the BrowserGym action syntax.

Available actions:
  click("bid")                    - click an element by its bid attribute
  fill("bid", "text")             - type text into an input field
  select_option("bid", "value")   - select a dropdown option
  send_msg_to_user("message")     - send a final answer/message
  noop(ms)                        - wait for ms milliseconds

Rules:
- Output ONLY the action string, nothing else.
- Do not add explanation, markdown, or code fences.
- Use bid values from the accessibility tree provided.
- If the task is complete, use send_msg_to_user with your answer.

Example output:
fill("bid_23", "hello world")
"""


def _obs_to_prompt(obs: dict, task_id: str) -> tuple[str, str | None]:
    """
    Convert a BrowserGym observation to (text_prompt, base64_image_or_None).
    Uses the accessibility tree as primary input; screenshot as secondary visual.
    """
    axtree = obs.get("axtree_txt", "")
    last_error = obs.get("last_action_error", "")
    goal = obs.get("goal", task_id)

    text = f"Task: {goal}\n\n"
    if last_error:
        text += f"Last action error: {last_error}\n\n"
    text += f"Current page accessibility tree:\n{axtree}\n\nWhat is your next action?"

    # encode screenshot if available
    screenshot = obs.get("screenshot")
    b64_image = None
    if screenshot is not None:
        try:
            from PIL import Image
            if isinstance(screenshot, bytes):
                img = Image.open(BytesIO(screenshot))
            else:
                # numpy array from playwright
                import numpy as np
                img = Image.fromarray(screenshot.astype("uint8"))
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64_image = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            b64_image = None

    return text, b64_image


class ModelPolicyError(Exception):
    pass


class Policy(Protocol):
    name: str

    def reset(self, task_id: str) -> None: ...

    def act(self, env, obs: dict) -> str: ...


# ── Hallucination baseline (unchanged) ──────────────────────────────────────

@dataclass
class HallucinationBaseline:
    """A deliberately weak pre-RLVR baseline: claims completion without state change."""

    name: str = "before_rlvr_hallucination_baseline"
    step: int = 0

    def reset(self, task_id: str) -> None:
        self.step = 0

    def act(self, env, obs: dict) -> str:
        self.step += 1
        if self.step == 1:
            return 'send_msg_to_user("Done")'
        return "noop(100)"


# ── Scripted oracle (unchanged) ──────────────────────────────────────────────

def bid_for(page, selector: str, index: int = 0) -> str:
    bid = page.locator(selector).nth(index).get_attribute("bid")
    if not bid:
        raise LookupError(f"No BrowserGym bid found for {selector} at index {index}")
    return bid


def bid_for_value(page, selector: str, value: str) -> str:
    count = page.locator(selector).count()
    for index in range(count):
        locator = page.locator(selector).nth(index)
        if locator.get_attribute("value") == value:
            bid = locator.get_attribute("bid")
            if bid:
                return bid
    raise LookupError(f"No BrowserGym bid found for {selector} with value {value!r}")


@dataclass
class VerifierSelectedPolicy:
    """Successful trajectories selected by the verifier reward."""

    name: str = "after_rlvr_verifier_selected"
    task_id: str = ""
    cursor: int = 0
    recipients: list[str] = field(default_factory=lambda: ["Kabir", "Meera", "Riya", "Vivaan", "Zara"])

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.cursor = 0

    def act(self, env, obs: dict) -> str:
        page = env.unwrapped.page
        if self.task_id.endswith("report.extract_tracking_code"):
            if self.cursor == 0:
                self.cursor += 1
                return f'fill("{bid_for(page, "input")}", "TRV-8429-IN")'
            self.cursor += 1
            return f'click("{bid_for(page, "button")}")'

        if self.task_id.endswith("gallery.aesthetic_travel_to_meera"):
            if self.cursor == 0:
                self.cursor += 1
                return f'select_option("{bid_for(page, "select", 1)}", "Meera")'
            self.cursor += 1
            return f'click("{bid_for(page, "button", 1)}")'

        if self.task_id.endswith("messages.last_five_new_year"):
            recipient = self.recipients[self.cursor // 3]
            phase = self.cursor % 3
            self.cursor += 1
            if phase == 0:
                return f'select_option("{bid_for(page, "select")}", "{recipient}")'
            if phase == 1:
                return f'fill("{bid_for(page, "textarea")}", "Happy New Year")'
            return f'click("{bid_for(page, "button")}")'

        if self.task_id.endswith("orders.priority_followup"):
            if self.cursor == 0:
                self.cursor += 1
                return f'select_option("{bid_for(page, "select")}", "Meera")'
            if self.cursor == 1:
                self.cursor += 1
                return f'fill("{bid_for(page, "input")}", "REF-MEERA-774")'
            if self.cursor == 2:
                self.cursor += 1
                return f'fill("{bid_for(page, "textarea")}", "Priority follow-up")'
            self.cursor += 1
            return f'click("{bid_for(page, "button")}")'

        if self.task_id.endswith("schedule.design_review_shared_slot"):
            if self.cursor == 0:
                self.cursor += 1
                return f'fill("{bid_for(page, "input")}", "Design review")'
            if self.cursor == 1:
                self.cursor += 1
                return f'select_option("{bid_for(page, "select")}", "Fri 14:00")'
            if self.cursor == 2:
                self.cursor += 1
                return f'click("{bid_for_value(page, "input[type=checkbox]", "Kabir")}")'
            if self.cursor == 3:
                self.cursor += 1
                return f'click("{bid_for_value(page, "input[type=checkbox]", "Zara")}")'
            self.cursor += 1
            return f'click("{bid_for(page, "button")}")'

        raise KeyError(f"No policy for {self.task_id}")


# ── ZeroShotVLMPolicy — THE REAL ZERO-SHOT POLICY ───────────────────────────

@dataclass
class ZeroShotVLMPolicy:
    """
    Real zero-shot policy: Qwen2.5-VL-3B-Instruct generates actions directly
    from screenshots + accessibility tree. No training, no fine-tuning.
    This is the genuine 'before RLVR' baseline.
    """

    name: str = "zero_shot_qwen25vl_3b"
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    task_id: str = ""

    # set after __post_init__
    model: object = field(default=None, repr=False)
    processor: object = field(default=None, repr=False)

    def __post_init__(self):
        # lazy import so script can still be imported without torch
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        print(f"[ZeroShotVLMPolicy] Loading {self.model_name} ...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",          # spreads across both A6000s automatically
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[ZeroShotVLMPolicy] Model loaded on {next(self.model.parameters()).device}")

    def reset(self, task_id: str) -> None:
        self.task_id = task_id

    def act(self, env, obs: dict) -> str:
        import torch

        text_prompt, b64_image = _obs_to_prompt(obs, self.task_id)

        # build message in Qwen2-VL chat format
        if b64_image is not None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image/png;base64,{b64_image}",
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                },
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_prompt},
            ]

        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # processor handles image + text together
            inputs = self.processor(
                text=[text],
                images=[f"data:image/png;base64,{b64_image}"] if b64_image else None,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,        # greedy for reproducibility
                    temperature=None,
                    top_p=None,
                )

            # decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[0][input_len:]
            raw = self.processor.decode(generated, skip_special_tokens=True).strip()

            action = self._parse_action(raw)
            return action

        except Exception as exc:
            raise ModelPolicyError(f"Qwen inference failed: {exc}") from exc

    @staticmethod
    def _parse_action(raw: str) -> str:
        """
        Extract a valid BrowserGym action from model output.
        Model should output just the action, but sometimes adds extra text.
        """
        raw = raw.strip()

        # strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

        # known action prefixes
        valid_prefixes = (
            "click(",
            "fill(",
            "select_option(",
            "send_msg_to_user(",
            "noop(",
            "scroll(",
            "hover(",
            "press(",
            "go_back(",
            "go_forward(",
            "goto(",
        )

        # if first line is a valid action, use it
        first_line = raw.split("\n")[0].strip()
        if any(first_line.startswith(p) for p in valid_prefixes):
            return first_line

        # scan all lines for an action
        for line in raw.split("\n"):
            line = line.strip()
            if any(line.startswith(p) for p in valid_prefixes):
                return line

        # fallback: return noop and log the bad output
        print(f"[ZeroShotVLMPolicy] Could not parse action from: {raw!r}")
        return "noop(500)"


# ── GRPOFinetunedPolicy placeholder ─────────────────────────────────────────

@dataclass
class GRPOFinetunedPolicy:
    """
    After GRPO training — same Qwen2.5-VL-3B base with LoRA adapter loaded.
    Swap in after train_grpo.py produces a checkpoint.
    """

    name: str = "grpo_qwen25vl_3b"
    base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_path: str = "artifacts/grpo_checkpoint"
    task_id: str = ""

    model: object = field(default=None, repr=False)
    processor: object = field(default=None, repr=False)

    def __post_init__(self):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        print(f"[GRPOFinetunedPolicy] Loading base + LoRA from {self.adapter_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()
        print("[GRPOFinetunedPolicy] Model + adapter loaded")

    def reset(self, task_id: str) -> None:
        self.task_id = task_id

    def act(self, env, obs: dict) -> str:
        # identical inference to ZeroShotVLMPolicy
        # reuse the same logic via composition
        return ZeroShotVLMPolicy.act(self, env, obs)


# ── Episode runner (unchanged from original) ────────────────────────────────

def run_episode(task_id: str, policy: Policy, max_eval_steps: int) -> dict:
    env = gym.make(
        task_id,
        headless=True,
        slow_mo=0,
        pre_observation_delay=0.05,
    )
    policy.reset(task_id)
    trajectory = []
    reward = 0.0
    terminated = False
    truncated = False
    info = {"task_info": {"success": False, "verifier_message": "not evaluated"}}
    try:
        obs, reset_info = env.reset()
        for step_idx in range(1, max_eval_steps + 1):
            try:
                action = policy.act(env, obs)
            except (ModelPolicyError, RLVRPolicyError) as exc:
                trajectory.append(
                    {
                        "step": step_idx,
                        "action": "<model_error>",
                        "reward": 0.0,
                        "success": False,
                        "verifier_message": str(exc),
                        "last_action_error": str(exc),
                    }
                )
                info = {"task_info": {"success": False, "verifier_message": str(exc)}}
                break
            obs, reward, terminated, truncated, info = env.step(action)
            task_info = info.get("task_info", {})
            trajectory.append(
                {
                    "step": step_idx,
                    "action": action,
                    "reward": reward,
                    "success": task_info.get("success", False),
                    "verifier_message": task_info.get("verifier_message", ""),
                    "last_action_error": obs.get("last_action_error", ""),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    task_info = info.get("task_info", {})
    success = bool(task_info.get("success", False))
    steps = len(trajectory)
    optimal = OPTIMAL_STEPS[task_id]
    rrr = (optimal / steps) if success and steps else 0.0
    return {
        "policy": policy.name,
        "task_id": task_id,
        "success": success,
        "reward": float(reward),
        "steps": steps,
        "optimal_steps": optimal,
        "rrr": round(rrr, 4),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "verifier_message": task_info.get("verifier_message", ""),
        "trajectory": trajectory,
    }


# ── Metrics (unchanged) ──────────────────────────────────────────────────────

def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for policy_name in sorted({row["policy"] for row in rows}):
        subset = [row for row in rows if row["policy"] == policy_name]
        summaries.append(
            {
                "policy": policy_name,
                "episodes": len(subset),
                "success_rate": round(sum(row["success"] for row in subset) / len(subset), 4),
                "mean_reward": round(sum(row["reward"] for row in subset) / len(subset), 4),
                "mean_steps": round(sum(row["steps"] for row in subset) / len(subset), 2),
                "mean_rrr": round(sum(row["rrr"] for row in subset) / len(subset), 4),
            }
        )
    return summaries


def print_table(rows: list[dict]) -> None:
    headers = ["policy", "episodes", "success_rate", "mean_reward", "mean_steps", "mean_rrr"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def write_artifacts(results: list[dict], summary: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episode_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "policy", "task_id", "success", "reward", "steps",
                "optimal_steps", "rrr", "terminated", "truncated", "verifier_message",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    with (out_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--baseline-steps", type=int, default=3)
    parser.add_argument("--model-steps", type=int, default=20)
    parser.add_argument(
        "--policies",
        default="baseline,zero_shot,scripted",
        help="Comma-separated: baseline, zero_shot, scripted, grpo, rlvr",
    )
    parser.add_argument(
        "--grpo-adapter",
        type=str,
        default="artifacts/grpo_checkpoint",
        help="Path to LoRA adapter for grpo policy.",
    )
    parser.add_argument(
        "--rlvr-policy",
        type=Path,
        default=Path("artifacts") / "rlvr_training" / "learned_policy.json",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(TASK_IDS),
        help="Comma-separated BrowserGym task ids to evaluate.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "eval_browsergym_rlvr")
    args = parser.parse_args()

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(".playwright").resolve()))

    # zero_shot and grpo are loaded lazily so non-GPU runs still work
    _zero_shot_policy = None
    _grpo_policy = None

    def get_zero_shot():
        nonlocal _zero_shot_policy
        if _zero_shot_policy is None:
            _zero_shot_policy = ZeroShotVLMPolicy()
        return _zero_shot_policy

    def get_grpo():
        nonlocal _grpo_policy
        if _grpo_policy is None:
            _grpo_policy = GRPOFinetunedPolicy(adapter_path=args.grpo_adapter)
        return _grpo_policy

    policy_factories = {
        "baseline": lambda: HallucinationBaseline(),
        "scripted":  lambda: VerifierSelectedPolicy(),
        "zero_shot": get_zero_shot,
        "grpo":      get_grpo,
        "rlvr":      lambda: LearnedTrajectoryPolicy(args.rlvr_policy),
    }

    policies: list[Policy] = []
    for policy_name in [p.strip() for p in args.policies.split(",") if p.strip()]:
        if policy_name not in policy_factories:
            raise SystemExit(
                f"Unknown policy {policy_name!r}. Choose from: {sorted(policy_factories)}"
            )
        policies.append(policy_factories[policy_name]())

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(task_ids) - set(TASK_IDS)
    if unknown:
        raise SystemExit(f"Unknown task ids: {sorted(unknown)}")

    results = []
    for _ in range(args.repeats):
        for policy in policies:
            for task_id in task_ids:
                if isinstance(policy, HallucinationBaseline):
                    max_steps = args.baseline_steps
                elif isinstance(policy, (VerifierSelectedPolicy, LearnedTrajectoryPolicy)):
                    max_steps = OPTIMAL_STEPS[task_id]
                else:
                    max_steps = args.model_steps
                result = run_episode(task_id, policy, max_steps)
                results.append(result)
                print(
                    f"{policy.name} | {task_id} | "
                    f"success={result['success']} rrr={result['rrr']} | "
                    f"{result['verifier_message']}"
                )

    summary = summarize(results)
    print()
    print_table(summary)
    write_artifacts(results, summary, args.out)
    print()
    print(f"Wrote metrics to {args.out.resolve()}")


if __name__ == "__main__":
    main()


