from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from social_rlvr_web.observation import axtree_to_text, dom_to_text, screenshot_to_base64_png


ACTION_RE = re.compile(
    r"^(click|fill|select_option|noop|send_msg_to_user|scroll|keyboard_press|press)\(.*\)$"
)


class ModelPolicyError(RuntimeError):
    pass


@dataclass
class OllamaVisionPolicy:
    name: str = ""
    model: str = "qwen2.5vl"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.2
    num_ctx: int = 2048
    num_predict: int = 512
    timeout_seconds: int = 600
    use_images: bool = True
    task_id: str = ""
    step: int = 0

    def __post_init__(self) -> None:
        self.host = os.environ.get("OLLAMA_HOST", self.host).rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", self.model)
        if not self.name:
            safe_model = self.model.replace(":", "_").replace(".", "_")
            modality = "vision" if self.use_images else "text"
            self.name = f"{safe_model}_ollama_{modality}_zero_shot"

    def reset(self, task_id: str) -> None:
        self.task_id = task_id
        self.step = 0

    def act(self, env, obs: dict[str, Any]) -> str:
        del env
        self.step += 1
        prompt = self._build_prompt(obs)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if self.use_images:
            payload["images"] = [screenshot_to_base64_png(obs["screenshot"], max_width=320)]
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise ModelPolicyError(
                f"Ollama model call timed out after {self.timeout_seconds}s for {self.model}."
            ) from exc
        except requests.RequestException as exc:
            raise ModelPolicyError(
                f"Could not reach Ollama at {self.host}. Install/start Ollama and pull {self.model}."
            ) from exc
        if response.status_code != 200:
            raise ModelPolicyError(f"Ollama returned {response.status_code}: {response.text[:500]}")

        data = response.json()
        raw = data.get("response", "")
        print(f"[DEBUG RAW MODEL OUTPUT]: {raw[:400]}")
        action = self._parse_action(raw)
        print(f"[DEBUG PARSED ACTION]: {action}")
        if action is None:
            return "noop(100)"
        return action

    def _build_prompt(self, obs: dict[str, Any]) -> str:
        goal = obs.get("goal") or obs.get("instruction", "")
        last_error = obs.get("last_action_error", "")

        axtree = axtree_to_text(obs.get("axtree_object"), max_lines=80)
        if not axtree.strip():
            axtree = obs.get("accessibility_tree", "")

        dom = dom_to_text(obs.get("dom_object"), max_lines=100)
        if not dom.strip():
            raw_dom = obs.get("dom", [])
            if isinstance(raw_dom, list):
                dom = "\n".join(
                    f'[{el.get("index")}] {el.get("tag")} name="{el.get("name")}" text="{el.get("text")}"'
                    for el in raw_dom[:100]
                )

        valid_bids = self._valid_bid_lines(obs.get("axtree_object"))
        if not valid_bids.strip():
            valid_bids = self._valid_bid_lines_from_dom(obs.get("dom", []))

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
- Never output literal placeholder strings like click("bid") or fill("bid", "text").
- Do not say the task is done unless the page state was actually changed.
- Prefer one concrete UI action per turn.
- For select dropdowns, use select_option.
- For text fields, use fill.
- For checkboxes and buttons, use click.
- For select_option, use only the plain option text exactly as shown in the DOM (e.g. "Riya" not "5. Riya").
- For select_option, use the exact visible option text (e.g. "Fri 14:00" not "fri-14" or "3").
- For checkboxes, use click("bid") to toggle them.
- Read the availability table carefully before picking a slot - choose the slot where ALL required attendees show "Open".

Last action error:
{last_error}

Valid element IDs:
{valid_bids}

Accessibility tree:
{axtree}

DOM summary:
{dom}
""".strip()

    @staticmethod
    def _valid_bid_lines(axtree: Any) -> str:
        if not isinstance(axtree, dict):
            return ""
        INTERACTIVE = {
            "textbox", "button", "combobox", "menuitem", "link",
            "checkbox", "radio", "spinbutton", "switch",
        }
        seen_bids = set()
        lines = []
        for node in axtree.get("nodes", []):
            if not isinstance(node, dict) or "browsergym_id" not in node:
                continue
            role = node.get("role", {})
            name = node.get("name", {})
            role_value = role.get("value", "") if isinstance(role, dict) else str(role)
            name_value = (name.get("value", "") if isinstance(name, dict) else str(name)).strip()
            bid = node["browsergym_id"]
            if bid in seen_bids:
                continue
            # include checked state for checkboxes
            props = node.get("properties", [])
            checked = next(
                (p.get("value", {}).get("value") for p in props if p.get("name") == "checked"),
                None,
            )
            val = node.get("value", {})
            val_v = (val.get("value", "") if isinstance(val, dict) else str(val) if val else "").strip()
            if role_value in INTERACTIVE:
                extra = ""
                if checked is not None:
                    extra += f" checked={checked}"
                if val_v:
                    extra += f" value=\"{val_v}\""
                lines.append(f'- bid="{bid}" role={role_value} name="{name_value}"{extra}')
                seen_bids.add(bid)
        return "\n".join(lines[:80])
    
    @staticmethod
    def _valid_bid_lines_from_dom(dom: list) -> str:
        if not isinstance(dom, list):
            return ""
        lines = []
        for el in dom[:80]:
            idx = el.get("index", "")
            tag = el.get("tag", "")
            name = el.get("name", "")
            role = el.get("role", "")
            if tag in {"input", "button", "select", "textarea", "a"} or name:
                lines.append(f'- bid="{idx}" role={role} name="{name}"')
        return "\n".join(lines)

    @staticmethod
    def _parse_action(raw: str) -> str | None:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0] if data else {}
            action = str(data.get("action", "")).strip()
            if ACTION_RE.match(action):
                return action
        except json.JSONDecodeError:
            pass

        match = re.search(r'"action"\s*:\s*"(.+)"[\s}]*$', raw, re.DOTALL)
        if match:
            action = match.group(1).strip().rstrip('"').rstrip('}').rstrip('"').strip()
            if ACTION_RE.match(action):
                return action

        direct = re.search(
            r'(click|fill|select_option|noop|scroll|press|keyboard_press)\([^`]{1,100}\)',
            raw,
        )
        if direct:
            action = direct.group(0).strip()
            if ACTION_RE.match(action):
                return action

        return None
