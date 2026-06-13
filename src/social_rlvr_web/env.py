from __future__ import annotations

import re
from typing import Any

import gymnasium as gym
import requests
from gymnasium import spaces
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from social_rlvr_web.local_app import LocalAppServer
from social_rlvr_web.tasks import TASKS, TaskSpec, fetch_state, reset_state


ACTION_RE = re.compile(r"^(?P<name>click|fill|select|press|noop)\((?P<args>.*)\)$")


class BrowserRLVREnv(gym.Env):
    """Small Playwright env with verifier-only rewards.

    Actions are strings:
    - click(3)
    - fill(4, "Happy New Year")
    - select(2, "Meera")
    - press("Enter")
    - noop()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task_id: str, headless: bool = True, port: int = 8765):
        if task_id not in TASKS:
            raise KeyError(f"Unknown task_id {task_id!r}. Available: {sorted(TASKS)}")
        self.task: TaskSpec = TASKS[task_id]
        self.server = LocalAppServer(port=port)
        self.base_url = self.server.base_url
        self.headless = headless
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None
        self.steps = 0
        self.action_space = spaces.Text(max_length=256)
        self.observation_space = spaces.Dict(
            {
                "instruction": spaces.Text(max_length=512),
                "url": spaces.Text(max_length=2048),
                "dom": spaces.Sequence(spaces.Dict({})),
                "accessibility_tree": spaces.Text(max_length=16000),
                "screenshot": spaces.Sequence(spaces.Discrete(256)),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.server.start()
        self._wait_for_server()
        reset_state(self.base_url)
        self._ensure_browser()
        assert self.page is not None
        self.steps = 0
        self.page.goto(f"{self.base_url}{self.task.start_path}", wait_until="networkidle")
        return self._observe(), {"task_id": self.task.task_id}

    def step(self, action: str):
        self.steps += 1
        error = None
        try:
            self._apply_action(action)
        except Exception as exc:  # noqa: BLE001 - keep failed UI actions in trajectory info
            error = str(exc)

        state = fetch_state(self.base_url)
        success, verifier_message = self.task.verifier(state)
        terminated = success
        truncated = self.steps >= self.task.max_steps
        reward = 1.0 if success else 0.0
        info = {
            "success": success,
            "verifier_message": verifier_message,
            "state": state,
            "action_error": error,
        }
        return self._observe(), reward, terminated, truncated, info

    def close(self):
        if self.browser is not None:
            self.browser.close()
            self.browser = None
        if self.playwright is not None:
            self.playwright.stop()
            self.playwright = None
        self.server.stop()

    def _wait_for_server(self) -> None:
        response = requests.get(self.base_url, timeout=5)
        response.raise_for_status()

    def _ensure_browser(self) -> None:
        if self.playwright is None:
            self.playwright = sync_playwright().start()
        if self.browser is None:
            self.browser = self.playwright.chromium.launch(headless=self.headless)
        if self.page is None:
            self.page = self.browser.new_page(viewport={"width": 1280, "height": 900})

    def _observe(self) -> dict[str, Any]:
        assert self.page is not None
        elements = self._mark_and_collect_elements()
        screenshot = self.page.screenshot(type="png", full_page=True)
        tree_lines = [
            f'[{item["index"]}] {item["tag"]} role={item["role"]} name="{item["name"]}" '
            f'text="{item["text"]}" selector={item["selector"]}'
            for item in elements
        ]
        return {
            "instruction": self.task.instruction,
            "url": self.page.url,
            "dom": elements,
            "accessibility_tree": "\n".join(tree_lines),
            "screenshot": list(screenshot),
        }

    def _mark_and_collect_elements(self) -> list[dict[str, Any]]:
        assert self.page is not None
        return self.page.evaluate(
            """
            () => {
              const selector = [
                'a', 'button', 'input', 'textarea', 'select',
                '[role="button"]', '[tabindex]:not([tabindex="-1"])'
              ].join(',');
              const nodes = Array.from(document.querySelectorAll(selector))
                .filter((el) => {
                  const rect = el.getBoundingClientRect();
                  const style = window.getComputedStyle(el);
                  return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
                });
              return nodes.map((el, index) => {
                el.setAttribute('data-agent-id', String(index));
                const rect = el.getBoundingClientRect();
                const label = el.getAttribute('aria-label') || el.name || el.placeholder || el.innerText || el.value || '';
                let selector = `[data-agent-id="${index}"]`;
                return {
                  index,
                  tag: el.tagName.toLowerCase(),
                  role: el.getAttribute('role') || el.tagName.toLowerCase(),
                  name: label.trim().slice(0, 120),
                  text: (el.innerText || el.value || '').trim().slice(0, 160),
                  selector,
                  box: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                };
              });
            }
            """
        )

    def _apply_action(self, action: str) -> None:
        assert self.page is not None
        match = ACTION_RE.match(action.strip())
        if match is None:
            raise ValueError(f"Invalid action syntax: {action!r}")
        name = match.group("name")
        args = self._parse_args(match.group("args"))
        if name == "noop":
            return
        if name == "press":
            key = str(args[0])
            self.page.keyboard.press(key)
            return
        index = int(args[0])
        locator = self.page.locator(f'[data-agent-id="{index}"]').first
        if name == "click":
            locator.click()
        elif name == "fill":
            locator.fill(str(args[1]))
        elif name == "select":
            option = str(args[1])
            try:
                locator.select_option(value=option, timeout=3000)
            except Exception:
                locator.select_option(label=option, timeout=3000)

    @staticmethod
    def _parse_args(raw: str) -> list[str]:
        raw = raw.strip()
        if not raw:
            return []
        args: list[str] = []
        current = []
        in_quote = False
        quote_char = ""
        for char in raw:
            if char in {'"', "'"}:
                if not in_quote:
                    in_quote = True
                    quote_char = char
                    continue
                if quote_char == char:
                    in_quote = False
                    quote_char = ""
                    continue
            if char == "," and not in_quote:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        args.append("".join(current).strip())
        return args
