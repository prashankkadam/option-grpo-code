"""Generation back-ends for hosted models.

Any service that implements the OpenAI-compatible ``/chat/completions`` route can be
evaluated with the same prompts, temperature and number of samples as the local models.
The API key is read from an environment variable; nothing else is sent besides the
prompt.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from option_grpo.tasks.dataset import Task
from option_grpo.tasks.prompts import BEGIN_MARKER, END_MARKER, build_prompt, extract_solution
from option_grpo.utils import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You complete Python programs. You are given the beginning of a file that ends with the line "
    f"'{BEGIN_MARKER}'. Reply with the code that follows that line and finish with the line '{END_MARKER}'. "
    "Reply with code only."
)


@dataclass
class ApiModel:
    name: str  # label used in reports
    model: str  # model identifier understood by the endpoint
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 120.0
    max_retries: int = 5
    concurrency: int = 4


class ChatCompletionsGenerator:
    """Samples programs from an OpenAI-compatible chat endpoint."""

    def __init__(self, spec: ApiModel, context_fn: Callable[[Task], str] | None = None) -> None:
        self.spec = spec
        self.context_fn = context_fn
        key = os.environ.get(spec.api_key_env)
        if not key:
            raise RuntimeError(f"environment variable {spec.api_key_env} is not set")
        self._key = key

    def _request(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": self.spec.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": [END_MARKER],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.spec.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
            method="POST",
        )
        delay = 2.0
        for attempt in range(self.spec.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.spec.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == self.spec.max_retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.spec.max_retries - 1:
                    raise
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
        return ""

    def generate(self, task: Task, n: int, temperature: float, max_new_tokens: int) -> list[str]:
        prompt = build_prompt(task, self.context_fn(task) if self.context_fn else "")
        with ThreadPoolExecutor(max_workers=max(self.spec.concurrency, 1)) as pool:
            completions = list(pool.map(lambda _: self._request(prompt, temperature, max_new_tokens), range(n)))
        return [extract_solution(c) for c in completions]
