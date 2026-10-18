"""Test doubles for the LLM seam."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.llm.base import LLMClient, LLMResponse, chunk_text

Scripted = str | BaseException


def build_understanding(
    *,
    restated: str = "Find the requested value.",
    subject: str = "math",
    topic: str = "arithmetic",
    difficulty: str = "advanced_school",
    knowns: list[str] | None = None,
    unknowns: list[str] | None = None,
    is_answerable: bool = True,
) -> str:
    """Serialise an ``understand`` reply exactly as a model would return it."""
    return json.dumps(
        {
            "restated_question": restated,
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "knowns": knowns if knowns is not None else ["20 m"],
            "unknowns": unknowns if unknowns is not None else ["the time to fall"],
            "is_answerable": is_answerable,
        }
    )


def build_solution(
    *,
    answer: str = "The time taken is approximately 2.0 seconds.",
    steps: list[str] | None = None,
    assumptions: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            # `is not None` so an explicit [] stays empty: the no-working
            # guardrail has to be reachable from a test.
            "steps": (
                steps
                if steps is not None
                else ["Use s = ut + ½at².", "Substitute u = 0.", "Solve for t."]
            ),
            "assumptions": assumptions if assumptions is not None else [],
        }
    )


def build_self_check(
    *, passed: bool = True, confidence: float = 0.9, issues: list[str] | None = None
) -> str:
    return json.dumps(
        {
            "passed": passed,
            "confidence": confidence,
            "issues": issues if issues is not None else [],
        }
    )


EXPLANATION = (
    "We use the constant-acceleration equation.\n\n"
    "**Step 1.** Start from s = ut + ½at², with u = 0 because the ball is dropped.\n"
    "**Step 2.** Rearrange to t = sqrt(2s/a).\n"
    "**Step 3.** Substitute s = 20 m and a = 9.81 m/s².\n\n"
    "**t ≈ 2.0 s**\n\n"
    "Try the same method for a 45 m drop to check it has stuck."
)


class ScriptedLLMClient(LLMClient):
    """Returns queued replies per task, raising queued exceptions instead.

    The queue for a task is consumed left to right; once exhausted the last
    entry repeats, which is what makes "always fails" scripts terse.
    """

    name = "scripted"
    model = "scripted-model"
    supports_streaming = False

    def __init__(self, script: dict[str, list[Scripted]]) -> None:
        self._script = {task: list(items) for task, items in script.items()}
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _detect_task(system: str) -> str:
        for line in system.splitlines():
            if line.strip().upper().startswith("TASK:"):
                return line.split(":", 1)[1].strip().lower()
        return "unknown"

    def call_count(self, task: str) -> int:
        return sum(1 for recorded_task, _ in self.calls if recorded_task == task)

    def prompts_for(self, task: str) -> list[str]:
        return [user for recorded_task, user in self.calls if recorded_task == task]

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> LLMResponse:
        task = self._detect_task(system)
        self.calls.append((task, user))
        queue = self._script.get(task)
        if not queue:
            raise AssertionError(f"ScriptedLLMClient has no script for task {task!r}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(text=item, model=self.model, raw={"task": task})


class StreamingScriptedLLMClient(ScriptedLLMClient):
    """A scripted client that advertises and implements native streaming."""

    supports_streaming = True

    def __init__(self, script: dict[str, list[Scripted]], *, chunk_size: int = 12) -> None:
        super().__init__(script)
        self._chunk_size = chunk_size
        self.stream_calls = 0

    async def stream(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        response = await self.generate(
            system=system, user=user, temperature=temperature, max_tokens=max_tokens
        )
        for piece in chunk_text(response.text, size=self._chunk_size):
            yield piece


def happy_script() -> dict[str, list[Scripted]]:
    """The script for a question that passes the self-check first time."""
    return {
        "understand": [build_understanding()],
        "solve": [build_solution()],
        "self_check": [build_self_check(passed=True, confidence=0.92)],
        "explain": [EXPLANATION],
    }
