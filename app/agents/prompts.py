"""Prompt templates for the tutoring graph.

Each system prompt starts with a ``TASK: <name>`` line. Besides documenting
intent, it is the hook the offline :class:`~app.llm.providers.StubLLMClient`
uses to pick its deterministic response.

The pedagogical framing (explain, don't just answer; never fabricate) is
carried over from the v1 Streamlit system message and then split across the
four nodes.
"""

from __future__ import annotations

from typing import Any

from app.models.domain import HistoryTurn
from app.models.enums import Difficulty, Subject

SUBJECTS = ", ".join(s.value for s in Subject)
DIFFICULTIES = ", ".join(d.value for d in Difficulty)

TUTOR_PERSONA = (
    "You are a patient, encouraging homework tutor for school and early "
    "university students. You explain reasoning rather than only giving answers, "
    "and you never invent facts, formulas or citations."
)

UNDERSTAND_SYSTEM = f"""TASK: understand
{TUTOR_PERSONA}

Your job right now is ONLY to understand the question - do not solve it.
Return ONLY a JSON object with exactly these keys:
  "restated_question": the question in your own words, one or two sentences
  "subject":           one of [{SUBJECTS}]
  "topic":             a short topic label, e.g. "projectile motion"
  "difficulty":        one of [{DIFFICULTIES}]
  "knowns":            array of the quantities or facts the question provides
  "unknowns":          array of what the student needs to find
  "is_answerable":     false if the question is too vague, incomplete or
                       off-topic to attempt; true otherwise
Do not include any text outside the JSON object."""

SOLVE_SYSTEM = f"""TASK: solve
{TUTOR_PERSONA}

Solve the problem correctly and show your working.
Return ONLY a JSON object with exactly these keys:
  "answer":      the final answer, stated plainly, with units where relevant
  "steps":       array of the working, one short step per element
  "assumptions": array of any assumptions you had to make (empty if none)
Rules:
- Use only the information given plus standard knowledge for the subject.
- If a value is missing, state the assumption in "assumptions" rather than
  silently inventing it.
- Keep each step to one idea so it can be explained on its own.
Do not include any text outside the JSON object."""

SELF_CHECK_SYSTEM = """TASK: self_check
You are a strict marker reviewing a tutor's proposed answer before a student
sees it. Be sceptical: your job is to catch errors, not to be agreeable.

Return ONLY a JSON object with exactly these keys:
  "passed":     true only if the answer is correct and the working supports it
  "confidence": a float between 0 and 1 for how sure you are
  "issues":     array of short strings describing each problem found
Fail the answer when it: contains an arithmetic or algebraic error, skips a
step the conclusion depends on, contradicts the question, invents data, uses
wrong units, or does not actually answer what was asked.
Do not include any text outside the JSON object."""

EXPLAIN_SYSTEM = f"""TASK: explain
{TUTOR_PERSONA}

The answer below has already been checked. Explain it to the student step by
step so they could reproduce it themselves.
Rules:
- Open with one sentence naming the method or principle used.
- Then walk through the steps in order, with a short "why" for each.
- Use markdown; put the final answer on its own line in bold.
- Finish with one sentence suggesting what to practise next.
- Do not contradict or re-derive the checked answer.
Output the explanation text only - no preamble and no JSON."""


def format_history(history: list[HistoryTurn], *, limit: int = 8) -> str:
    """Render prior turns as plain dialogue for prompt interpolation."""
    if not history:
        return "(this is the first question in the conversation)"
    recent = history[-limit:]
    return "\n".join(f"{turn.role.value}: {turn.content}" for turn in recent)


def build_understand_user(*, question: str, history: list[HistoryTurn]) -> str:
    return (
        f"Conversation so far:\n{format_history(history)}\n\n"
        f"New question:\n<question>{question}</question>"
    )


def build_solve_user(
    *,
    question: str,
    understanding: dict[str, Any],
    previous_answer: str | None = None,
    previous_issues: list[str] | None = None,
) -> str:
    """Solve prompt; on a retry it carries the marker's objections forward."""
    sections = [
        f"Question:\n<question>{question}</question>",
        (
            "Your understanding of it:\n"
            f"- restated: {understanding.get('restated_question', '')}\n"
            f"- subject: {understanding.get('subject', 'other')}\n"
            f"- topic: {understanding.get('topic', '')}\n"
            f"- knowns: {understanding.get('knowns', [])}\n"
            f"- unknowns: {understanding.get('unknowns', [])}"
        ),
    ]
    if previous_answer and previous_issues:
        issues = "\n".join(f"- {issue}" for issue in previous_issues)
        sections.append(
            "Your previous attempt was REJECTED by the marker.\n"
            f"Previous answer:\n<previous_answer>{previous_answer}</previous_answer>\n"
            f"Problems the marker found:\n{issues}\n"
            "Produce a corrected solution that fixes every problem above."
        )
    return "\n\n".join(sections)


def build_self_check_user(*, question: str, solution: dict[str, Any]) -> str:
    steps = "\n".join(f"- {step}" for step in solution.get("steps") or [])
    return (
        f"Question:\n<question>{question}</question>\n\n"
        f"Proposed answer:\n<answer>{solution.get('answer', '')}</answer>\n\n"
        f"Working:\n<steps>\n{steps}\n</steps>"
    )


def build_explain_user(
    *, question: str, understanding: dict[str, Any], solution: dict[str, Any]
) -> str:
    steps = "\n".join(f"- {step}" for step in solution.get("steps") or [])
    return (
        f"Question:\n<question>{question}</question>\n\n"
        f"Subject: {understanding.get('subject', 'other')} / "
        f"{understanding.get('topic', '')}\n"
        f"Difficulty: {understanding.get('difficulty', 'advanced_school')}\n\n"
        f"Checked answer:\n<answer>{solution.get('answer', '')}</answer>\n\n"
        f"Working to explain:\n<steps>\n{steps}\n</steps>"
    )


def build_fallback_answer(
    *,
    reason: str,
    question: str,
    solution: dict[str, Any] | None,
    issues: list[str],
) -> str:
    """The "partial answer, please verify" response.

    This is what the student sees when the self-check never passed, so it must
    be explicit that the content is unverified rather than quietly presenting a
    rejected answer as if it were checked.
    """
    lines = [
        "**I could not fully verify this answer, so please check it yourself "
        "before relying on it.**",
        "",
    ]
    if solution and solution.get("answer"):
        lines.append(f"Here is my best partial attempt at *{question.strip()}*:")
        lines.append("")
        lines.append(str(solution["answer"]))
        steps = solution.get("steps") or []
        if isinstance(steps, list) and steps:
            lines.append("")
            lines.append("Working I had so far:")
            lines.extend(f"- {step}" for step in steps)
    else:
        lines.append("I was not able to produce a usable attempt at this question.")

    if issues:
        lines.extend(["", "What I was unsure about:"])
        lines.extend(f"- {issue}" for issue in issues)

    lines.extend(
        [
            "",
            f"_(unverified - reason: {reason})_",
            "",
            "If you can share the full problem statement, any diagram, and what "
            "you have tried so far, I can give you a checked answer.",
        ]
    )
    return "\n".join(lines)
