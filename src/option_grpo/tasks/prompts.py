"""Prompt format.

The base model is a completion model, so a task is rendered as the beginning of a
Python file: the problem statement as a docstring, the imports, the setup code and an
opening ``# BEGIN SOLUTION`` marker.  Generation stops at ``# END SOLUTION``.
"""

from __future__ import annotations

import ast
import textwrap

from option_grpo.tasks.dataset import Task

BEGIN_MARKER = "# BEGIN SOLUTION"
END_MARKER = "# END SOLUTION"


def render_context(chunks: list[str], header: str = "Reference material retrieved from the SDK") -> str:
    """Render retrieved passages as a comment block placed in front of the problem."""
    if not chunks:
        return ""
    lines = [f"# {header}:"]
    for number, chunk in enumerate(chunks, start=1):
        lines.append(f"# [{number}]")
        lines.extend(f"# {line}" if line.strip() else "#" for line in chunk.rstrip().splitlines())
    return "\n".join(lines) + "\n\n"


def build_prompt(task: Task, context: str = "") -> str:
    """Prompt for ``task``; ``context`` is optional retrieved material (RAG baselines)."""
    statement = textwrap.fill(f"Problem: {task.description.splitlines()[0]}", width=88)
    extra = task.description.splitlines()[1:]
    docstring = '"""' + statement + ("\n" + "\n".join(extra) if extra else "") + '"""'
    parts = [context + docstring, task.prelude.rstrip()]
    if task.setup.strip():
        parts.append("")
        parts.append("# Setup")
        parts.append(task.setup.rstrip())
    parts.append("")
    parts.append(BEGIN_MARKER)
    return "\n".join(parts) + "\n"


def extract_solution(completion: str) -> str:
    """Solution code from a raw completion: everything before the end marker."""
    text = completion
    cut = text.find(END_MARKER)
    if cut >= 0:
        text = text[:cut]
    # A model may repeat the begin marker or wrap the code in a fence.
    begin = text.rfind(BEGIN_MARKER)
    if begin >= 0:
        text = text[begin + len(BEGIN_MARKER):]
    lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
    if not any(line.strip() for line in lines):
        return ""
    return textwrap.dedent("\n".join(lines)).strip("\n") + "\n"


def parses(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True
