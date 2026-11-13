"""Extraction of intra-option behaviour from the trained policy.

Algorithm 2 defines every certified option with an "intra-option policy extracted from
``pi_k``".  The option's behaviour already lives in the shared network -- the policy has
just demonstrated the call pattern -- but nothing yet tells the network *when to
initiate* the freshly added option action.  Extraction closes that gap:

1. every successful solution sampled during certification is re-expressed as an
   option-level trajectory: its chain-level API calls are covered by certified options
   and an option decision is inserted in front of the code each option accounts for;
2. the rewritten trajectory is replayed through the option automata, so it is kept
   only if it is exactly a trajectory the constrained decoder could have produced;
3. the policy is fitted to these trajectories by maximum likelihood for a few steps.

After extraction the option actions have calibrated initiation probabilities and the
reference policy of the next phase (snapshotted afterwards) already contains them.
"""

from __future__ import annotations

from dataclasses import dataclass

from option_grpo.analysis.callseq import ProgramAnalysis, analyze_program, char_offset
from option_grpo.options.constraints import ConstraintEngine
from option_grpo.options.odl import cover
from option_grpo.options.option import CodeOption, OptionVocabulary
from option_grpo.options.tracker import InitiationState, InitiationTable
from option_grpo.policy.sampling import DECISION, GLUE, INTRA, OptionCall, StepMask, Trajectory
from option_grpo.sdk.introspect import SDKIndex
from option_grpo.sdk.types import TypeRef


@dataclass
class _Insertion:
    option: CodeOption
    offsets: list[int]  # candidate character offsets, most preferred first


def _token_tape(trajectory: Trajectory, strings: list[str], stop_ids: set[int]) -> tuple[list[int], list[int]]:
    """Token actions of a trajectory and the character offset at which each begins."""
    tokens: list[int] = []
    starts: list[int] = []
    cursor = 0
    for action, kind in zip(trajectory.actions, trajectory.kinds):
        if kind == DECISION:
            continue
        tokens.append(action)
        starts.append(cursor)
        if action not in stop_ids:
            cursor += len(strings[action])
    return tokens, starts


def _plan(
    analysis: ProgramAnalysis,
    code: str,
    patterns: dict[tuple[str, ...], CodeOption],
    by_function: dict[str, CodeOption],
) -> list[_Insertion]:
    """Which option should start where, from the cover of the program's call sequence."""
    primary = analysis.primary
    assert analysis.tree is not None
    plan: list[_Insertion] = []
    for start, end, option in cover([c.name for c in primary], patterns, min_len=1):
        first = primary[start]
        if end - start == 1 and first.func in by_function:
            option = by_function[first.func]  # the atomic option of the resolved function
        offsets: list[int] = []
        if 0 <= first.stmt < len(analysis.tree.body):
            stmt = analysis.tree.body[first.stmt]
            offsets.append(char_offset(code, stmt.lineno, stmt.col_offset))
        offsets.append(char_offset(code, first.name_lineno, first.name_col))
        plan.append(_Insertion(option, offsets))
    return plan


def annotate_trajectory(
    trajectory: Trajectory,
    index: SDKIndex,
    vocab: OptionVocabulary,
    constraints: ConstraintEngine,
    initiation: InitiationTable,
    base_vocab_size: int,
    stop_ids: set[int],
    context: str = "",
    value_types: list[TypeRef] | None = None,
    max_option_tokens: int = 96,
) -> Trajectory | None:
    """Rewrite a successful trajectory with the options of ``initiation``'s action space.

    Returns None when the program does not parse or no option could be inserted.
    """
    strings = constraints.vocab.strings
    tokens, starts = _token_tape(trajectory, strings, stop_ids)
    code = "".join(strings[t] for t in tokens if t not in stop_ids)
    analysis = analyze_program(trajectory.text, index, context)
    if not analysis.ok or analysis.tree is None or not code.startswith(trajectory.text):
        return None
    options = [vocab[i] for i in initiation.available]
    available: dict[tuple[str, ...], CodeOption] = {}
    for option in options:
        current = available.get(option.call_names)
        if current is None or option.weight > current.weight:
            available[option.call_names] = option
    by_function = {o.path[0]: o for o in options if o.level == 0}
    plan = _plan(analysis, trajectory.text, available, by_function)
    if not plan:
        return None

    state = InitiationState(initiation, value_types)
    out = Trajectory(trajectory.task_id, list(trajectory.prompt_ids), text=trajectory.text, finish=trajectory.finish)
    cursor = 0  # index of the next token to emit
    inserted = 0

    def emit_glue(upto: int) -> None:
        nonlocal cursor
        while cursor < upto:
            out.actions.append(tokens[cursor])
            out.kinds.append(GLUE)
            out.masks.append(StepMask(state.allowed, None))
            out.logp.append(0.0)
            cursor += 1

    for insertion in plan:
        option = insertion.option
        if option.index not in state.allowed:
            continue
        for offset in insertion.offsets:
            # Largest token boundary that does not come after the target offset.
            boundary = max((i for i, s in enumerate(starts) if s <= offset), default=None)
            if boundary is None:
                continue
            if boundary < cursor:
                # The previous option's last token already covers the target (typically
                # a token such as ")." that closes one call and opens the next access).
                # Starting right after it is equivalent as long as nothing but the dot
                # separates the two positions.
                if cursor >= len(starts) or code[offset : starts[cursor]].strip() not in {"", "."}:
                    continue
                boundary = cursor
            prefix = "".join(strings[t] for t in tokens[:boundary] if t not in stop_ids)
            automaton = constraints.start(option.index, prefix)
            masks: list[StepMask] = []
            position = boundary
            ok = False
            while position < len(tokens) and position - boundary < max_option_tokens:
                token = tokens[position]
                spec = constraints.mask(automaton)
                if token in stop_ids or spec.empty or not spec.allows(token):
                    break
                masks.append(StepMask((), spec))
                constraints.advance(automaton, token)
                position += 1
                if automaton.terminated:
                    ok = True
                    break
            if not ok:
                continue
            emit_glue(boundary)
            decision_index = len(out.actions)
            out.actions.append(base_vocab_size + option.index)
            out.kinds.append(DECISION)
            out.masks.append(StepMask(state.allowed, None))
            out.logp.append(0.0)
            for token, mask in zip(tokens[boundary:position], masks):
                out.actions.append(token)
                out.kinds.append(INTRA)
                out.masks.append(mask)
                out.logp.append(0.0)
            out.calls.append(OptionCall(option.index, decision_index, len(out.actions), True))
            cursor = position
            state.on_terminate(option, True)
            inserted += 1
            break
    if inserted == 0:
        return None
    emit_glue(len(tokens))
    return out
