"""Rollouts of the hierarchical policy and evaluation of action log-probabilities.

Decoding proceeds one action at a time.  Each sample is in one of two modes:

* **high level** -- the next action is a token of ``V`` or one of the options whose
  initiation set contains the current state.  Choosing an option is an *option
  decision*; it emits no text.
* **intra option** -- an option is active: the next action is a token admitted by the
  option's automaton (see :mod:`option_grpo.options.constraints`).  When the automaton
  reports termination the sample returns to the high level.

For every step the engine records the action, the log-probability the (masked) policy
assigned to it at temperature 1 and a compact description of the mask, so that the
exact same distribution can be re-evaluated under updated parameters during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from option_grpo.config import OptionDecodeConfig
from option_grpo.options.constraints import ConstraintEngine, MaskSpec, OptionAutomaton
from option_grpo.options.option import OptionVocabulary
from option_grpo.options.tracker import InitiationState, InitiationTable
from option_grpo.policy.hierarchical import HierarchicalPolicy
from option_grpo.sdk.types import TypeRef

GLUE, DECISION, INTRA = 0, 1, 2
NEG_INF = float("-inf")


@dataclass
class StepMask:
    """Mask of one decoding step: admissible options and (optionally) a token mask."""

    options: tuple[int, ...] = ()
    tokens: MaskSpec | None = None  # None: every token is admissible


@dataclass
class OptionCall:
    """One invocation of an option inside a trajectory."""

    option: int
    start: int  # index of the decision action
    end: int = -1  # index one past the last intra-option action
    completed: bool = False


@dataclass
class Trajectory:
    task_id: str
    prompt_ids: list[int]
    actions: list[int] = field(default_factory=list)  # extended action ids
    logp: list[float] = field(default_factory=list)  # log pi_old(a_t | s_t), temperature 1
    kinds: list[int] = field(default_factory=list)
    masks: list[StepMask] = field(default_factory=list)
    calls: list[OptionCall] = field(default_factory=list)
    text: str = ""  # program text (tokens only; option decisions emit no text)
    finish: str = "length"  # stop | eos | length

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def options_used(self) -> list[int]:
        """Options invoked in this trajectory, in order (the option sequence)."""
        return [call.option for call in self.calls]

    @property
    def n_tokens(self) -> int:
        return sum(1 for kind in self.kinds if kind != DECISION)

    def summary(self) -> dict[str, Any]:
        return {
            "finish": self.finish,
            "actions": len(self.actions),
            "tokens": self.n_tokens,
            "options": self.options_used,
            "completed": [call.completed for call in self.calls],
        }


class _Sample:
    """Mutable decoding state of one sequence."""

    __slots__ = ("traj", "automaton", "call", "initiation", "option_tokens", "done", "scan_from")

    def __init__(self, traj: Trajectory, initiation: InitiationState | None) -> None:
        self.traj = traj
        self.automaton: OptionAutomaton | None = None
        self.call: OptionCall | None = None
        self.initiation = initiation
        self.option_tokens = 0
        self.done = False
        self.scan_from = 0


def expand_cache(past: Any, n: int) -> Any:
    """Repeat a batch-1 KV cache ``n`` times along the batch dimension."""
    if past is None or n == 1:
        return past
    if hasattr(past, "batch_repeat_interleave"):
        past.batch_repeat_interleave(n)
        return past
    return tuple(tuple(t.repeat_interleave(n, dim=0) for t in layer) for layer in past)


class RolloutEngine:
    """Samples trajectories from a :class:`HierarchicalPolicy` and re-scores them."""

    def __init__(
        self,
        policy: HierarchicalPolicy,
        tokenizer,
        vocab: OptionVocabulary,
        constraints: ConstraintEngine | None,
        cfg: OptionDecodeConfig | None = None,
        stop_strings: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.constraints = constraints
        self.cfg = cfg or OptionDecodeConfig()
        self.stop_strings = [s for s in (stop_strings or []) if s]
        stops = set(stop_token_ids or [])
        if tokenizer.eos_token_id is not None:
            stops.add(int(tokenizer.eos_token_id))
        self.stop_token_ids = stops
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0))
        self._strings = constraints.vocab.strings if constraints is not None else None
        self._max_stop = max((len(s) for s in self.stop_strings), default=0)

    # ------------------------------------------------------------------ helpers
    def encode_prompt(self, prompt: str) -> list[int]:
        return list(self.tokenizer(prompt)["input_ids"])

    def _token_text(self, token_id: int) -> str:
        if self._strings is not None:
            return self._strings[token_id]
        return self.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _close_option(self, sample: _Sample, end: int, completed: bool) -> None:
        call = sample.call
        assert call is not None
        call.end = end
        call.completed = completed
        if sample.initiation is not None:
            sample.initiation.on_terminate(self.vocab[call.option], completed)
        sample.automaton = None
        sample.call = None
        sample.option_tokens = 0

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def rollout(
        self,
        task_id: str,
        prompt: str,
        n: int,
        initiation: InitiationTable | None = None,
        value_types: list[TypeRef] | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 384,
        generator: torch.Generator | None = None,
    ) -> list[Trajectory]:
        """Sample ``n`` trajectories for one prompt.

        ``initiation`` selects the action space: ``None`` gives token-level decoding
        (``A_0``); a table for level ``k`` makes the certified options below ``k``
        available.
        """
        policy = self.policy
        was_training = policy.training
        policy.eval()
        try:
            return self._rollout(task_id, prompt, n, initiation, value_types, temperature, max_new_tokens, generator)
        finally:
            policy.train(was_training)

    def _rollout(self, task_id, prompt, n, initiation, value_types, temperature, max_new_tokens, generator):
        policy = self.policy
        device = policy.device
        vocab_size = policy.base_vocab_size
        use_options = initiation is not None and self.constraints is not None and bool(initiation.available)
        prompt_ids = self.encode_prompt(prompt)
        samples = [
            _Sample(Trajectory(task_id, prompt_ids), InitiationState(initiation, value_types) if use_options else None)
            for _ in range(n)
        ]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        hidden, past = policy.hidden(policy.embed(ids), use_cache=True)
        past = expand_cache(past, n)
        last = hidden[:, -1].expand(n, -1)
        if past is None:  # model without a cache: fall back to full re-encoding
            raise RuntimeError("the language model did not return a key/value cache")

        for step in range(max_new_tokens):
            token_logits, option_logits = policy.heads(last)
            option_mask = torch.full_like(option_logits, NEG_INF)
            step_masks: list[StepMask | None] = [None] * n
            for g, sample in enumerate(samples):
                if sample.done:
                    continue
                if sample.automaton is not None:
                    spec = None
                    if sample.option_tokens < self.cfg.max_option_tokens:
                        spec = self.constraints.mask(sample.automaton)  # type: ignore[union-attr]
                    if spec is None or spec.empty:
                        # Length cap reached or dead end: abort and hand control back.
                        self._close_option(sample, step, completed=False)
                    else:
                        spec.apply_(token_logits[g])
                        step_masks[g] = StepMask((), spec)
                        continue
                allowed = sample.initiation.allowed if sample.initiation is not None else ()
                if allowed:
                    option_mask[g, list(allowed)] = 0.0
                step_masks[g] = StepMask(tuple(allowed), None)
            joint = torch.cat([token_logits, option_logits + option_mask], dim=-1)
            logp = torch.log_softmax(joint, dim=-1)
            if temperature <= 0.0:
                actions = joint.argmax(dim=-1)
            else:
                probs = torch.softmax(joint / temperature, dim=-1)
                actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            chosen_logp = logp.gather(1, actions[:, None]).squeeze(1).tolist()
            action_list = actions.tolist()

            for g, sample in enumerate(samples):
                if sample.done:
                    action_list[g] = self.pad_id
                    continue
                action = action_list[g]
                traj = sample.traj
                mask = step_masks[g]
                assert mask is not None
                traj.actions.append(action)
                traj.logp.append(chosen_logp[g])
                traj.masks.append(mask)
                if action >= vocab_size:  # option decision
                    option_index = action - vocab_size
                    traj.kinds.append(DECISION)
                    sample.call = OptionCall(option_index, start=step)
                    traj.calls.append(sample.call)
                    sample.automaton = self.constraints.start(option_index, traj.text)  # type: ignore[union-attr]
                    sample.option_tokens = 0
                    continue
                inside = sample.automaton is not None
                traj.kinds.append(INTRA if inside else GLUE)
                if action in self.stop_token_ids:
                    traj.finish = "eos"
                    sample.done = True
                else:
                    traj.text += self._token_text(action)
                    if inside:
                        sample.option_tokens += 1
                        self.constraints.advance(sample.automaton, action)  # type: ignore[union-attr,arg-type]
                        if sample.automaton.terminated:  # type: ignore[union-attr]
                            self._close_option(sample, step + 1, completed=True)
                    if self._max_stop:
                        window = traj.text[max(sample.scan_from - self._max_stop, 0):]
                        for stop in self.stop_strings:
                            position = window.find(stop)
                            if position >= 0:
                                cut = max(sample.scan_from - self._max_stop, 0) + position
                                traj.text = traj.text[:cut]
                                traj.finish = "stop"
                                sample.done = True
                                break
                        sample.scan_from = len(traj.text)
                if sample.done and sample.call is not None:
                    self._close_option(sample, step + 1, completed=False)
            if all(sample.done for sample in samples):
                break
            next_ids = torch.tensor(action_list, dtype=torch.long, device=device)[:, None]
            hidden, past = policy.hidden(policy.embed(next_ids), past_key_values=past, use_cache=True)
            last = hidden[:, -1]

        for sample in samples:
            if sample.call is not None:  # ran out of budget inside an option
                self._close_option(sample, len(sample.traj.actions), completed=False)
            self._finalise_text(sample.traj)
        return [sample.traj for sample in samples]

    def _finalise_text(self, traj: Trajectory) -> None:
        """Re-decode the whole token sequence when per-token decoding split a character."""
        if "\ufffd" not in traj.text:
            return
        ids = [a for a, kind in zip(traj.actions, traj.kinds) if kind != DECISION and a not in self.stop_token_ids]
        text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for stop in self.stop_strings:
            cut = text.find(stop)
            if cut >= 0:
                text = text[:cut]
        traj.text = text

    # ------------------------------------------------------------------ scoring
    def log_probs(
        self,
        trajectories: list[Trajectory],
        policy: torch.nn.Module | None = None,
    ) -> list[torch.Tensor]:
        """Log-probabilities of the recorded actions under ``policy`` (default: own policy).

        The masks recorded at sampling time are re-applied, so the returned values are
        ``log pi(a_t | s_t)`` of exactly the distribution the actions were drawn from
        (at temperature 1).  Gradients flow when called outside ``torch.no_grad``.
        Returns one 1-D tensor per trajectory.
        """
        module = policy if policy is not None else self.policy
        inner: HierarchicalPolicy = getattr(module, "module", module)  # unwrap DDP if present
        device = inner.device
        lengths = [len(t.prompt_ids) + len(t.actions) for t in trajectories]
        width = max(lengths)
        input_ids = torch.full((len(trajectories), width), self.pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(trajectories), width), dtype=torch.long, device=device)
        rows: list[int] = []
        cols: list[int] = []
        targets: list[int] = []
        for b, traj in enumerate(trajectories):
            sequence = traj.prompt_ids + traj.actions
            input_ids[b, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention[b, : len(sequence)] = 1
            offset = len(traj.prompt_ids) - 1
            for t, action in enumerate(traj.actions):
                rows.append(b)
                cols.append(offset + t)
                targets.append(action)
        if not rows:
            return [torch.zeros(0, device=device) for _ in trajectories]
        row_index = torch.tensor(rows, device=device)
        col_index = torch.tensor(cols, device=device)
        token_logits, option_logits = module(input_ids, attention, row_index, col_index)

        ban = torch.zeros_like(token_logits, dtype=torch.bool)
        option_ban = torch.ones_like(option_logits, dtype=torch.bool)
        flat = 0
        for traj in trajectories:
            for mask in traj.masks:
                if mask.options:
                    option_ban[flat, list(mask.options)] = False
                spec = mask.tokens
                if spec is not None:
                    ids = spec.ids.to(device)
                    if spec.mode == "ban":
                        ban[flat, ids] = True
                    else:
                        ban[flat] = True
                        ban[flat, ids] = False
                flat += 1
        joint = torch.cat(
            [token_logits.masked_fill(ban, NEG_INF), option_logits.masked_fill(option_ban, NEG_INF)], dim=-1
        )
        logp = torch.log_softmax(joint, dim=-1)
        chosen = logp.gather(1, torch.tensor(targets, device=device)[:, None]).squeeze(1)
        out: list[torch.Tensor] = []
        start = 0
        for traj in trajectories:
            out.append(chosen[start : start + len(traj.actions)])
            start += len(traj.actions)
        return out
