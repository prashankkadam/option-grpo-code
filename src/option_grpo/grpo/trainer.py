"""Curriculum-staged Option-GRPO training (Algorithm 4).

    Stage 0   graphs, candidate options, tasks and property tests (prepared offline)
    Phase 1   Level-0 tasks, token-level GRPO, layered execution reward      -> certify O_0
    Phase 2   Level-1 tasks, action space A_1 = O_0 ∪ V, + quality reward,
              adaptive composition length                                     -> certify O_1
    Phase 3   Level-2 tasks, action space A_2 = O_0 ∪ O_1 ∪ V, + consistency reward

Every training step samples ``G`` trajectories per prompt, executes the programs in the
sandbox, converts the rewards into sequence-level and option-level group-relative
advantages and takes one clipped policy-gradient step.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from option_grpo.analysis.callseq import analyze_program
from option_grpo.analysis.quality import tokenizer_offsets
from option_grpo.config import Config, save_config
from option_grpo.execution.sandbox import Executor
from option_grpo.graphs.build import SDKGraphs
from option_grpo.grpo.advantages import option_advantages, sequence_advantages
from option_grpo.grpo.curriculum import LengthPacer, Phase, build_phases, phase_tasks
from option_grpo.grpo.loss import grpo_loss
from option_grpo.grpo.optim import apply_partial_freeze, build_optimizer, schedule_factor, set_learning_rates
from option_grpo.grpo.reference import ReferencePolicy
from option_grpo.options.certification import SolutionRecord, certify_options
from option_grpo.options.constraints import ConstraintEngine, LanguageTables, VocabIndex
from option_grpo.options.extraction import annotate_trajectory
from option_grpo.options.option import CERTIFIED, OptionVocabulary
from option_grpo.options.tracker import InitiationTable
from option_grpo.policy.hierarchical import HierarchicalPolicy
from option_grpo.policy.sampling import DECISION, RolloutEngine, Trajectory
from option_grpo.rewards.reward import RewardBreakdown, RewardComputer
from option_grpo.tasks.dataset import Task, TaskSampler, TaskSet
from option_grpo.tasks.prompts import build_prompt, extract_solution
from option_grpo.utils import append_jsonl, get_logger, read_json, write_json

log = get_logger(__name__)

ContextFn = Callable[[Task], str]


@dataclass
class TrainState:
    phase: int = 0  # index into the phase list
    step: int = 0  # steps completed in the current phase
    global_step: int = 0
    pacer: dict[str, Any] = field(default_factory=dict)
    certification: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Group:
    """The ``G`` trajectories sampled for one prompt, with their rewards."""

    task: Task
    trajectories: list[Trajectory]
    programs: list[str]
    rewards: list[RewardBreakdown]


class OptionGRPOTrainer:
    def __init__(
        self,
        cfg: Config,
        policy: HierarchicalPolicy,
        tokenizer,
        graphs: SDKGraphs,
        vocab: OptionVocabulary,
        tasks: TaskSet,
        executor: Executor,
        accelerator=None,
        context_fn: ContextFn | None = None,
        init_options: bool = True,
    ) -> None:
        if accelerator is None:
            from accelerate import Accelerator

            accelerator = Accelerator()
        self.cfg = cfg
        self.acc = accelerator
        self.tokenizer = tokenizer
        self.graphs = graphs
        self.index = graphs.index
        self.vocab = vocab
        self.tasks = tasks
        self.executor = executor
        self.context_fn = context_fn
        self.output_dir = Path(cfg.output_dir)
        self.phases = build_phases(cfg.curriculum)
        self.state = TrainState()
        self.pacer = LengthPacer(cfg.curriculum)

        if cfg.curriculum.use_options and not cfg.curriculum.use_curriculum:
            # Options-only ablation: the whole candidate inventory is available at once.
            for option in vocab.options:
                option.status = CERTIFIED
        if init_options and cfg.curriculum.use_options and len(vocab) > 0:
            policy.init_options(vocab.options, tokenizer)

        self.raw_policy = policy
        self.optimizer = build_optimizer(policy, cfg.optim)
        self.policy, self.optimizer = self.acc.prepare(policy, self.optimizer)
        self.is_deepspeed = str(getattr(self.acc, "distributed_type", "")).upper().endswith("DEEPSPEED")
        if self.is_deepspeed and cfg.grpo.reference == "copy":
            raise ValueError(
                "grpo.reference=copy is not available with DeepSpeed ZeRO; use adapter_off (LoRA) or none"
            )

        self.vocab_index = VocabIndex.from_tokenizer(tokenizer, policy.base_vocab_size)
        self.lang = LanguageTables.from_index(self.index)
        self.constraints = ConstraintEngine(
            self.vocab_index, self.lang, vocab.options, cfg.decode.strict_chain, cfg.decode.mask_cache_size
        )
        self.engine = RolloutEngine(
            policy, tokenizer, vocab, self.constraints, cfg.decode,
            stop_strings=cfg.model.stop_strings, stop_token_ids=cfg.model.extra_stop_token_ids,
        )
        self.rewards = RewardComputer(
            cfg.reward, executor, self.index, vocab, tokenizer_offsets(tokenizer), cfg.benchmark.odl_max_options
        )
        self.reference = ReferencePolicy(policy, cfg.grpo.reference)
        self._initiation: InitiationTable | None = None
        self._sampler: TaskSampler | None = None
        self._wandb = None
        self._autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.model.autocast)

    # ------------------------------------------------------------------ utilities
    @property
    def is_main(self) -> bool:
        return bool(self.acc.is_main_process)

    def _autocast(self):
        device_type = self.raw_policy.device.type
        enabled = self._autocast_dtype is not None and device_type == "cuda"
        return torch.autocast(device_type=device_type if enabled else "cpu", dtype=self._autocast_dtype or torch.bfloat16, enabled=enabled)

    def _prompt(self, task: Task) -> str:
        context = self.context_fn(task) if self.context_fn is not None else ""
        return build_prompt(task, context)

    def initiation_for(self, action_level: int) -> InitiationTable | None:
        """Initiation table of the action space ``A_action_level`` (None for ``A_0``)."""
        if action_level <= 0 or not self.cfg.curriculum.use_options:
            return None
        table = InitiationTable(
            self.vocab, self.index, action_level, self.cfg.decode.initiation, self.cfg.decode.min_initiation_compat
        )
        return table if table.available else None

    def sample_group(self, task: Task, n: int, initiation: InitiationTable | None, temperature: float) -> list[Trajectory]:
        prompt = self._prompt(task)
        chunk = max(self.cfg.grpo.rollout_chunk, 1)
        trajectories: list[Trajectory] = []
        with self._autocast():
            while len(trajectories) < n:
                trajectories += self.engine.rollout(
                    task.id, prompt, min(chunk, n - len(trajectories)), initiation,
                    task.setup_value_types(), temperature, self.cfg.grpo.max_new_tokens,
                )
        return trajectories

    # ------------------------------------------------------------------ phases
    def _begin_phase(self, phase: Phase) -> None:
        self._initiation = self.initiation_for(phase.action_level)
        max_length = self.pacer.length if phase.adaptive else None
        pool = phase_tasks(self.tasks, phase, "train", max_length)
        if not pool:
            raise RuntimeError(f"no training tasks for {phase.name}")
        seed = self.cfg.seed + 7919 * phase.number + 104729 * int(self.acc.process_index)
        self._sampler = TaskSampler(pool, seed)
        if self.reference.enabled and (self.cfg.grpo.reference_refresh == "phase" or self.reference.frozen is None):
            self.reference.refresh()
        available = len(self._initiation.available) if self._initiation is not None else 0
        log.info(
            "%s: %d steps, %d tasks, %d options in the action space, reward level %d",
            phase.name, phase.steps, len(pool), available, phase.reward_level,
        )

    def _end_phase(self, number: int, phase: Phase) -> None:
        if phase.certify is not None:
            summary, successes = self.certify(phase.certify, phase.action_level)
            summary["extraction"] = self.extract(phase.certify + 1, successes)
            self.state.certification.append(summary)
        apply_partial_freeze(self.optimizer, self.raw_policy, number + 1, self.cfg.curriculum.freeze, self.cfg.optim)
        self.save_checkpoint(f"phase{phase.number}")

    def fit(self) -> None:
        """Run all remaining phases."""
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_config(self.cfg, self.output_dir / "config.yaml")
        self._init_tracking()
        for number in range(self.state.phase, len(self.phases)):
            phase = self.phases[number]
            self.state.phase = number
            if phase.steps > 0:
                self._begin_phase(phase)
                while self.state.step < phase.steps:
                    metrics = self.train_step(phase)
                    self._log(phase, metrics)
                    every = self.cfg.logging.save_every
                    if every and self.state.step % every == 0 and self.state.step < phase.steps:
                        self.save_checkpoint("last")
            self._end_phase(number, phase)
            self.state.step = 0
            self.state.phase = number + 1
            self.pacer = LengthPacer(self.cfg.curriculum)
        self.save_checkpoint("final")

    # ------------------------------------------------------------------ one step
    def collect(self, phase: Phase) -> list[Group]:
        assert self._sampler is not None
        grpo = self.cfg.grpo
        groups: list[Group] = []
        for task in self._sampler.sample(grpo.prompts_per_step):
            trajectories = self.sample_group(task, grpo.group_size, self._initiation, grpo.temperature)
            programs = [extract_solution(t.text) for t in trajectories]
            groups.append(Group(task, trajectories, programs, []))
        scored = self.rewards.score([(g.task, g.programs) for g in groups], phase.reward_level)
        for group, breakdowns in zip(groups, scored):
            group.rewards = breakdowns
        return groups

    def advantages(self, groups: list[Group]) -> tuple[list[Trajectory], torch.Tensor, list[torch.Tensor]]:
        """Flatten the groups and attach sequence-level and option-level advantages."""
        grpo = self.cfg.grpo
        trajectories: list[Trajectory] = []
        seq_parts: list[torch.Tensor] = []
        opt_parts: list[torch.Tensor] = []
        base = self.raw_policy.base_vocab_size
        for group in groups:
            totals = [r.total for r in group.rewards]
            seq_parts.append(sequence_advantages(torch.tensor(totals), grpo.adv_eps))
            option_sets = [set(t.options_used) for t in group.trajectories]
            per_option = option_advantages(totals, option_sets, grpo.normalize_option_advantage, grpo.adv_eps)
            for traj in group.trajectories:
                values = torch.zeros(len(traj.actions))
                for position, (action, kind) in enumerate(zip(traj.actions, traj.kinds)):
                    if kind == DECISION:
                        values[position] = per_option.get(action - base, 0.0)
                opt_parts.append(values)
                trajectories.append(traj)
        return trajectories, torch.cat(seq_parts), opt_parts

    def update(self, trajectories: list[Trajectory], seq_adv: torch.Tensor, opt_adv: list[torch.Tensor]) -> dict[str, float]:
        """One optimiser step on a batch of trajectories."""
        grpo = self.cfg.grpo
        device = self.raw_policy.device
        keep = [i for i, t in enumerate(trajectories) if len(t.actions) > 0]
        n_sequences = len(trajectories)
        n_actions = sum(len(trajectories[i].actions) for i in keep)
        order = sorted(keep, key=lambda i: len(trajectories[i].prompt_ids) + len(trajectories[i].actions))
        micro = len(order) if self.is_deepspeed else max(grpo.train_micro_batch, 1)
        stats = {"loss": 0.0, "kl": 0.0, "clip_fraction": 0.0, "surrogate_seq": 0.0, "surrogate_opt": 0.0, "grad_norm": 0.0}
        self.policy.train()
        for _ in range(max(grpo.ppo_epochs, 1)):
            self.optimizer.zero_grad(set_to_none=True)
            clip_weighted = 0.0
            epoch_loss = 0.0
            for start in range(0, len(order), max(micro, 1)):
                chunk_ids = order[start : start + micro]
                chunk = [trajectories[i] for i in chunk_ids]
                with self._autocast():
                    logp_new = self.engine.log_probs(chunk, self.policy)
                    logp_ref = self.reference.log_probs(self.engine, chunk) if grpo.kl_beta > 0 else None
                terms = grpo_loss(
                    logp_new=logp_new,
                    logp_old=[torch.tensor(t.logp, device=device) for t in chunk],
                    kinds=[torch.tensor(t.kinds, device=device) for t in chunk],
                    seq_adv=seq_adv[chunk_ids].to(device),
                    opt_adv=[opt_adv[i].to(device) for i in chunk_ids],
                    logp_ref=logp_ref,
                    clip_eps=grpo.clip_eps,
                    kl_beta=grpo.kl_beta,
                    option_weight=grpo.option_loss_weight,
                    length_normalize=grpo.length_normalize,
                    n_sequences=n_sequences,
                    n_actions=n_actions,
                )
                self.acc.backward(terms.loss)
                epoch_loss += float(terms.loss.detach())
                stats["kl"] = stats["kl"] + terms.kl
                stats["surrogate_seq"] += terms.surrogate_seq
                stats["surrogate_opt"] += terms.surrogate_opt
                clip_weighted += terms.clip_fraction * terms.n_actions
            if not self.is_deepspeed:
                params = [p for p in self.raw_policy.parameters() if p.requires_grad]
                norm = self.acc.clip_grad_norm_(params, self.cfg.optim.max_grad_norm)
                stats["grad_norm"] = float(norm) if norm is not None else 0.0
            self.optimizer.step()
            stats["loss"] = epoch_loss
            stats["clip_fraction"] = clip_weighted / max(n_actions, 1)
        self.optimizer.zero_grad(set_to_none=True)
        epochs = max(grpo.ppo_epochs, 1)
        stats["kl"] = stats["kl"] / max(n_actions * epochs, 1)
        stats["surrogate_seq"] /= max(n_sequences * epochs, 1)
        stats["surrogate_opt"] /= max(n_sequences * epochs, 1)
        return stats

    def train_step(self, phase: Phase) -> dict[str, Any]:
        started = time.time()
        total_steps = phase.steps if self.cfg.optim.schedule_scope == "phase" else sum(p.steps for p in self.phases)
        position = self.state.step if self.cfg.optim.schedule_scope == "phase" else self.state.global_step
        factor = schedule_factor(position, total_steps, self.cfg.optim)
        set_learning_rates(self.optimizer, factor)

        groups = self.collect(phase)
        trajectories, seq_adv, opt_adv = self.advantages(groups)
        rewards = [r for g in groups for r in g.rewards]
        informative = bool(seq_adv.abs().sum() > 0) or any(bool(v.abs().sum() > 0) for v in opt_adv)
        if informative or (self.cfg.grpo.kl_beta > 0 and self.reference.enabled):
            stats = self.update(trajectories, seq_adv, opt_adv)
        else:
            stats = {"loss": 0.0, "kl": 0.0, "clip_fraction": 0.0, "surrogate_seq": 0.0, "surrogate_opt": 0.0, "grad_norm": 0.0}

        pass_rate = sum(r.passed for r in rewards) / max(len(rewards), 1)
        advanced = False
        if phase.adaptive:
            advanced = self.pacer.update(pass_rate)
            if advanced and self._sampler is not None:
                self._sampler.set_tasks(phase_tasks(self.tasks, phase, "train", self.pacer.length))
                log.info("%s: composition length advanced to %d", phase.name, self.pacer.length)
        self.state.step += 1
        self.state.global_step += 1
        n_decisions = sum(len(t.calls) for t in trajectories)
        completed = sum(1 for t in trajectories for c in t.calls if c.completed)
        stage_counts: dict[str, int] = {}
        for r in rewards:
            stage_counts[r.stage] = stage_counts.get(r.stage, 0) + 1
        return {
            "phase": phase.number,
            "level": phase.level,
            "step": self.state.step,
            "global_step": self.state.global_step,
            "lr": self.cfg.optim.lr * factor,
            "reward": sum(r.total for r in rewards) / max(len(rewards), 1),
            "reward_exec": sum(r.exec for r in rewards) / max(len(rewards), 1),
            "reward_quality": sum(r.quality for r in rewards) / max(len(rewards), 1),
            "reward_consistency": sum(r.consistency for r in rewards) / max(len(rewards), 1),
            "pass_rate": pass_rate,
            "tokens": sum(t.n_tokens for t in trajectories) / max(len(trajectories), 1),
            "option_decisions": n_decisions / max(len(trajectories), 1),
            "option_completion": completed / n_decisions if n_decisions else 0.0,
            "composition_length": self.pacer.length if phase.adaptive else None,
            "advanced": advanced,
            "updated": informative,
            "stages": stage_counts,
            "seconds": time.time() - started,
            **stats,
        }

    # ------------------------------------------------------------------ certification
    def certify(self, level: int, action_level: int) -> tuple[dict[str, Any], list[tuple[Task, Trajectory]]]:
        """Algorithm 2: sample solutions for the tasks of ``level`` and certify options.

        Returns the certification summary and the successful ``(task, trajectory)``
        pairs of this process, which feed the option extraction step.
        """
        cert = self.cfg.certification
        tasks = self.tasks.select(level=level, split="train")
        if cert.max_tasks is not None:
            tasks = tasks[: cert.max_tasks]
        shard = tasks[int(self.acc.process_index) :: max(int(self.acc.num_processes), 1)]
        initiation = self.initiation_for(action_level)
        records: list[SolutionRecord] = []
        successes: list[tuple[Task, Trajectory]] = []
        for task in shard:
            trajectories = self.sample_group(task, cert.samples_per_task, initiation, cert.temperature)
            programs = [extract_solution(t.text) for t in trajectories]
            (breakdowns,) = self.rewards.score([(task, programs)], level=0)  # R_exec only
            for trajectory, program, breakdown in zip(trajectories, programs, breakdowns):
                analysis = analyze_program(program, self.index, task.context_code())
                primary = analysis.primary if analysis.ok else []
                records.append(
                    SolutionRecord(task.id, breakdown.exec, [c.name for c in primary], [c.func for c in primary])
                )
                if breakdown.exec >= cert.success_threshold:
                    successes.append((task, trajectory))
        if int(self.acc.num_processes) > 1:
            records = self.acc.gather_for_metrics(records, use_gather_object=True)
        summary = certify_options(self.vocab, level, records, cert, self.graphs)
        for option in self.vocab.level(level):
            self.constraints.register(option)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.vocab.save(self.output_dir / "options.json")
        return summary, successes

    def extract(self, action_level: int, successes: list[tuple[Task, Trajectory]]) -> dict[str, Any]:
        """Fit the policy to option-level rewrites of its own successful solutions.

        ``action_level`` is the level of the action space that the newly certified
        options extend (see :mod:`option_grpo.options.extraction`).
        """
        cert = self.cfg.certification
        initiation = self.initiation_for(action_level)
        pool: list[Trajectory] = []
        if initiation is not None and cert.extraction_steps > 0:
            for task, trajectory in successes:
                rewritten = annotate_trajectory(
                    trajectory, self.index, self.vocab, self.constraints, initiation,
                    self.raw_policy.base_vocab_size, self.engine.stop_token_ids, task.context_code(),
                    task.setup_value_types(), self.cfg.decode.max_option_tokens,
                )
                if rewritten is not None:
                    pool.append(rewritten)
        sizes = [len(pool)]
        if int(self.acc.num_processes) > 1:
            sizes = self.acc.gather_for_metrics(sizes, use_gather_object=True)
        info = {"trajectories": int(sum(sizes)), "steps": 0, "loss": None}
        if min(sizes) == 0 or cert.extraction_steps <= 0:
            log.info("option extraction skipped: %s", info)
            return info
        rng = random.Random(self.cfg.seed + 31 * action_level + int(self.acc.process_index))
        set_learning_rates(self.optimizer, 1.0)
        self.policy.train()
        last = 0.0
        for _ in range(cert.extraction_steps):
            batch = rng.sample(pool, k=min(cert.extraction_batch, len(pool)))
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logp = self.engine.log_probs(batch, self.policy)
            total = sum(int(lp.numel()) for lp in logp)
            loss = -torch.stack([lp.sum() for lp in logp]).sum() / max(total, 1)
            self.acc.backward(loss)
            if not self.is_deepspeed:
                params = [p for p in self.raw_policy.parameters() if p.requires_grad]
                self.acc.clip_grad_norm_(params, self.cfg.optim.max_grad_norm)
            self.optimizer.step()
            last = float(loss.detach())
        self.optimizer.zero_grad(set_to_none=True)
        info.update(steps=cert.extraction_steps, loss=last, decisions=sum(len(t.calls) for t in pool))
        log.info("option extraction: %s", info)
        return info

    # ------------------------------------------------------------------ logging / io
    def _init_tracking(self) -> None:
        project = self.cfg.logging.wandb_project
        if project and self.is_main and self._wandb is None:
            try:
                import wandb

                self._wandb = wandb.init(project=project, config=self.cfg.to_dict(), dir=str(self.output_dir))
            except ImportError:
                log.warning("wandb is not installed; continuing without it")

    def _log(self, phase: Phase, metrics: dict[str, Any]) -> None:
        if not self.is_main:
            return
        append_jsonl(self.output_dir / "metrics.jsonl", metrics)
        if self._wandb is not None:
            self._wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=metrics["global_step"])
        every = max(self.cfg.logging.log_every, 1)
        if metrics["step"] % every == 0 or metrics["step"] == phase.steps:
            log.info(
                "%s step %d/%d | reward %.3f | pass %.2f | tokens %.0f | options %.2f | loss %.4f | kl %.4f | %.1fs",
                phase.name, metrics["step"], phase.steps, metrics["reward"], metrics["pass_rate"], metrics["tokens"],
                metrics["option_decisions"], metrics["loss"], metrics["kl"], metrics["seconds"],
            )

    def save_checkpoint(self, name: str) -> Path | None:
        if not self.is_main:
            return None
        directory = self.output_dir / "checkpoints" / name
        directory.mkdir(parents=True, exist_ok=True)
        if self.is_deepspeed:
            state = self.acc.get_state_dict(self.policy)
            lm_state = {k[len("lm."):]: v for k, v in state.items() if k.startswith("lm.")}
            self.raw_policy.lm.save_pretrained(directory / "lm", state_dict=lm_state)
            torch.save(
                {
                    "option_in": state["option_in.weight"].float().cpu(),
                    "option_out": state["option_out.weight"].float().cpu(),
                    "option_bias": state["option_out.bias"].float().cpu(),
                },
                directory / "options.pt",
            )
        else:
            self.raw_policy.save(directory)
        self.tokenizer.save_pretrained(directory / "lm")
        self.vocab.save(directory / "options.json")
        self.state.pacer = self.pacer.state()
        write_json(directory / "state.json", asdict(self.state))
        save_config(self.cfg, directory / "config.yaml")
        log.info("saved checkpoint %s", directory)
        return directory

    def load_state(self, directory: str | Path) -> None:
        """Restore curriculum progress (weights are restored when the policy is built)."""
        path = Path(directory) / "state.json"
        if not path.exists():
            return
        data = read_json(path)
        self.state = TrainState(**{k: v for k, v in data.items() if k in TrainState.__dataclass_fields__})
        self.pacer.load_state(self.state.pacer)
        completed = self.state.phase
        if completed > 0:
            apply_partial_freeze(self.optimizer, self.raw_policy, completed, self.cfg.curriculum.freeze, self.cfg.optim)


def relative_gradient_variance(metrics_path: str | Path, window: int = 200) -> list[dict[str, float]]:
    """Per-window relative gradient-norm variance ``Var(||g||) / E[||g||]^2``.

    This is the quantity plotted in the gradient-variance analysis; it equals the
    inverse squared signal-to-noise ratio of the gradient norm.
    """
    norms: list[tuple[int, float]] = []
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("updated") and row.get("grad_norm"):
                norms.append((int(row["global_step"]), float(row["grad_norm"])))
    out: list[dict[str, float]] = []
    for start in range(0, len(norms), window):
        chunk = [n for _, n in norms[start : start + window]]
        if len(chunk) < 2:
            continue
        mean = sum(chunk) / len(chunk)
        var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
        out.append({"step": norms[start][0], "mean": mean, "relative_variance": var / (mean * mean) if mean else math.nan})
    return out
