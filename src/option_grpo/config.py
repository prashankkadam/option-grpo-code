"""Typed configuration.

Defaults reproduce the hyperparameter table of the paper (Appendix B.2).  Every value
can be overridden from a YAML file or from the command line with dotted keys, e.g.
``-o grpo.group_size=8 -o model.name_or_path=Qwen/Qwen3-8B-Base``.
"""

from __future__ import annotations

import copy
import dataclasses
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@dataclass
class LoraConfig:
    """Optional low-rank adaptation.  The paper fine-tunes all parameters (disabled)."""

    enabled: bool = False
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-32B-Base"
    revision: str | None = None
    dtype: str = "bf16"  # bf16 | fp16 | fp32: precision of the loaded weights
    autocast: str = "none"  # none | bf16 | fp16: autocast precision of the forward passes
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    # Stop conditions are always passed explicitly to the sampler.
    stop_strings: list[str] = field(default_factory=lambda: ["# END SOLUTION"])
    extra_stop_token_ids: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# GRPO
# --------------------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    group_size: int = 16  # G
    temperature: float = 0.8  # sampling temperature during training
    clip_eps: float = 0.2  # PPO clip epsilon
    kl_beta: float = 0.04  # KL penalty against the reference policy
    option_loss_weight: float = 0.3  # lambda, weight of the option-level term
    adv_eps: float = 1e-4  # epsilon in the group-normalised advantage
    normalize_option_advantage: bool = False
    length_normalize: bool = False  # divide each sequence term by its length
    ppo_epochs: int = 1  # gradient steps per sampled batch
    prompts_per_step: int = 1  # prompts sampled per process and step
    train_micro_batch: int = 4  # sequences per backward pass
    rollout_chunk: int = 16  # sequences decoded in parallel
    max_new_tokens: int = 384
    # Reference policy for the KL term: "copy" keeps a frozen snapshot taken at the start
    # of each phase, "adapter_off" disables LoRA adapters, "none" drops the KL term.
    reference: str = "copy"
    reference_refresh: str = "phase"  # phase | never


@dataclass
class OptionDecodeConfig:
    """Decode-time behaviour of options (initiation, intra-option masking, termination)."""

    max_option_tokens: int = 96  # an active option is aborted after this many tokens
    initiation: str = "last_value"  # last_value | scope | any
    option_logit_bias: float = 0.0  # initial value of the learnable option-logit bias
    min_initiation_compat: float = 0.5
    strict_chain: bool = True  # chain-level attribute names must be path or namespace names
    mask_cache_size: int = 50_000


# --------------------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------------------
@dataclass
class RewardConfig:
    quality_weight: float = 0.2  # lambda_q
    consistency_weight: float = 0.1  # lambda_c
    pattern_bonus: float = 0.1  # alpha_1
    no_redundancy_bonus: float = 0.1  # alpha_2
    minimal_api_bonus: float = 0.1  # alpha_3
    reimplementation_penalty: float = 0.1  # alpha_4
    # Quality and consistency terms are only granted to programs whose execution reward
    # reaches this value (0.5 = "executes"); set to 0.0 to always grant them.
    quality_gate: float = 0.5
    # Layered execution reward (Section 3.7).
    r_parse_fail: float = 0.0
    r_import_fail: float = 0.1
    r_runtime_error: float = 0.3
    r_executes: float = 0.5
    r_type_match: float = 0.7
    r_tests_pass: float = 1.0
    # First curriculum level at which each auxiliary term is switched on (Algorithm 4).
    quality_from_level: int = 1
    consistency_from_level: int = 2


# --------------------------------------------------------------------------------------
# Optimiser and schedule
# --------------------------------------------------------------------------------------
@dataclass
class OptimConfig:
    lr: float = 5e-5  # policy learning rate
    option_lr: float = 5e-6  # learning rate of partially frozen parameters
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    schedule: str = "cosine"  # cosine | constant
    schedule_scope: str = "phase"  # phase: restart per curriculum phase | global
    min_lr_ratio: float = 0.0


# --------------------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------------------
@dataclass
class FreezeConfig:
    """Partial freeze applied after each curriculum phase.

    After phase ``k`` the bottom ``bottom_fraction[k]`` of the transformer blocks moves to
    the slow parameter group (``mode="slow"``, learning rate ``optim.option_lr``) or stops
    receiving gradients altogether (``mode="hard"``).
    """

    mode: str = "slow"  # slow | hard | none
    bottom_fraction: list[float] = field(default_factory=lambda: [0.25, 0.5])
    freeze_token_embeddings: bool = True


@dataclass
class CurriculumConfig:
    steps: list[int] = field(default_factory=lambda: [5_000, 10_000, 15_000])  # N_0, N_1, N_2
    advance_threshold: float = 0.5  # rho_adv
    advance_window: int = 50  # training steps over which the pass rate is averaged
    start_length: int = 2  # initial l_curr in phase 2
    max_composition_length: int = 4  # L_max
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    # Ablation switches (Table 9).
    use_options: bool = True  # False -> token-level action space at every level
    use_curriculum: bool = True  # False -> a single phase over the union of all levels


# --------------------------------------------------------------------------------------
# Graphs, option discovery and certification
# --------------------------------------------------------------------------------------
@dataclass
class GraphConfig:
    min_compat: float = 0.5  # minimum type compatibility for an edge of G_flow
    semantic_threshold: float = 0.7  # tau, edges of G_sem
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"  # or "hashing"
    include_private: bool = False
    include_properties: bool = False
    exclude_dirs: list[str] = field(
        default_factory=lambda: ["tests", "test", "testing", "benchmarks", "examples", "docs", "_vendor", "vendor"]
    )


@dataclass
class DiscoveryConfig:
    max_path_length: int = 4  # L_max
    min_path_length: int = 2
    cluster_threshold: float = 0.8  # tau_cluster
    min_path_weight: float = 0.25  # prune paths whose weight product falls below this
    max_branch: int = 12  # successors expanded per node, by decreasing weight
    max_paths_per_start: int = 256
    max_paths: int = 6_000  # cap on the number of paths that enter clustering
    merge_subsumed: bool = True  # enforce option non-subsumption


@dataclass
class CertificationConfig:
    samples_per_task: int = 10
    temperature: float = 0.8
    success_threshold: float = 0.7  # rho
    frequency_threshold: int = 5  # nu
    # Level 0 has one task per function, so a pattern is required to occur in at least
    # ``frequency_threshold`` successful samples but only ``level0_min_tasks`` tasks.
    level0_min_tasks: int = 1
    min_invocation_success_rate: float = 0.0
    allow_novel_patterns: bool = False
    max_tasks: int | None = None  # subsample tasks for certification (None = all)
    # Option extraction: after certification the successful solutions are re-expressed as
    # option-level trajectories and the policy is fitted to them for a few steps, which
    # gives every newly certified option a calibrated initiation probability.
    extraction_steps: int = 200
    extraction_batch: int = 8


# --------------------------------------------------------------------------------------
# Benchmark, execution, evaluation
# --------------------------------------------------------------------------------------
@dataclass
class BenchmarkConfig:
    level1_per_function: float = 2.0  # Level-1 tasks per public function
    level2_tasks: int = 33  # Level-2 tasks per SDK
    level2_min_calls: int = 4
    level2_max_calls: int = 8
    test_fraction: float = 0.2  # held-out share of tasks per level
    max_tests_per_function: int = 15
    # Function-level properties must be exercised: a program that never calls a function
    # of the task's path cannot pass that function's property tests vacuously.
    require_calls: bool = True
    reference_equivalence: bool = False  # add a differential test against the skeleton
    seed: int = 0
    odl_max_options: int = 8  # m_max


@dataclass
class ExecutionConfig:
    backend: str = "subprocess"  # subprocess | docker
    timeout_s: float = 30.0
    memory_mb: int = 2048
    allow_network: bool = False
    workers: int = 8
    python: str | None = None  # interpreter used by the subprocess backend
    docker_image: str | None = None
    docker_extra_args: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    n_samples: int = 10
    temperature: float = 0.8
    ks: list[int] = field(default_factory=lambda: [1, 5, 10])
    max_new_tokens: int = 384
    split: str = "test"
    retrieval: str = "none"  # none | rag | docprompting
    retrieval_top_k: int = 10
    retriever_model: str = "microsoft/codebert-base"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_every: int = 10
    eval_every: int = 0  # 0 disables periodic evaluation
    save_every: int = 1_000
    wandb_project: str | None = None
    track_grad_norm: bool = True


@dataclass
class Config:
    sdk: str = "daft"
    seed: int = 0
    output_dir: str = "runs/default"
    artifacts_dir: str = "artifacts"
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    decode: OptionDecodeConfig = field(default_factory=OptionDecodeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    certification: CertificationConfig = field(default_factory=CertificationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------ convenience
    def sdk_artifacts(self) -> Path:
        """Directory holding the Stage-0 artefacts (index, graphs, options, tasks) of the SDK."""
        return Path(self.artifacts_dir) / self.sdk

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if self.grpo.group_size < 2:
            raise ValueError("grpo.group_size must be at least 2 for group-relative advantages")
        if len(self.curriculum.steps) != 3:
            raise ValueError("curriculum.steps must list the step budget of the three phases")
        if not 0.0 <= self.benchmark.test_fraction < 1.0:
            raise ValueError("benchmark.test_fraction must lie in [0, 1)")
        if self.grpo.reference not in {"copy", "adapter_off", "none"}:
            raise ValueError(f"unknown grpo.reference: {self.grpo.reference}")
        if self.grpo.reference == "adapter_off" and not self.model.lora.enabled:
            raise ValueError("grpo.reference=adapter_off requires model.lora.enabled=true")
        if self.decode.initiation not in {"last_value", "scope", "any"}:
            raise ValueError(f"unknown decode.initiation: {self.decode.initiation}")
        if self.curriculum.freeze.mode not in {"slow", "hard", "none"}:
            raise ValueError(f"unknown curriculum.freeze.mode: {self.curriculum.freeze.mode}")
        if self.execution.backend not in {"subprocess", "docker"}:
            raise ValueError(f"unknown execution.backend: {self.execution.backend}")
        if self.model.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"unknown model.dtype: {self.model.dtype}")
        if self.model.autocast not in {"none", "bf16", "fp16"}:
            raise ValueError(f"unknown model.autocast: {self.model.autocast}")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _coerce(tp: Any, value: Any, path: str) -> Any:
    """Coerce ``value`` (parsed YAML) to the annotated type ``tp``."""
    origin = get_origin(tp)
    if tp is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        last_error: Exception | None = None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, path)
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise TypeError(f"{path}: cannot interpret {value!r} as {tp}") from last_error
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _from_dict(tp, value, path)
    if origin in (list, typing.List):  # noqa: UP006
        (inner,) = get_args(tp) or (Any,)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (tuple, typing.Tuple):  # noqa: UP006
        inner_types = get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path}: expected a sequence, got {type(value).__name__}")
        if len(inner_types) == 2 and inner_types[1] is Ellipsis:
            return tuple(_coerce(inner_types[0], v, path) for v in value)
        if len(inner_types) != len(value):
            raise TypeError(f"{path}: expected {len(inner_types)} values, got {len(value)}")
        return tuple(_coerce(t, v, path) for t, v in zip(inner_types, value))
    if origin in (dict, typing.Dict):  # noqa: UP006
        if not isinstance(value, dict):
            raise TypeError(f"{path}: expected a mapping, got {type(value).__name__}")
        return dict(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"{path}: expected a boolean, got {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise TypeError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool):
            raise TypeError(f"{path}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)  # YAML parses "5e-5" as a string
        raise TypeError(f"{path}: expected a number, got {value!r}")
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        prefix = f"{path}." if path else ""
        raise KeyError(f"unknown configuration key(s): {', '.join(sorted(prefix + k for k in unknown))}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, f"{path}.{name}" if path else name)
    return cls(**kwargs)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_override(tree: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must look like key.subkey=value, got {override!r}")
    key, raw = override.split("=", 1)
    node = tree
    parts = key.strip().split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(
    paths: str | Path | list[str | Path] | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Build a :class:`Config` from defaults, YAML files (later files win) and overrides."""
    tree: dict[str, Any] = Config().to_dict()
    if paths is not None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for p in paths:
            with Path(p).open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise TypeError(f"{p}: top level of a config file must be a mapping")
            _deep_update(tree, loaded)
    for override in overrides or []:
        _apply_override(tree, override)
    cfg = _from_dict(Config, tree)
    cfg.validate()
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
