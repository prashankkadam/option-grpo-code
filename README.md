# Option-GRPO

Code for **Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs**.

Option-GRPO adapts a code LLM to an SDK it has never seen, using only the SDK's source
code: no usage examples and no hand-written tests. It optimises two things at once,
functional correctness and code elegance, so that the adapted model writes short,
idiomatic programs instead of verbose, redundant ones ("AI slop").

## The idea in brief

SDK source code has a compositional structure. Single API calls are atomic actions,
commonly composed call sequences are reusable patterns, and complete programs are
compositions of those patterns. This is the structure of the *options* framework in
hierarchical reinforcement learning, and the paper builds the whole method on that
correspondence:

1. **Graphs from source.** The SDK is analysed statically into a type graph `G_api`, a
   parameter-flow graph `G_flow` (an edge `f_a -> f_b` when the value returned by `f_a`
   can be passed to `f_b`) and a semantic similarity graph `G_sem`.
2. **Code options.** A code option is a 4-tuple `o = (I, F, beta, pi)`: an initiation set
   given by the input types of its first function, an allowed function set `F`, a
   termination predicate that fires once a value of the output type is produced, and an
   intra-option policy that is the shared network restricted to `F` by masking logits.
   Level-0 options are single SDK functions. Level-1 candidates are mined from `G_flow`
   by enumerating composable paths, clustering them and keeping one path per cluster
   (Algorithm 1).
3. **Curriculum as option construction.** Training runs in three phases. Each phase
   trains with GRPO on tasks of one difficulty level, then *certifies* the options the
   policy has shown it can invoke (Algorithm 2). Certified options become actions of the
   next phase, so the action space grows as `A_0 ⊂ A_1 ⊂ A_2`.
4. **Hierarchical credit assignment.** Next to the usual group-relative advantage of a
   whole sample, every option gets its own advantage: the mean reward of the samples in
   the group that used the option minus the mean reward of those that did not.
5. **Reward without test cases.** Programs are executed in a sandbox and scored on a
   ladder (parses, imports, runs, returns a value, returns the expected type, passes
   property-based tests generated from type annotations, Algorithm 3), plus small
   bonuses for code quality and for agreement within the sampled group.

The paper shows that code options induce a valid semi-Markov decision process and that
the optimal value does not decrease as the curriculum adds levels. It gives conditional
bounds under which a shorter option description length implies shorter and, in
expectation, more API-efficient programs, and it analyses, under stated assumptions, how
option-level credit assignment improves the gradient signal-to-noise ratio by a factor
of `sqrt(T/K)`.

### Results reported in the paper

On UnseenSDK-Bench (12 SDKs, Qwen3-32B base model), averaged over SDKs:

| Method | pass@1 | pass@10 | Code length ↓ | API calls ↓ | Idiomaticity ↑ |
|---|---|---|---|---|---|
| Frozen LLM | 28.5 | 42.1 | 94.6 | 7.8 | 0.18 |
| RAG | 41.3 | 56.8 | 88.1 | 6.4 | 0.31 |
| ExploraCoder | 50.2 | 67.5 | 78.4 | 4.9 | 0.43 |
| ReCode | 55.1 | 71.2 | 74.1 | 4.6 | 0.51 |
| Flat GRPO | 52.7 | 68.4 | 71.3 | 4.9 | 0.41 |
| Flat GRPO + curriculum | 55.4 | 71.0 | 66.8 | 4.4 | 0.49 |
| **Option-GRPO** | **61.4** | **76.8** | **52.0** | **3.2** | **0.74** |

These numbers are quoted from the paper. This repository contains the method, the
benchmark construction and the evaluation harness needed to run such experiments.

## Repository layout

```
src/option_grpo/
  sdk/          static analysis of SDK source (types, functions, classes), SDK registry
  graphs/       G_api, G_flow, G_sem and identifier embeddings
  options/      code options, discovery (Alg. 1), certification (Alg. 2), extraction,
                decode-time constraints, initiation sets, option description length
  analysis/     API-call sequences of programs, redundancy, re-implementation, quality
  tasks/        input synthesis, curriculum task generation, validation, prompts
  proptest/     property-based test generation (Alg. 3)
  execution/    sandbox back-ends and the stand-alone execution harness
  rewards/      layered execution reward, quality reward, self-consistency reward
  policy/       hierarchical policy over the extended action space, rollouts
  grpo/         advantages, objective, optimiser and schedule, curriculum, trainer (Alg. 4)
  eval/         pass@k and quality metrics, retrieval baselines, API back-end, probe
  pipeline.py   Stage 0 orchestration        cli.py   the `option-grpo` command
configs/        default (paper) configuration, model presets, ablations, ZeRO-3 setup
examples/       tinyframe, a small typed SDK used for the quick start and the tests
docker/         sandbox image       scripts/   end-to-end run scripts       tests/
```

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/prashankkadam/option-grpo-code.git
cd option-grpo-code
pip install -e .                 # core
pip install -e ".[embed]"        # sentence-transformer embeddings for option discovery
pip install -e ".[lora,zero3]"   # optional: LoRA, DeepSpeed ZeRO-3
pip install -e ".[dev]"          # tests and linting
```

The SDK you want to adapt to must be installed in the environment that executes the
generated programs (your environment for the subprocess sandbox, the image for the
Docker sandbox).

## Quick start

The repository ships `tinyframe`, a miniature typed dataframe library, so the complete
pipeline can be run without installing a real SDK.

```bash
# Stage 0: graphs, candidate options, tasks and property tests
option-grpo prepare -c configs/smoke.yaml --sdk-registry examples/tinyframe/sdk.yaml

# Three training phases with a 0.6B base model and a tiny step budget, then evaluation
option-grpo train -c configs/smoke.yaml --sdk-registry examples/tinyframe/sdk.yaml
```

`configs/smoke.yaml` uses a handful of steps per phase. It exercises every component
and is meant as a functional check, not as a training recipe.

Stage 0 writes its artefacts to `artifacts/<sdk>/`:

| File | Content |
|---|---|
| `index.json` | public functions, methods and constructors with resolved types |
| `graphs.json`, `embeddings.npz` | `G_flow`, `G_sem`, identifier embeddings |
| `options.json` | candidate options `O_0^cand` and `O_1^cand` |
| `tasks.jsonl` | curriculum tasks `D_0`, `D_1`, `D_2` with their property tests |
| `stats.json` | function, task and test counts, option statistics, composability density |
| `skipped.json` | paths for which no task could be built, and tasks dropped by validation |

A training run writes `metrics.jsonl` (one row per step), `options.json` (the vocabulary
with certification status) and checkpoints `phase1`, `phase2`, `phase3`, `final` under
`<output_dir>/checkpoints/`.

## Running the method

```bash
# 1. Stage 0 for an SDK of the benchmark
option-grpo prepare -c configs/default.yaml --sdk daft

# 2. Algorithm 4: three curriculum phases (single process)
option-grpo train -c configs/default.yaml -c configs/models/qwen3_8b.yaml --sdk daft \
    --output-dir runs/daft/seed0

# 3. Evaluation: 10 samples per task at temperature 0.8
option-grpo evaluate -c configs/default.yaml -c configs/models/qwen3_8b.yaml --sdk daft \
    --checkpoint runs/daft/seed0/checkpoints/final --output-dir runs/daft/seed0
```

`scripts/run_sdk.sh <sdk> [seed]` runs the three steps, `scripts/run_benchmark.sh` loops
over the twelve SDKs and three seeds, and `scripts/run_baselines.sh <sdk>` runs the
baselines and ablations listed below.

### Multi-GPU

Training is built on `accelerate`. Every process samples its own prompts, executes the
programs locally and contributes gradients.

```bash
# DDP
accelerate launch --multi_gpu --num_processes 8 -m option_grpo.cli train \
    -c configs/default.yaml -c configs/models/qwen3_8b.yaml --sdk daft

# DeepSpeed ZeRO-3 + bf16 (the 32B setting of the paper)
accelerate launch --config_file configs/accelerate/zero3.yaml -m option_grpo.cli train \
    -c configs/default.yaml -c configs/models/qwen3_32b.yaml --sdk daft
```

Notes on memory:

* `grpo.reference=copy` keeps a frozen snapshot of the policy for the KL term. It needs
  room for a second copy of the model and is not available under ZeRO-3.
* `grpo.reference=adapter_off` (with `model.lora.enabled=true`) uses the base model with
  adapters disabled as the reference and needs no second copy.
* `grpo.reference=none` drops the KL term.
* Under ZeRO-3 each optimiser step uses a single backward pass over the whole group
  (`grpo.train_micro_batch` is ignored) and gradient clipping is taken from the
  DeepSpeed configuration.

The checkpoint name `Qwen/Qwen3-32B` is the model stated in the paper. If it is not
available to you, `configs/models/qwen3_14b.yaml` and `configs/models/qwen3_8b.yaml` use
the smaller dense base models of the same family (the size sweep of the paper).

### Sandboxing

Generated programs are untrusted code. The default `subprocess` back-end runs each
program in a fresh interpreter with a timeout, an address-space limit (Linux) and
sockets disabled inside the process. For real isolation build one image per SDK and
switch to the Docker back-end, which runs every program in its own container with no
network, a memory limit and a read-only root file system:

```bash
scripts/build_sandbox.sh daft "daft==<version>"
option-grpo train ... -o execution.backend=docker -o execution.docker_image=option-grpo-sandbox:daft
```

### Baselines, ablations and analyses

| What | Command |
|---|---|
| Frozen base model | `option-grpo baseline --method frozen` |
| RAG (top-10 CodeBERT retrieval over SDK source) | `option-grpo baseline --method rag` |
| DocPrompting (retrieval over documentation) | `option-grpo baseline --method docprompting` |
| Flat GRPO | `option-grpo train -c configs/ablations/flat_grpo.yaml` |
| Flat GRPO + curriculum | `option-grpo train -c configs/ablations/curriculum_only.yaml` |
| Options without curriculum | `option-grpo train -c configs/ablations/options_only.yaml` |
| Option-GRPO + retrieval at inference | `option-grpo evaluate --retrieval rag` |
| Cross-SDK transfer | `option-grpo train --init-from <checkpoint of the source SDK>` |
| Hosted models (OpenAI-compatible endpoint) | `option-grpo api --model-name <label> --model <id> --base-url <url>` |
| Memorisation probe (contamination check) | `option-grpo probe --sdk <sdk>` |
| ODL vs. quality correlations (Spearman) | `option-grpo analyze odl <eval dir>` |
| Relative gradient-norm variance | `option-grpo analyze gradvar <run dir>` |
| Training curve | `option-grpo analyze curve <run dir>` |

Checkpoints trained without options are evaluated with `option-grpo evaluate --flat`.
Pass `--options <run>/checkpoints/final/options.json` to a baseline so that its
idiomaticity is measured against the options certified by the Option-GRPO run.

The critic-based baselines of the paper (CodeRL, PPOCoder, StepCoder) and Gorilla,
ExploraCoder and ReCode are separate methods and are not part of this repository. The
`Generator` interface in `eval/evaluator.py` (one method, `generate(task, n, ...)`) is
the integration point for evaluating any other system on the benchmark.

## From the paper to the code

| Paper | Code |
|---|---|
| `G_api`, `G_flow`, `G_sem`, `type_compat` | `graphs/build.py`, `sdk/types.py` |
| Definition 1, code option | `options/option.py` |
| Algorithm 1, Level-1 option discovery | `options/discovery.py` |
| Intra-option policy by logit masking, termination | `options/constraints.py` |
| Initiation sets, composability graph `G_opt` | `options/tracker.py`, `options/option.py` |
| Algorithm 2, certification | `options/certification.py`, `options/extraction.py` |
| Algorithm 3, property tests | `proptest/generator.py`, `execution/harness.py` |
| Layered reward, quality and consistency terms | `rewards/reward.py` |
| Standard and option-level GRPO | `grpo/advantages.py`, `grpo/loss.py` |
| Algorithm 4, training procedure | `grpo/trainer.py`, `grpo/curriculum.py` |
| Definitions 2 to 4, ODL and quality metrics | `options/odl.py`, `analysis/quality.py` |
| Metrics (pass@k, CL, AE, ID, OR) | `eval/evaluator.py`, `eval/stats.py` |
| Contamination check | `eval/probe.py` |
| Hyperparameters (Table 3) | `config.py`, `configs/default.yaml` |

### How options behave at decode time

The policy is a causal LM whose action space is the token vocabulary extended with one
action per option. Option actions have their own input and output vectors, initialised
from the tokens that spell the option's call path.

* **Initiation.** Outside an option the policy chooses between a token and any option
  whose first function accepts the type of the last value produced. Options that need
  no SDK object are always available. Choosing an option emits no text.
* **Masking.** While an option is active, a character-level automaton tracks strings,
  comments, bracket nesting, the identifier being typed and the progress along the
  option's call path. On the main call chain an SDK call must be the next function of
  the path. Inside argument lists only the option's helper functions are allowed (the
  type constructors of its parameters, for example `col` inside `where`). Other code is
  unconstrained. Tokens that would break these rules get a logit of `-inf`.
* **Termination.** The option ends when the bracket of its last call closes. An option
  that exceeds `decode.max_option_tokens` or reaches a state with no admissible token is
  aborted and does not count as completed.

Each step of a rollout stores a compact description of its mask, so the exact sampling
distribution is re-evaluated under the updated parameters during training.

### Reward

`R(c) = R_exec(c) + λ_q R_quality(c) + λ_c R_consistency(c)` with `λ_q = 0.2`,
`λ_c = 0.1`.

| `R_exec` | Stage reached |
|---|---|
| 0.0 | does not parse (or contains no statement) |
| 0.1 | parses, an import fails |
| 0.3 | runtime error, timeout or crash |
| 0.5 | executes (no output, or output of another type) |
| 0.7 | output of the expected type |
| 1.0 | passes the property-based tests |

Phase 1 uses `R_exec` only, phase 2 adds the quality term (pattern use, no redundant
variables, minimal number of API calls, no re-implementation), phase 3 adds the
self-consistency term.

## Configuration

`configs/default.yaml` lists the main settings with the names used in the paper. Any key
of `option_grpo/config.py` can be set in a YAML file or with `-o key.subkey=value`.

| Group | Setting | Default |
|---|---|---|
| GRPO | group size `G`, temperature, clip `ε`, KL `β`, option weight `λ` | 16, 0.8, 0.2, 0.04, 0.3 |
| Reward | `λ_q`, `λ_c`, `α_1..α_4` | 0.2, 0.1, 0.1 each |
| Optimiser | AdamW, policy lr, option fine-tune lr, weight decay, grad clip, warm-up | 5e-5, 5e-6, 0.01, 1.0, 500, cosine |
| Curriculum | `N_0`, `N_1`, `N_2`, `ρ_adv`, `L_max` | 5000, 10000, 15000, 0.5, 4 |
| Discovery | `τ`, `τ_cluster` | 0.7, 0.8 |
| Certification | samples per task, `ρ`, `ν` | 10, 0.7, 5 |
| Execution | timeout, memory | 30 s, 2 GB |

## Implementation notes

The paper fixes the method and its hyperparameters. The points below are the places
where an implementation has to choose a concrete mechanism. Each one is a configuration
option or a clearly separated function.

* **Type compatibility.** `type_compat` scores exact class matches 1.0, subclasses 0.9,
  membership in a multi-class union 0.8, and unknown or `Any` types 0.1. Flows between
  primitives (`str -> str`) score 0 because they carry no structural signal. Edges of
  `G_flow` require a score of at least `graph.min_compat = 0.5`.
* **Path enumeration.** Depth-first search expands the strongest successors of each node
  and keeps the heaviest paths (`discovery.max_branch`, `max_paths_per_start`,
  `max_paths`), which bounds the enumeration on densely connected SDKs. Ties in path
  weight, which are common with exact type matches, are broken by proximity to the mean
  embedding of the cluster.
* **Argument constructors.** The allowed function set of an option contains its call
  path plus the type constructors of the path's parameters, which may appear inside
  argument lists only.
* **Certification thresholds.** Level 1 requires a pattern to occur in at least `ν`
  successful solutions from at least `ν` distinct tasks. Level 0 has one task per
  function, so it requires `ν` successful samples and `certification.level0_min_tasks`
  tasks.
* **Option extraction.** After certification, the successful solutions are rewritten as
  option-level trajectories (an option decision is inserted in front of the code each
  certified option accounts for, and the result is replayed through the option automata)
  and the policy is fitted to them for `certification.extraction_steps` steps. This
  gives newly certified options a calibrated initiation probability before the next
  phase starts.
* **Partial freeze.** After each phase the bottom share of transformer blocks given by
  `curriculum.freeze.bottom_fraction` continues at the option fine-tune learning rate
  (`mode: slow`) or stops being updated (`mode: hard`).
* **Schedule.** Warm-up and cosine decay restart at every phase
  (`optim.schedule_scope: phase`) or run once over all phases (`global`).
* **Objective normalisation.** The sequence term sums over the actions of a sample, as
  in the paper's equations. `grpo.length_normalize` divides by the sample length.
* **Task generation.** For every graph path a call skeleton is synthesised from type
  annotations. It supplies the setup code of the prompt, the minimal number of API calls
  and the reference length, and it is never shown to the policy. Task statements are
  assembled from the docstrings of the path's functions.
* **Task validation.** `option-grpo prepare` executes every skeleton. Tasks whose
  skeleton fails are dropped, properties the skeleton violates are removed, and the
  behaviour on each edge-case variant is recorded as the expectation of the edge-case
  test. `--no-validate` skips this step.
* **Properties must be exercised.** A function-level property (type preservation, shape,
  invariant) counts only when the program actually calls the function
  (`benchmark.require_calls`). Without this rule a program that returns one of its
  inputs would pass. `benchmark.reference_equivalence` additionally requires the output
  to equal the skeleton's output.
* **Held-out tasks.** `benchmark.test_fraction` (default 0.2) of the tasks of each level
  are held out. Training and certification use the training split, evaluation uses the
  held-out split. Set it to 0 and `eval.split=all` to train and evaluate on all tasks.
* **Option description length.** ODL is computed on the API-call abstraction of a
  program: the minimum number of option call paths that segment its chain-level SDK
  calls (dynamic programming, at most `benchmark.odl_max_options = 8`). It is infinite
  for programs that use third-party libraries or re-implement functionality by hand.
* **Option reuse (OR).** The share of a solution's SDK calls that fall inside a matched
  Level-1 option.
* **Asynchronous APIs.** Skeletons wrap coroutine functions in `asyncio.run`.

## Tests

```bash
pytest                 # everything, about half a minute on a laptop CPU
pytest -m "not slow"   # skip the two end-to-end training tests
```

The suite runs offline. It covers the static analysis, graph construction, option
discovery (clustering is checked against a brute-force implementation), the option
automaton and the compiled masks (checked against exhaustive simulation over the whole
vocabulary), program analysis and quality metrics, task generation and validation,
every execution stage of the sandbox, the reward, the objective and its gradients, the
rollout engine on GPT-2, Qwen and Llama style decoders, option extraction, and the full
three-phase pipeline on `tinyframe` with a tiny language model.

## Limitations

* The reward cannot verify full semantic correctness. Programs that run and type-check
  but compute the wrong thing receive partial credit.
* Option discovery relies on type annotations. Untyped or dynamically typed SDKs would
  need a type-inference pass first.
* Within an option, calls must appear in path order on the main call chain. A composition
  written as nested calls (`g(f(x))`) is produced with ordinary tokens instead.
* Training is per SDK and needs a sandbox able to execute the SDK.

## Citation

```bibtex
@inproceedings{kadam2027optiongrpo,
  title     = {Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs},
  author    = {Kadam, Prashank},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  note      = {To appear}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
