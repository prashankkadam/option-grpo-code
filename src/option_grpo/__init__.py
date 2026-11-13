"""Option-GRPO: hierarchical reinforcement learning for code generation on unseen SDKs.

The package is organised along the four stages of the method:

* :mod:`option_grpo.sdk`, :mod:`option_grpo.graphs` -- static analysis of SDK source
  code and construction of the API, parameter-flow and semantic graphs.
* :mod:`option_grpo.options` -- code options: discovery (Algorithm 1), certification
  (Algorithm 2), decode-time constraints and option description length.
* :mod:`option_grpo.tasks`, :mod:`option_grpo.proptest`, :mod:`option_grpo.execution`,
  :mod:`option_grpo.rewards` -- curriculum tasks, property-based tests (Algorithm 3),
  sandboxed execution and the layered execution reward.
* :mod:`option_grpo.policy`, :mod:`option_grpo.grpo` -- the hierarchical policy over
  the extended action space and curriculum-staged GRPO training (Algorithm 4).
* :mod:`option_grpo.eval` -- pass@k, code-quality metrics, baselines and analyses.
"""

__version__ = "0.1.0"
