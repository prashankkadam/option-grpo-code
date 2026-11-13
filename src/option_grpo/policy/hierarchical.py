"""The hierarchical policy over the extended action space ``A_k = O_{<k} ∪ V``.

The policy is a causal language model whose action space is the token vocabulary ``V``
extended with one action per option.  Selecting an option action is the *option
decision* ``mu_theta(o | ctx)``; the tokens generated while the option is active come
from the same network with out-of-set tokens masked (the intra-option policy ``pi_k``).

Option actions have their own input embeddings and output vectors, kept separate from
the language model's (possibly tied) embedding matrix.  This keeps the pretrained
vocabulary untouched, lets the option vocabulary grow between curriculum phases and
allows a dedicated learning rate for option parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from option_grpo.options.option import CodeOption
from option_grpo.utils import get_logger

log = get_logger(__name__)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class HierarchicalPolicy(nn.Module):
    """Causal LM with an option head.

    Extended action ids: ``[0, V)`` are tokens, ``V + i`` is option ``i``.
    """

    def __init__(self, lm: nn.Module, num_options: int, option_logit_bias: float = 0.0) -> None:
        super().__init__()
        self.lm = lm
        embeddings = self.core.get_input_embeddings()
        self.base_vocab_size: int = int(embeddings.num_embeddings)
        self.hidden_size: int = int(embeddings.embedding_dim)
        self.num_options = int(num_options)
        size = max(self.num_options, 1)
        self.option_in = nn.Embedding(size, self.hidden_size)
        self.option_out = nn.Linear(self.hidden_size, size, bias=True)
        with torch.no_grad():
            scale = float(embeddings.weight.detach().float().std())
            self.option_in.weight.normal_(mean=0.0, std=scale)
            self.option_out.weight.normal_(mean=0.0, std=scale)
            self.option_out.bias.fill_(option_logit_bias)
        reference = embeddings.weight
        self.option_in.to(device=reference.device, dtype=reference.dtype)
        self.option_out.to(device=reference.device, dtype=reference.dtype)
        self.initial_option_bias = float(option_logit_bias)

    # ------------------------------------------------------------------ structure
    @property
    def core(self) -> nn.Module:
        """The underlying Hugging Face model (adapters, if any, are injected in place)."""
        lm = self.lm
        return lm.get_base_model() if hasattr(lm, "get_base_model") else lm

    @property
    def decoder(self) -> nn.Module:
        core = self.core
        return getattr(core, core.base_model_prefix)

    @property
    def token_embeddings(self) -> nn.Module:
        return self.core.get_input_embeddings()

    @property
    def token_head(self) -> nn.Module:
        return self.core.get_output_embeddings()

    @property
    def device(self) -> torch.device:
        return self.option_in.weight.device

    def transformer_layers(self) -> list[nn.Module]:
        """The transformer blocks, bottom first (used by the partial freeze)."""
        decoder = self.decoder
        for name in ("layers", "h", "block", "blocks"):
            layers = getattr(decoder, name, None)
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return list(layers)
        best: list[nn.Module] = []
        for module in decoder.modules():
            if isinstance(module, nn.ModuleList) and len(module) > len(best):
                best = list(module)
        return best

    def option_parameters(self) -> list[nn.Parameter]:
        return [self.option_in.weight, self.option_out.weight, self.option_out.bias]

    # ------------------------------------------------------------------ pieces
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Input embeddings for extended action ids."""
        is_token = ids < self.base_vocab_size
        token_embeds = self.token_embeddings(ids.clamp(max=self.base_vocab_size - 1))
        # The option table takes part in every forward pass, also when the batch holds no
        # option action, so that distributed training never sees it as an unused parameter.
        option_ids = (ids - self.base_vocab_size).clamp(min=0)
        option_embeds = self.option_in(option_ids).to(token_embeds.dtype)
        return torch.where(is_token.unsqueeze(-1), token_embeds, option_embeds)

    def hidden(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        """Final hidden states (after the last normalisation layer) and the KV cache."""
        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        return out.last_hidden_state, getattr(out, "past_key_values", None)

    def heads(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Token logits ``[..., V]`` and option logits ``[..., |O|]`` in float32."""
        token_logits = self.token_head(hidden).float()
        option_logits = self.option_out(hidden.to(self.option_out.weight.dtype)).float()
        return token_logits, option_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced pass returning the logits at the selected positions only.

        ``rows`` / ``cols`` index the ``(sequence, position)`` pairs whose next-action
        distribution is needed; the vocabulary projection is skipped everywhere else.
        """
        hidden, _ = self.hidden(self.embed(input_ids), attention_mask=attention_mask)
        return self.heads(hidden[rows, cols])

    # ------------------------------------------------------------------ options
    @torch.no_grad()
    def init_options(self, options: list[CodeOption], tokenizer, only: set[int] | None = None) -> None:
        """Initialise option vectors from the tokens of the option's function names.

        The input embedding of an option is the mean input embedding of the tokens that
        spell its call path, and its output vector is the mean output embedding of the
        same tokens, so a freshly added option starts with a sensible prior.
        """
        token_in = self.token_embeddings.weight
        head = self.token_head
        token_out = head.weight if head is not None and hasattr(head, "weight") else token_in
        for option in options:
            if only is not None and option.index not in only:
                continue
            if option.index >= self.num_options:
                raise IndexError(f"option {option.index} exceeds the option head ({self.num_options})")
            ids = tokenizer(" " + " ".join(option.call_names), add_special_tokens=False)["input_ids"]
            ids = [i for i in ids if i < self.base_vocab_size]
            if not ids:
                continue
            index = torch.tensor(ids, device=token_in.device)
            self.option_in.weight[option.index] = token_in[index].float().mean(dim=0).to(self.option_in.weight.dtype)
            self.option_out.weight[option.index] = token_out[index].float().mean(dim=0).to(self.option_out.weight.dtype)
            self.option_out.bias[option.index] = self.initial_option_bias

    # ------------------------------------------------------------------ persistence
    def option_state(self) -> dict[str, torch.Tensor]:
        return {
            "option_in": self.option_in.weight.detach().float().cpu(),
            "option_out": self.option_out.weight.detach().float().cpu(),
            "option_bias": self.option_out.bias.detach().float().cpu(),
        }

    def load_option_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            n = min(state["option_in"].shape[0], self.option_in.weight.shape[0])
            self.option_in.weight[:n] = state["option_in"][:n].to(self.option_in.weight)
            self.option_out.weight[:n] = state["option_out"][:n].to(self.option_out.weight)
            self.option_out.bias[:n] = state["option_bias"][:n].to(self.option_out.bias)

    def save(self, directory: str | Path) -> None:
        """Save the language model (or its adapters) and the option head."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(directory / "lm")
        torch.save(self.option_state(), directory / "options.pt")

    def gradient_checkpointing(self, enable: bool = True) -> None:
        core = self.core
        if enable and hasattr(core, "gradient_checkpointing_enable"):
            core.gradient_checkpointing_enable()
        elif not enable and hasattr(core, "gradient_checkpointing_disable"):
            core.gradient_checkpointing_disable()


def load_language_model(cfg, checkpoint: str | Path | None = None):
    """Load tokenizer and causal LM according to a :class:`~option_grpo.config.ModelConfig`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = str(checkpoint) if checkpoint is not None and not cfg.lora.enabled else cfg.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
    import transformers

    # The keyword for the weight precision was renamed in transformers 5.
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = {dtype_key: DTYPES[cfg.dtype], "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if checkpoint is None or cfg.lora.enabled:
        kwargs["revision"] = cfg.revision
    lm = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    if cfg.lora.enabled:
        try:
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LoRA training needs peft: pip install 'option-grpo[lora]'") from exc
        if checkpoint is not None:
            lm = PeftModel.from_pretrained(lm, str(checkpoint), is_trainable=True)
        else:
            lm = get_peft_model(
                lm,
                LoraConfig(
                    r=cfg.lora.r,
                    lora_alpha=cfg.lora.alpha,
                    lora_dropout=cfg.lora.dropout,
                    target_modules=list(cfg.lora.target_modules),
                    task_type="CAUSAL_LM",
                ),
            )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, lm


def build_policy(cfg, num_options: int, option_logit_bias: float = 0.0, checkpoint: str | Path | None = None):
    """Tokenizer and :class:`HierarchicalPolicy` for a model configuration.

    ``checkpoint`` is a directory written by :meth:`HierarchicalPolicy.save`.
    """
    lm_dir = Path(checkpoint) / "lm" if checkpoint is not None else None
    tokenizer, lm = load_language_model(cfg, lm_dir)
    policy = HierarchicalPolicy(lm, num_options, option_logit_bias)
    if checkpoint is not None:
        state_path = Path(checkpoint) / "options.pt"
        if state_path.exists():
            policy.load_option_state(torch.load(state_path, map_location="cpu"))
    if cfg.gradient_checkpointing:
        policy.gradient_checkpointing(True)
    return tokenizer, policy
