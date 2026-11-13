"""Retrieval over the SDK for the retrieval-augmented baselines.

* ``rag`` retrieves source-code chunks (one chunk per public function or class);
* ``docprompting`` retrieves documentation passages (signature and docstring).

Chunks are embedded with a code encoder (CodeBERT by default, mean pooling) and the
``top_k`` most similar chunks to the task description are placed in front of the prompt
as a comment block.  The same module provides inference-time retrieval for a trained
Option-GRPO policy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from option_grpo.sdk.introspect import SDKIndex
from option_grpo.tasks.dataset import Task
from option_grpo.tasks.prompts import render_context
from option_grpo.utils import get_logger

log = get_logger(__name__)

Encoder = Callable[[list[str]], np.ndarray]


def transformer_encoder(model_name: str = "microsoft/codebert-base", device: str | None = None, batch_size: int = 32) -> Encoder:
    """Mean-pooled embeddings from a Hugging Face encoder.

    Inputs longer than the encoder's window are embedded from their leading window; the
    text placed in the prompt is always the complete chunk.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    limit = min(int(getattr(tokenizer, "model_max_length", 512) or 512), 512)

    @torch.no_grad()
    def encode(texts: list[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start : start + batch_size], padding=True, truncation=True, max_length=limit, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    return encode


def build_corpus(index: SDKIndex, mode: str) -> list[str]:
    """Chunks of the SDK: source code (``rag``) or documentation (``docprompting``)."""
    chunks: list[str] = []
    if mode == "docprompting":
        for fn in index.public_functions(include_properties=True):
            doc = (fn.doc or "").strip()
            chunks.append(f"{fn.signature()}\n{doc}".strip())
        for cls in index.classes.values():
            if cls.public_path and cls.doc:
                chunks.append(f"class {cls.public_path}\n{cls.doc.strip()}")
        return chunks
    if mode != "rag":
        raise ValueError(f"unknown retrieval mode: {mode}")
    for fn in index.public_functions(include_properties=True):
        if fn.kind == "constructor" and not index.functions.get(fn.id, fn).lineno:
            continue
        source = index.read_source(fn)
        if source.strip():
            chunks.append(f"# {fn.public_path}\n{source}")
        else:  # source unavailable (e.g. the index was built on another machine)
            chunks.append(f"{fn.signature()}\n{(fn.doc or '').strip()}".strip())
    return chunks


class Retriever:
    def __init__(self, index: SDKIndex, encoder: Encoder, mode: str = "rag", top_k: int = 10) -> None:
        self.mode = mode
        self.top_k = top_k
        self.encoder = encoder
        self.chunks = build_corpus(index, mode)
        self.embeddings = encoder(self.chunks) if self.chunks else np.zeros((0, 1), dtype=np.float32)
        self._cache: dict[str, str] = {}
        log.info("retriever (%s): %d chunks", mode, len(self.chunks))

    def retrieve(self, query: str) -> list[str]:
        if not self.chunks:
            return []
        vector = self.encoder([query])[0]
        with np.errstate(all="ignore"):
            scores = self.embeddings @ vector
        order = np.argsort(-scores)[: self.top_k]
        return [self.chunks[i] for i in order.tolist()]

    def context(self, task: Task) -> str:
        """Comment block with the passages retrieved for a task (cached per task)."""
        cached = self._cache.get(task.id)
        if cached is None:
            header = "Documentation retrieved from the SDK" if self.mode == "docprompting" else "Source code retrieved from the SDK"
            cached = render_context(self.retrieve(task.description), header)
            self._cache[task.id] = cached
        return cached
