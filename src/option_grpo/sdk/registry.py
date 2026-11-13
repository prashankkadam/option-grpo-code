"""Registry of the SDKs of UnseenSDK-Bench.

Each entry records how to locate the SDK's source (import name / distribution) and the
domain category used when reporting results.  Additional SDKs can be registered at
runtime with :func:`register_sdk` or loaded from a YAML file with :func:`load_registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SDKSpec:
    name: str  # key used on the command line and in artefact paths
    import_name: str  # top-level Python package
    dist: str  # distribution name on the package index
    category: str = "General"
    display_name: str | None = None
    version: str | None = None  # optional pin, e.g. "0.4.2"
    source_root: str | None = None  # explicit path to the package directory
    prelude: str | None = None  # import line(s) shown in prompts; defaults to ``import <import_name>``
    extra_dists: list[str] = field(default_factory=list)  # extra requirements for the sandbox image

    @property
    def title(self) -> str:
        return self.display_name or self.name

    @property
    def requirement(self) -> str:
        return f"{self.dist}=={self.version}" if self.version else self.dist

    @property
    def import_prelude(self) -> str:
        return self.prelude or f"import {self.import_name}"


_BUILTIN: list[SDKSpec] = [
    SDKSpec("daft", "daft", "daft", "DataFrame", "Daft"),
    SDKSpec("lancedb", "lancedb", "lancedb", "Vector DB", "LanceDB"),
    SDKSpec("cuallee", "cuallee", "cuallee", "Data Valid.", "cuallee"),
    SDKSpec("docling", "docling", "docling", "Doc Proc.", "Docling"),
    SDKSpec("marker", "marker", "marker-pdf", "Doc Proc.", "Marker"),
    SDKSpec("surya", "surya", "surya-ocr", "Comp. Vision", "Surya"),
    SDKSpec("faststream", "faststream", "faststream", "Messaging", "FastStream"),
    SDKSpec("tracecat", "tracecat", "tracecat", "Security", "Tracecat"),
    SDKSpec("mcp", "mcp", "mcp", "Protocol", "MCP Python"),
    SDKSpec("vortex", "vortex", "vortex-data", "Scientific", "Vortex"),
    SDKSpec("libdebug", "libdebug", "libdebug", "Debugging", "libdebug"),
    SDKSpec("reflexio", "reflexio", "reflexio", "Agent Frmwk", "reflexio"),
]

_REGISTRY: dict[str, SDKSpec] = {spec.name: spec for spec in _BUILTIN}

# The twelve SDKs of the benchmark, in the order of the paper's tables.
BENCHMARK_SDKS: tuple[str, ...] = tuple(spec.name for spec in _BUILTIN)


def register_sdk(spec: SDKSpec, overwrite: bool = True) -> None:
    if not overwrite and spec.name in _REGISTRY:
        raise KeyError(f"SDK {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def get_sdk(name: str) -> SDKSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown SDK {name!r}; known SDKs: {known}") from None


def list_sdks() -> list[SDKSpec]:
    return list(_REGISTRY.values())


def load_registry(path: str | Path) -> list[SDKSpec]:
    """Register every SDK described in a YAML file (a mapping ``name -> fields``)."""
    with Path(path).open("r", encoding="utf-8") as f:
        data: dict[str, dict[str, Any]] = yaml.safe_load(f) or {}
    loaded: list[SDKSpec] = []
    base = Path(path).resolve().parent
    for name, fields_ in data.items():
        fields_ = dict(fields_ or {})
        root = fields_.get("source_root")
        if root and not Path(root).is_absolute():
            fields_["source_root"] = str((base / root).resolve())
        fields_.setdefault("import_name", name)
        fields_.setdefault("dist", fields_["import_name"])
        spec = SDKSpec(name=name, **fields_)
        register_sdk(spec)
        loaded.append(spec)
    return loaded
