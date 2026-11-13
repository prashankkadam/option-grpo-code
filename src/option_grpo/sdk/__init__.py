"""SDK source analysis: type references, the static index and the SDK registry."""

from option_grpo.sdk.introspect import ClassInfo, FunctionInfo, ParamInfo, SDKAnalyzer, SDKIndex, analyze_sdk
from option_grpo.sdk.registry import SDKSpec, get_sdk, list_sdks, register_sdk
from option_grpo.sdk.types import ANY, CompatWeights, TypeRef, parse_annotation, type_compat

__all__ = [
    "ANY",
    "ClassInfo",
    "CompatWeights",
    "FunctionInfo",
    "ParamInfo",
    "SDKAnalyzer",
    "SDKIndex",
    "SDKSpec",
    "TypeRef",
    "analyze_sdk",
    "get_sdk",
    "list_sdks",
    "parse_annotation",
    "register_sdk",
    "type_compat",
]
