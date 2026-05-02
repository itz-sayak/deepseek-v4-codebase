"""Aether-2B style 2B reference implementation."""

from .configuration import Aether2BConfig

__all__ = ["Aether2BConfig", "Aether2BForCausalLM", "Aether2BModel", "Aether2BServingEngine", "Muon", "SpeculativeDecoder", "SpecDecodeSummary", "build_self_spec_draft_model"]


def __getattr__(name: str):
    if name in {"Aether2BForCausalLM", "Aether2BModel"}:
        from .modeling import Aether2BForCausalLM, Aether2BModel

        return {
            "Aether2BForCausalLM": Aether2BForCausalLM,
            "Aether2BModel": Aether2BModel,
        }[name]
    if name == "Aether2BServingEngine":
        from .serving import Aether2BServingEngine

        return Aether2BServingEngine
    if name == "Muon":
        from .muon import Muon

        return Muon
    if name in {"SpeculativeDecoder", "SpecDecodeSummary", "SpecDecodeResult", "build_self_spec_draft_model"}:
        from .speculative import SpeculativeDecoder, SpecDecodeSummary, SpecDecodeResult, build_self_spec_draft_model

        return {"SpeculativeDecoder": SpeculativeDecoder,
                "SpecDecodeSummary": SpecDecodeSummary,
                "SpecDecodeResult": SpecDecodeResult,
                "build_self_spec_draft_model": build_self_spec_draft_model}[name]
    raise AttributeError(name)
