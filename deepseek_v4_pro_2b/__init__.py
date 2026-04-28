"""DeepSeek-V4-Pro style 2B reference implementation."""

from .configuration import DeepSeekV4Pro2BConfig

__all__ = ["DeepSeekV4Pro2BConfig", "DeepSeekV4Pro2BForCausalLM", "DeepSeekV4Pro2BModel", "DeepSeekV4Pro2BServingEngine", "Muon", "SpeculativeDecoder", "SpecDecodeSummary", "build_self_spec_draft_model"]


def __getattr__(name: str):
    if name in {"DeepSeekV4Pro2BForCausalLM", "DeepSeekV4Pro2BModel"}:
        from .modeling import DeepSeekV4Pro2BForCausalLM, DeepSeekV4Pro2BModel

        return {
            "DeepSeekV4Pro2BForCausalLM": DeepSeekV4Pro2BForCausalLM,
            "DeepSeekV4Pro2BModel": DeepSeekV4Pro2BModel,
        }[name]
    if name == "DeepSeekV4Pro2BServingEngine":
        from .serving import DeepSeekV4Pro2BServingEngine

        return DeepSeekV4Pro2BServingEngine
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
