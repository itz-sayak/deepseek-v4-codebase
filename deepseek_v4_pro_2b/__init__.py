"""DeepSeek-V4-Pro style 2B reference implementation."""

from .configuration import DeepSeekV4Pro2BConfig

__all__ = ["DeepSeekV4Pro2BConfig", "DeepSeekV4Pro2BForCausalLM", "DeepSeekV4Pro2BModel", "DeepSeekV4Pro2BServingEngine", "Muon", "SpeculativeDecoder", "SpecDecodeSummary"]


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
    if name in {"SpeculativeDecoder", "SpecDecodeSummary", "SpecDecodeResult"}:
        from .speculative import SpeculativeDecoder, SpecDecodeSummary, SpecDecodeResult

        return {"SpeculativeDecoder": SpeculativeDecoder,
                "SpecDecodeSummary": SpecDecodeSummary,
                "SpecDecodeResult": SpecDecodeResult}[name]
    raise AttributeError(name)
