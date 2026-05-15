from .manifest import DPO_SOURCES, LARGE_PRETRAIN_SOURCES, NEW_PRETRAIN_SOURCES, PRETRAIN_SOURCES, SFT_SOURCES
from .tokenizer import AetherTokenizer, load_aether_tokenizer

__all__ = [
    "PRETRAIN_SOURCES",
    "NEW_PRETRAIN_SOURCES",
    "LARGE_PRETRAIN_SOURCES",
    "SFT_SOURCES",
    "DPO_SOURCES",
    "AetherTokenizer",
    "load_aether_tokenizer",
    "CudaSparseAttentionBackend",
    "HybridCacheLayout",
    "LongContextServingManager",
    "OnDiskPrefixKVStore",
    "PrefixCacheMetadata",
    "PrefixReuseResult",
    "PytorchAttentionBackend",
    "RestorePlan",
    "SWACacheMode",
]


def __getattr__(name: str):
    if name in {
        "CudaSparseAttentionBackend",
        "HybridCacheLayout",
        "LongContextServingManager",
        "OnDiskPrefixKVStore",
        "PrefixCacheMetadata",
        "PrefixReuseResult",
        "PytorchAttentionBackend",
        "RestorePlan",
        "SWACacheMode",
    }:
        from .serving import (
            CudaSparseAttentionBackend,
            HybridCacheLayout,
            LongContextServingManager,
            OnDiskPrefixKVStore,
            PrefixCacheMetadata,
            PrefixReuseResult,
            PytorchAttentionBackend,
            RestorePlan,
            SWACacheMode,
        )

        return {
            "CudaSparseAttentionBackend": CudaSparseAttentionBackend,
            "HybridCacheLayout": HybridCacheLayout,
            "LongContextServingManager": LongContextServingManager,
            "OnDiskPrefixKVStore": OnDiskPrefixKVStore,
            "PrefixCacheMetadata": PrefixCacheMetadata,
            "PrefixReuseResult": PrefixReuseResult,
            "PytorchAttentionBackend": PytorchAttentionBackend,
            "RestorePlan": RestorePlan,
            "SWACacheMode": SWACacheMode,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
