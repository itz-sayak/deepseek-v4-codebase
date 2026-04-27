import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_pipeline.serving import CudaSparseAttentionBackend, PytorchAttentionBackend
from deepseek_v4_pro_2b.configuration import DeepSeekV4Pro2BConfig
from deepseek_v4_pro_2b.modeling import DeepSeekV4Pro2BForCausalLM
from deepseek_v4_pro_2b.serving import DeepSeekV4Pro2BServingEngine


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA serving test requires a CUDA-capable PyTorch environment")


def tiny_config():
    return DeepSeekV4Pro2BConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        attention_head_dim=16,
        query_compression_dim=32,
        indexer_num_heads=2,
        indexer_head_dim=8,
        csa_compression=2,
        hca_compression=4,
        csa_top_k=3,
        sliding_window=4,
        rope_dim=8,
        output_groups=2,
        group_output_dim=32,
        num_routed_experts=4,
        num_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        hash_routed_layers=1,
        mhc_expansion=2,
        mtp_depth=1,
    )


def test_cuda_backend_matches_pytorch_serving_engine():
    torch.manual_seed(0)
    model = DeepSeekV4Pro2BForCausalLM(tiny_config()).eval().to("cuda")
    cuda_engine = DeepSeekV4Pro2BServingEngine(model, backend=CudaSparseAttentionBackend(), device="cuda")
    ref_engine = DeepSeekV4Pro2BServingEngine(model, backend=PytorchAttentionBackend(), device="cuda")
    tokens = [1, 7, 3, 9, 2, 11, 5, 4, 8]

    ref_state = ref_engine.prefill(tokens)
    cuda_state = cuda_engine.prefill(tokens)

    torch.testing.assert_close(cuda_state.last_logits.float(), ref_state.last_logits.float(), atol=5e-3, rtol=5e-3)
