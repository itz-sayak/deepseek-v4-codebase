from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("deepseek_v4_pro_2b requires PyTorch to run") from exc


def hybrid_newton_schulz(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """DeepSeek-V4 hybrid Newton-Schulz orthogonalization.

    Uses 8 fast convergence iterations with (3.4445, -4.7750, 2.0315),
    followed by 2 stabilizing iterations with (2, -1.5, 0.5).
    """

    original_shape = x.shape
    if x.ndim < 2:
        return x
    x = x.reshape(x.shape[0], -1) if x.ndim > 2 else x
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    x = x / x.norm(p="fro").clamp_min(eps)
    for i in range(10):
        if i < 8:
            a, b, c = 3.4445, -4.7750, 2.0315
        else:
            a, b, c = 2.0, -1.5, 0.5
        xx_t = x @ x.t()
        x = a * x + b * (xx_t @ x) + c * ((xx_t @ xx_t) @ x)
    if transposed:
        x = x.t()
    return x.reshape(original_shape)


class Muon(torch.optim.Optimizer):
    """Muon optimizer matching Algorithm 1 in the DeepSeek-V4 paper.

    Use this for matrix-like parameters. Embeddings, LM heads, RMSNorm weights,
    and mHC static/gating parameters should usually remain on AdamW, as stated
    in the paper.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        update_rescale: float = 0.2,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            update_rescale=update_rescale,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            gamma = group["update_rescale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim < 2:
                    raise ValueError("Muon expects matrix-like parameters; use AdamW for vectors/scalars")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = hybrid_newton_schulz(momentum * buf + grad)
                update = update * (math.sqrt(max(update.shape[-2], update.shape[-1])) * gamma)
                p.mul_(1.0 - lr * weight_decay).add_(update, alpha=-lr)
        return loss


def split_muon_adamw_params(model: torch.nn.Module):
    """Return parameter groups that follow the paper's optimizer split."""

    muon, adamw = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_adamw = (
            param.ndim < 2
            or "embed_tokens" in name
            or "lm_head" in name
            or "norm" in name
            or ".s_pre" in name
            or ".s_res" in name
            or ".s_post" in name
            or ".alpha_" in name
        )
        (adamw if use_adamw else muon).append(param)
    return muon, adamw
