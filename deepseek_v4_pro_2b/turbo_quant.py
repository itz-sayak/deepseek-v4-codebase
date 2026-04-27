"""TurboQuant — PolarQuant stage for mHC compressed KV cache.

Stage 1 of TurboQuant (Google Research, ICLR 2026):
  Hadamard rotation (WHT with random Rademacher diagonal) followed by
  symmetric per-vector Lloyd-Max scalar quantization.

Stage 2 (QJL residual correction) is intentionally omitted until PolarQuant-only
correctness is validated on this architecture — community benchmarks show QJL can
HURT quality when softmax amplifies JL variance, especially at 4-bit.

Architecture notes
------------------
* Target tensors: ``state.compressed`` [B, T_blocks, D=128] and
  ``state.index_compressed`` [B, T_blocks, D_idx=64].  Both dimensions are
  exact powers of 2, so the Walsh-Hadamard transform (WHT) applies without
  padding.
* ``state.window`` (SWA, O(W)) is intentionally NOT quantized: it covers the
  most recent ~128 tokens that dominate quality at each decode step.
* The n_expand=16 expansion geometry means compressed vectors may carry
  directional structure that the WHT does not fully de-correlate.  Always
  run the 8-bit correctness ladder (needle-in-haystack) before dropping to
  4-bit.  Never go below 4-bit values without explicit per-architecture
  validation.
* Keys and values in this architecture are the same (shared-KV compression),
  so separate K/V sensitivity analysis from the paper does not directly apply;
  treat the single compressed vector as "value-like" (higher sensitivity class).

Encode: y = WHT(D * x)
Decode: x' = D * WHT(y')          -- because WHT^{-1} = WHT for the normalised form
              where D is a fixed random ±1 diagonal (Rademacher, seeded per dim).

Usage
-----
>>> pq = PolarQuant(bits=8)
>>> data, scale = pq.encode(compressed_bf16)  # [B, T, D] -> int8 + float16 scale
>>> approx = pq.decode(data, scale)            # -> bf16, shape [B, T, D]
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def _wht(x: torch.Tensor) -> torch.Tensor:
    """Normalised Walsh-Hadamard transform on the last dimension.

    The last dimension must be a positive power of 2.  The transform is
    self-inverse: ``_wht(_wht(x)) == x`` up to floating-point precision.
    Operates in float32 internally to avoid bf16 truncation during the
    log2(d) butterfly stages; callers must cast the result to the desired dtype.
    """
    d = x.shape[-1]
    if d == 0 or (d & (d - 1)) != 0:
        raise ValueError(f"WHT requires last dim to be a power of 2, got {d}")

    # IMPORTANT: use torch.cat (not in-place assignment) so float32 inputs are
    # never corrupted.  x.to(torch.float32) returns the SAME tensor object when
    # x is already float32; any in-place butterfly modification would mutate the
    # caller's data, making _wht(_wht(x)) != x.
    leading = x.shape[:-1]
    out = x.to(torch.float32)  # may alias x for float32 inputs — never modify in-place
    h = 1
    while h < d:
        out = out.view(*leading, d // (2 * h), 2 * h)
        a = out[..., :h]   # view — NOT cloned; used read-only below
        b = out[..., h:]   # view — NOT cloned; used read-only below
        # torch.cat always allocates a fresh tensor, breaking the alias.
        out = torch.cat([a + b, a - b], dim=-1).view(*leading, d)
        h *= 2

    return out * (d ** -0.5)


# ---------------------------------------------------------------------------
# PolarQuant class
# ---------------------------------------------------------------------------

class PolarQuant:
    """PolarQuant: randomised Hadamard rotation + symmetric scalar quantization.

    Parameters
    ----------
    bits:
        Quantization width.  8 → ``torch.int8`` data tensor.
        4 → ``torch.uint8`` data tensor with two 4-bit signed values packed
        per byte (lower nibble = even index, upper nibble = odd index).
    seed:
        RNG seed for the fixed Rademacher diagonal.  Must be the same value
        at encode and decode time.  Default 42.
    """

    def __init__(self, bits: int, seed: int = 42) -> None:
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        self._seed = seed
        # Cache Rademacher diagonals by (dimension, device).  Populated lazily on
        # first encode/decode call for that (d, device) pair so multi-GPU sharding
        # never causes repeated .to(device) copies across calls.
        self._diag_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _diag(self, d: int, device: torch.device) -> torch.Tensor:
        """Return the fixed ±1 Rademacher diagonal for dimension *d* on *device*."""
        key = (d, str(device))
        if key not in self._diag_cache:
            gen = torch.Generator()
            # Unique seed per dimension so D=64 and D=128 get different diagonals.
            gen.manual_seed(self._seed * 10007 + d)
            r = torch.rand(d, generator=gen)
            diag = torch.where(r > 0.5, torch.ones(d), -torch.ones(d)).to(device)
            self._diag_cache[key] = diag
        return self._diag_cache[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a bf16/float32 tensor to *bits*-bit integers + per-vector scale.

        Parameters
        ----------
        x : Tensor [..., D]
            Input tensor, typically ``state.compressed`` with shape
            ``[B, T_blocks, D]``.  D must be a power of 2.

        Returns
        -------
        data : Tensor [..., D] int8   (bits=8)
               Tensor [..., D//2] uint8  (bits=4, packed)
        scale : Tensor [..., 1] float16  — per-vector symmetric scale
        """
        d = x.shape[-1]
        device = x.device
        diag = self._diag(d, device)

        # Rotate: y = WHT(diag * x).  Uses float32 inside _wht.
        # diag is broadcast over all leading dims.
        rotated = _wht(x.float() * diag)  # [..., D] float32

        if self.bits == 8:
            max_abs = rotated.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
            scale = (max_abs / 127.0).to(torch.float16)
            data = (rotated / max_abs * 127.0).round_().clamp_(-128, 127).to(torch.int8)
            return data, scale

        # bits == 4: symmetric, range [-7, 7] (avoids -8 for a clean zero-point)
        max_abs = rotated.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
        scale = (max_abs / 7.0).to(torch.float16)
        q = (rotated / max_abs * 7.0).round_().clamp_(-7, 7).to(torch.int32)

        # Pack pairs: even index → lower nibble, odd index → upper nibble.
        # We use int32 arithmetic to avoid sign issues from int8 bit-masking.
        q_even = q[..., 0::2] & 0x0F          # lower 4 bits of even elements
        q_odd = (q[..., 1::2] & 0x0F) << 4    # lower 4 bits of odd elements, shifted up
        data = (q_even | q_odd).to(torch.uint8)  # [..., D//2]
        return data, scale

    def decode(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize *data* + *scale* back to the original floating-point space.

        Parameters
        ----------
        data : int8 tensor [..., D] or uint8 tensor [..., D//2] for 4-bit
        scale : float16 tensor [..., 1]
        out_dtype : output dtype, default bfloat16

        Returns
        -------
        Tensor [..., D] in *out_dtype*
        """
        device = data.device
        scale_f32 = scale.to(torch.float32)

        if self.bits == 8:
            rotated = data.to(torch.float32) * scale_f32  # [..., D]
        else:
            # Unpack: lower nibble → even positions, upper nibble → odd positions.
            packed = data.to(torch.int32)
            even = packed & 0x0F                    # unsigned 4-bit (0..15)
            odd = (packed >> 4) & 0x0F              # unsigned 4-bit (0..15)
            # Sign-extend from 4-bit: values ≥ 8 represent negative numbers.
            even = torch.where(even >= 8, even - 16, even)
            odd = torch.where(odd >= 8, odd - 16, odd)
            d_full = data.shape[-1] * 2
            q_f32 = torch.empty(*data.shape[:-1], d_full, dtype=torch.float32, device=device)
            q_f32[..., 0::2] = even.float()
            q_f32[..., 1::2] = odd.float()
            rotated = q_f32 * scale_f32  # [..., D]

        d = rotated.shape[-1]
        diag = self._diag(d, device)

        # Inverse: x' = diag * WHT(rotated)   [because WHT(WHT(v)) = v]
        x = _wht(rotated) * diag  # float32

        return x.to(out_dtype)
