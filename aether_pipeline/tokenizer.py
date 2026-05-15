from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer


DEFAULT_AETHER_TOKENIZER = os.environ.get("AETHER_TOKENIZER_NAME", "deepseek-ai/DeepSeek-V3")


@dataclass
class AetherTokenizer:
    name_or_path: str = DEFAULT_AETHER_TOKENIZER
    cache_dir: Optional[str] = None
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        self._tok = AutoTokenizer.from_pretrained(
            self.name_or_path,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self.vocab_size = len(self._tok)
        # Use canonical DeepSeek-V3 special token IDs per spec; fall back to
        # whatever the loaded tokenizer reports if they differ.
        self.bos_token_id = self._tok.bos_token_id or 100000
        self.eos_token_id = self._tok.eos_token_id or 100001
        self.pad_token_id = self._tok.pad_token_id or self.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_ordinary(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def save_pretrained(self, output_dir: str) -> None:
        self._tok.save_pretrained(output_dir)


def load_aether_tokenizer(
    name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AetherTokenizer:
    return AetherTokenizer(
        name_or_path=name_or_path or DEFAULT_AETHER_TOKENIZER,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
