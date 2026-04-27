from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SourceSpec:
    name: str
    stage: str
    hf_name: str
    split: str
    text_kind: str
    weight: float = 0.0
    config_name: Optional[str] = None
    data_dir: Optional[str] = None
    limit: Optional[int] = None
    notes: str = ""
    extra_kwargs: Dict[str, object] = field(default_factory=dict)

    def hf_kwargs(self) -> Dict[str, object]:
        out: Dict[str, object] = {"split": self.split}
        if self.config_name is not None:
            out["name"] = self.config_name
        if self.data_dir is not None:
            out["data_dir"] = self.data_dir
        out.update(self.extra_kwargs)
        return out


PRETRAIN_SOURCES: List[SourceSpec] = [
    SourceSpec("openwebtext", "pretrain", "Skylion007/openwebtext", "train", "text", weight=0.20, notes="OpenWebText"),
    SourceSpec("c4_en", "pretrain", "allenai/c4", "train", "text", weight=0.09, config_name="en", notes="C4 English"),
    SourceSpec("code_python", "pretrain", "bigcode/the-stack", "train", "the_stack", weight=0.16, data_dir="data/python"),
    SourceSpec("code_java", "pretrain", "bigcode/the-stack", "train", "the_stack", weight=0.05, data_dir="data/java"),
    SourceSpec("code_javascript", "pretrain", "bigcode/the-stack", "train", "the_stack", weight=0.04, data_dir="data/javascript"),
    SourceSpec("openwebmath", "pretrain", "open-web-math/open-web-math", "train", "text", weight=0.09, notes="OpenWebMath"),
    SourceSpec("metamathqa", "pretrain", "meta-math/MetaMathQA", "train", "metamathqa", weight=0.04),
    SourceSpec("fineweb_edu", "pretrain", "HuggingFaceFW/fineweb-edu", "train", "text", weight=0.07, config_name="sample-10BT"),
    SourceSpec("wikipedia", "pretrain", "wikimedia/wikipedia", "train", "wikipedia", weight=0.08, config_name="20231101.en"),
    SourceSpec("cc_news", "pretrain", "cc_news", "train", "text", weight=0.07),
    SourceSpec("code_search_net", "pretrain", "code_search_net", "train", "code_search_net", weight=0.03),
    SourceSpec("code_github", "pretrain", "bigcode/the-stack-smol", "train", "the_stack_smol", weight=0.02),
    SourceSpec("redpajama_books", "pretrain", "EleutherAI/the_pile_deduplicated", "train", "text", weight=0.06),
    SourceSpec("arxiv_math", "pretrain", "EleutherAI/the_pile_deduplicated", "train", "text", weight=0.05),
    SourceSpec("stackexchange", "pretrain", "HuggingFaceH4/ultrachat_200k", "train_sft", "ultrachat", weight=0.01),
]

NEW_PRETRAIN_SOURCES: List[SourceSpec] = [
    SourceSpec("flan", "new_pretrain", "Muennighoff/flan", "train", "flan"),
    SourceSpec("smoltalk", "new_pretrain", "HuggingFaceTB/smol-smoltalk", "train", "smoltalk"),
    SourceSpec("alpaca", "new_pretrain", "tatsu-lab/alpaca", "train", "alpaca"),
]

LARGE_PRETRAIN_SOURCES: List[SourceSpec] = [
    SourceSpec("fineweb_350bt", "large_pretrain", "HuggingFaceFW/fineweb", "train", "text", config_name="sample-350BT"),
    SourceSpec("falcon_refinedweb", "large_pretrain", "tiiuae/falcon-refinedweb", "train", "content"),
    SourceSpec("dclm_baseline", "large_pretrain", "mlfoundations/dclm-baseline-1.0", "train", "text"),
]

SFT_SOURCES: List[SourceSpec] = [
    SourceSpec("openhermes", "sft", "teknium/OpenHermes-2.5", "train", "sft_messages"),
    SourceSpec("sharegpt", "sft", "anon8231489123/ShareGPT_Vicuna_unfiltered", "train", "sharegpt", extra_kwargs={"data_files": "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"}),
    SourceSpec("orca_math", "sft", "microsoft/orca-math-word-problems-200k", "train", "orca_math"),
    SourceSpec("wizardlm_evol", "sft", "WizardLMTeam/WizardLM_evol_instruct_V2_196k", "train", "wizardlm"),
    SourceSpec("code_alpaca", "sft", "sahil2801/CodeAlpaca-20k", "train", "alpaca"),
    SourceSpec("dolly", "sft", "databricks/databricks-dolly-15k", "train", "dolly"),
]

DPO_SOURCES: List[SourceSpec] = [
    SourceSpec("ultrafeedback", "dpo", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "preference"),
    SourceSpec("orca_dpo", "dpo", "Intel/orca_dpo_pairs", "train", "preference"),
]

ALL_SOURCES: List[SourceSpec] = PRETRAIN_SOURCES + NEW_PRETRAIN_SOURCES + LARGE_PRETRAIN_SOURCES + SFT_SOURCES + DPO_SOURCES

SOURCE_INDEX = {spec.name: spec for spec in ALL_SOURCES}
