import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_v4_pro_2b.scheduler import DecodeScheduler, GenerationRequest, group_by_shared_prefix


class StubState:
    def __init__(self, logits):
        self.last_logits = logits


class StubEngine:
    def __init__(self):
        self.prefill_calls = []
        self.step_calls = []

    def prefill_with_reuse(self, prompt_ids):
        self.prefill_calls.append(list(prompt_ids))
        # first generated token should come from these logits directly
        return StubState(torch.tensor([[0.0, 10.0, -1.0]], dtype=torch.float32))

    def step_token(self, token_id, state):
        self.step_calls.append(int(token_id))
        # any subsequent step will force token 2
        logits = torch.tensor([[-2.0, -3.0, 8.0]], dtype=torch.float32)
        return logits, StubState(logits)


def test_scheduler_uses_prefill_logits_for_first_token():
    engine = StubEngine()
    scheduler = DecodeScheduler(engine, max_batch_size=1, greedy=True)
    scheduler.submit(GenerationRequest("req-0", prompt_ids=[4, 5, 6], max_new_tokens=2))
    finished = scheduler.run_until_done()
    assert len(finished) == 1
    assert finished[0].output_ids == [1, 2]
    assert engine.prefill_calls == [[4, 5, 6]]
    assert engine.step_calls == [1]


def test_group_by_shared_prefix_hashes_correctly():
    req_a = GenerationRequest("a", prompt_ids=list(range(80)))
    req_b = GenerationRequest("b", prompt_ids=list(range(80)))
    req_c = GenerationRequest("c", prompt_ids=list(range(63)) + [999] + list(range(64, 80)))
    req_d = GenerationRequest("d", prompt_ids=[1, 2, 3])

    groups = group_by_shared_prefix([req_a, req_b, req_c, req_d], min_shared_tokens=64)
    hashed_groups = {k: [req.seq_id for req in v] for k, v in groups.items() if k}
    assert len(hashed_groups) == 2
    assert sorted(next(v for v in hashed_groups.values() if set(v) == {"a", "b"})) == ["a", "b"]
    assert groups[""][0].seq_id == "d"
