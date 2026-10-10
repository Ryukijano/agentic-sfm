"""Regression: VLLMRolloutAgent must store first-call gate fields.

Job 7924692 died on the first pair episode because run_episode read
``self.first_call_validity_weight`` but __init__ never assigned it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_grpo import VLLMRolloutAgent  # noqa: E402


def test_rollout_agent_stores_first_call_gate_defaults():
    agent = VLLMRolloutAgent(vllm_url="http://127.0.0.1:0", model_name="dummy")
    assert agent.first_call_validity_weight == 0.5
    assert agent.first_call_invalid_penalty == 1.0
    assert agent.gate_geometry_on_validity is True


def test_rollout_agent_stores_first_call_gate_overrides():
    agent = VLLMRolloutAgent(
        vllm_url="http://127.0.0.1:0",
        model_name="dummy",
        first_call_validity_weight=0.25,
        first_call_invalid_penalty=2.0,
        gate_geometry_on_validity=False,
    )
    assert agent.first_call_validity_weight == 0.25
    assert agent.first_call_invalid_penalty == 2.0
    assert agent.gate_geometry_on_validity is False
