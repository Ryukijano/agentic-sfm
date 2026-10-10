"""Phase 2 r5 OOM-cap contract: yaml + slurm + CPU KL load path.

These are the knobs that job 7975335 did not have. Keep them from drifting
back to group_size 6 / 50 images / 32k context / GPU KL.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_grpo import GRPOTrainer  # noqa: E402
from tests.test_scene_grpo_trainer import _make_trainer  # noqa: E402


class TestPhase2R5Config:
    def test_yaml_activation_caps(self):
        cfg = yaml.safe_load((ROOT / "configs" / "phase2_scene.yaml").read_text())
        assert cfg["rl"]["group_size"] == 4
        assert cfg["rollout"]["batch_size"] == 1
        assert cfg["data"]["scannet_root"] in (None, "null")
        assert cfg["training"]["kl_device"] == "cpu"
        stages = cfg["curriculum"]["stages"]
        assert stages, "curriculum stages missing"
        assert all(int(s["max_images"]) <= 20 for s in stages)
        kl = cfg["model"]["kl_adapter"]
        sft = cfg["model"]["sft_adapter"]
        assert kl.endswith("scene_sft_graph_retry2/checkpoints/epoch_5")
        assert sft == kl

    def test_slurm_vllm_context_matches_prompt_cap(self):
        text = (ROOT / "jobs" / "phase2_scene.slurm").read_text()
        assert "--max-model-len 8192" in text
        assert '"image":4' in text
        assert "--max-model-len 32768" not in text
        assert '"image":8' not in text


class TestKlCpuSeparateCopy:
    def test_module_device_reads_param_device(self):
        m = torch.nn.Linear(3, 3)
        assert GRPOTrainer._module_device(m, "cuda:0").type == "cpu"
        empty = torch.nn.Module()
        assert str(GRPOTrainer._module_device(empty, "cpu")) == "cpu"

    def test_cpu_kl_does_not_wrap_trainable_base(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.kl_device = "cpu"
        trainer.kl_adapter = str(tmp_path / "kl")
        (tmp_path / "kl").mkdir()
        trainer._model = torch.nn.Linear(2, 2)
        dummy_base = torch.nn.Linear(4, 4)
        captured: dict = {}

        def fake_load(name, **kwargs):
            captured["device_map"] = kwargs.get("device_map")
            return None, dummy_base

        class FakePeft:
            @staticmethod
            def from_pretrained(base, path, is_trainable=False):
                captured["peft_base_id"] = id(base)
                captured["trainable_id"] = id(trainer._model)
                captured["is_trainable"] = is_trainable
                return base

        with patch(
            "agentic_sfm.agent.policy.load_policy_processor_and_model",
            fake_load,
        ), patch("peft.PeftModel", FakePeft):
            ref = trainer._load_kl_reference()

        assert captured["device_map"] == "cpu"
        assert captured["peft_base_id"] == id(dummy_base)
        assert captured["peft_base_id"] != captured["trainable_id"]
        assert captured["is_trainable"] is False
        assert ref is dummy_base
        assert all(not p.requires_grad for p in ref.parameters())
