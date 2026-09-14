"""Unit tests for SceneGRPOTrainer — the scene-level GRPO loop in Phase 2.

No GPU / servers needed:
  - rollout collection runs the real ``run_scene_episode`` against the
    deterministic ``ScriptedSceneAgent`` + ``MockToolClient`` (retrieval
    embeddings are monkeypatched to deterministic hashes),
  - the LoRA policy-gradient step is exercised end-to-end with a tiny
    ``torch.nn.Linear`` stand-in and a stubbed ``compute_logprobs`` whose
    tensors are graph-connected to the module's parameters,
  - checkpoint/resume round-trips through a fake PEFT model that writes a
    real ``adapter_config.json``.
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import ToolCall  # noqa: E402
from agentic_sfm.rl.scene_episode import SceneRolloutEpisode  # noqa: E402
from scripts.run_scene_grpo import SceneGRPOTrainer  # noqa: E402
from scripts.scene_smoke_test import MockToolClient, ScriptedSceneAgent  # noqa: E402


def _config(tmp_path, group_size=4, reward_schedule="static", sgrpo_cgi=True):
    return {
        "model": {
            "name": "test-policy",
            "max_new_tokens": 64,
            "lora": {"rank": 4, "alpha": 8, "dropout": 0.0},
        },
        "rl": {
            "group_size": group_size,
            "max_tool_calls": 12,
            "max_turns": 15,
            "temperature": 1.0,
            "top_p": 0.95,
            "clip_low": 0.2,
            "clip_high": 0.28,
            "dynamic_sampling": True,
            "sgrpo_cgi": sgrpo_cgi,
            "reward_schedule": reward_schedule,
            "reward_warmup_steps": 20,
        },
        "training": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "total_epochs": 2,
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1.0,
            "save_freq": 1,
            "eval_freq": 1,
            "log_freq": 1,
            "eval_num_scenes": 3,
        },
        "reward": {"pose_weight": 0.9, "registration_weight": 0.5},
        "data": {
            # Point at paths that don't exist so no pair dataset is touched.
            "train_pairs": str(tmp_path / "no_train.json"),
            "val_pairs": str(tmp_path / "no_val.json"),
        },
        "output": {"wandb_enabled": False},
    }


def _make_trainer(tmp_path, **kw) -> SceneGRPOTrainer:
    return SceneGRPOTrainer(
        config=_config(tmp_path, **kw),
        tool_client=MockToolClient(),
        output_dir=str(tmp_path / "out"),
    )


def _episode(scene_id, reward, **kw):
    ep = SceneRolloutEpisode(
        scene_id=scene_id,
        image_paths=["a.jpg", "b.jpg"],
        num_images=2,
    )
    ep.reward = float(reward)
    ep.reward_components = {
        "total_reward": ep.reward,
        "registration_reward": ep.reward,
    }
    ep.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": '{"tool": "done", "args": {}}'},
    ]
    ep.assistant_responses = ['{"tool": "done", "args": {}}']
    ep.tool_calls = [ToolCall(tool="done", args={})]
    ep.done = True
    ep.recon_result = kw.get("recon_result", {"num_registered": 2})
    return ep


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Deterministic hash embeddings — keeps retrieval real but offline."""
    import agentic_sfm.rl.retrieval as retrieval

    def fake(image_path):
        seed = zlib.crc32(str(image_path).encode())
        return np.random.default_rng(seed).random(64).astype(np.float32)

    monkeypatch.setattr(retrieval, "compute_image_embedding", fake)


# ---------------------------------------------------------------------------
# Grouping / advantages
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_key_uses_scene_id(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        ep = _episode("scene_0042", 1.0)
        assert trainer._group_key(ep) == "scene_0042"

    def test_advantages_are_group_relative_by_scene(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        eps = [_episode("A", r) for r in (1.0, 2.0, 3.0)]
        eps += [_episode("B", r) for r in (0.0, 4.0)]
        adv = trainer.compute_advantages(eps)

        std_a = np.std([1.0, 2.0, 3.0]) + 1e-8
        std_b = np.std([0.0, 4.0]) + 1e-8
        np.testing.assert_allclose(
            adv[:3], [(r - 2.0) / std_a for r in (1.0, 2.0, 3.0)], rtol=1e-5)
        np.testing.assert_allclose(
            adv[3:], [(r - 2.0) / std_b for r in (0.0, 4.0)], rtol=1e-5)
        # below-mean rollouts get negative advantage, above-mean positive
        assert adv[0] < 0 < adv[2]
        assert adv[3] < 0 < adv[4]

    def test_zero_variance_filter_groups_by_scene(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        eps = [_episode("flat", 0.5) for _ in range(4)]
        eps += [_episode("varied", r) for r in (0.0, 1.0)]
        out = trainer._filter_zero_variance_groups(eps)
        assert {e.scene_id for e in out} == {"varied"}
        assert len(out) == 2

    def test_dynamic_sampling_disabled_keeps_flat_groups(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.dynamic_sampling = False
        eps = [_episode("flat", 0.5) for _ in range(3)]
        assert len(trainer._filter_zero_variance_groups(eps)) == 3


# ---------------------------------------------------------------------------
# Rollout collection (real run_scene_episode + scripted agent + mock tools)
# ---------------------------------------------------------------------------


class TestCollectSceneRollouts:
    def _scene(self, scene_id="s1", n_images=5):
        return {
            "scene_id": scene_id,
            "image_paths": [f"{scene_id}/img_{i:04d}.jpg" for i in range(n_images)],
            "num_images": n_images,
            "overlap_matrix": np.eye(n_images),
            "image_indices": list(range(n_images)),
            "gt_recon": {
                "num_images": n_images,
                "poses": {str(i): np.eye(4).tolist() for i in range(n_images)},
            },
        }

    def test_collects_group_size_episodes_per_scene(self, tmp_path):
        trainer = _make_trainer(tmp_path, group_size=3)
        agent = ScriptedSceneAgent(num_matches=2)
        scenes = [self._scene("s1"), self._scene("s2")]
        eps = trainer.collect_scene_rollouts(scenes, agent)
        assert len(eps) == 6
        assert sum(1 for e in eps if e.scene_id == "s1") == 3
        assert all(isinstance(e, SceneRolloutEpisode) for e in eps)
        assert all(e.messages and e.assistant_responses for e in eps)
        # scripted pipeline completes -> positive reward, no CGI needed
        assert all(e.reward > 0 for e in eps)
        assert trainer._last_collected_episodes is eps

    def test_collect_via_inherited_collect_rollouts(self, tmp_path):
        """train_step calls self.collect_rollouts — must hit the scene path."""
        trainer = _make_trainer(tmp_path, group_size=2)
        agent = ScriptedSceneAgent(num_matches=1)
        eps = trainer.collect_rollouts([self._scene("sx")], agent)
        assert len(eps) == 2 and eps[0].scene_id == "sx"

    def test_reward_config_pushed_to_agent(self, tmp_path):
        trainer = _make_trainer(tmp_path, group_size=1)
        agent = ScriptedSceneAgent(num_matches=1)
        trainer.collect_scene_rollouts([self._scene()], agent)
        assert agent.reward_config["pose_weight"] == pytest.approx(0.9)
        assert agent.reward_config["registration_weight"] == pytest.approx(0.5)

    def test_cgi_injects_oracle_when_all_rollouts_fail(self, tmp_path, monkeypatch):
        trainer = _make_trainer(tmp_path, group_size=2, sgrpo_cgi=True)

        def fake_episode(agent, scene_id, **kw):
            return _episode(scene_id, 0.0)

        oracle_calls = []

        def fake_oracle(agent, scene_id, failed_group_max_reward=0.0, **kw):
            oracle_calls.append(scene_id)
            ep = _episode(scene_id, failed_group_max_reward + 0.5)
            return ep

        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_episode", fake_episode)
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_oracle_episode", fake_oracle)

        eps = trainer.collect_scene_rollouts(
            [self._scene("dead")], ScriptedSceneAgent())
        assert oracle_calls == ["dead"]
        assert len(eps) == 2
        # oracle replaced the worst rollout in the group
        assert sorted(e.reward for e in eps) == [0.0, 0.5]

    def test_no_cgi_when_disabled(self, tmp_path, monkeypatch):
        trainer = _make_trainer(tmp_path, group_size=2, sgrpo_cgi=False)
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_episode",
            lambda agent, scene_id, **kw: _episode(scene_id, 0.0),
        )
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_oracle_episode",
            lambda *a, **kw: pytest.fail("oracle must not run"),
        )
        eps = trainer.collect_scene_rollouts(
            [self._scene("dead")], ScriptedSceneAgent())
        assert all(e.reward == 0.0 for e in eps)


# ---------------------------------------------------------------------------
# Reward schedule
# ---------------------------------------------------------------------------


class TestRewardSchedule:
    def test_dynamic_schedule_scales_pose_weight_during_warmup(self, tmp_path):
        trainer = _make_trainer(tmp_path, reward_schedule="dynamic")
        trainer._global_step = 0  # < reward_warmup_steps (20)
        cfg = trainer._scheduled_reward_config()
        assert cfg["pose_weight"] == pytest.approx(0.9 / 3.0)
        # base config untouched
        assert trainer._base_reward_config["pose_weight"] == pytest.approx(0.9)

        trainer._global_step = 25  # past warmup
        cfg = trainer._scheduled_reward_config()
        assert cfg["pose_weight"] == pytest.approx(0.9)

    def test_static_schedule_passes_config_through(self, tmp_path):
        trainer = _make_trainer(tmp_path, reward_schedule="static")
        trainer._global_step = 0
        assert trainer._scheduled_reward_config()["pose_weight"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Policy-gradient step (stubbed model + logprobs)
# ---------------------------------------------------------------------------


class TestTrainStep:
    def _stub_model(self, trainer):
        """Tiny stand-in for the PEFT model + AdamW on its params."""
        model = torch.nn.Linear(4, 4)
        trainer._model = model
        trainer._optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        trainer._load_training_model = lambda: None
        trainer.training_device = "cpu"  # test env has no cuda:1
        return model

    def _stub_logprobs(self, trainer, model):
        """compute_logprobs whose 'new' tensors depend on model params.

        The model-weight coefficient varies per episode so the per-episode
        clipped-surrogate losses (and their gradients) don't cancel to zero
        — group-relative advantages always sum to zero within a group.
        """
        old = torch.tensor([-1.0, -0.8, -0.6, -0.4])

        def fake(eps, requires_grad=False):
            out = []
            for i, _ in enumerate(eps):
                mask = torch.ones(4, dtype=torch.bool)
                if requires_grad:
                    lp = old + model.weight.view(-1)[0] * (0.05 + 0.02 * i)
                else:
                    lp = old.clone()
                out.append((lp, mask))
            return out

        trainer.compute_logprobs = fake

    def test_train_step_does_policy_gradient_update(self, tmp_path, monkeypatch):
        trainer = _make_trainer(tmp_path, group_size=3)
        model = self._stub_model(trainer)
        self._stub_logprobs(trainer, model)
        before = model.weight.detach().clone()

        rewards = iter([0.0, 0.5, 1.0] * 4)
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_episode",
            lambda agent, scene_id, **kw: _episode(scene_id, next(rewards)),
        )

        stats = trainer.train_step(
            [{"scene_id": "s1", "image_paths": []}], ScriptedSceneAgent(),
            accum_step=1, is_last_accum=True,
        )
        assert stats["num_episodes"] == 3
        assert stats["mean_reward"] == pytest.approx(0.5)
        assert stats["loss"] != 0.0
        assert not torch.equal(before, model.weight.detach())
        # scene metrics merged into stats
        assert "scene/mean_registered" in stats
        assert "scene/registration_reward" in stats

    def test_train_step_accumulates_without_step(self, tmp_path, monkeypatch):
        trainer = _make_trainer(tmp_path, group_size=3)
        model = self._stub_model(trainer)
        self._stub_logprobs(trainer, model)
        before = model.weight.detach().clone()

        rewards = iter([0.0, 0.5, 1.0] * 4)
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_episode",
            lambda agent, scene_id, **kw: _episode(scene_id, next(rewards)),
        )
        stats = trainer.train_step(
            [{"scene_id": "s1", "image_paths": []}], ScriptedSceneAgent(),
            accum_step=1, is_last_accum=False,
        )
        # grads produced but no optimizer step yet
        assert torch.equal(before, model.weight.detach())
        assert stats["loss"] != 0.0

    def test_train_step_skips_zero_variance_batch(self, tmp_path, monkeypatch):
        trainer = _make_trainer(tmp_path, group_size=2)
        model = self._stub_model(trainer)
        self._stub_logprobs(trainer, model)
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_episode",
            lambda agent, scene_id, **kw: _episode(scene_id, 1.0),
        )
        stats = trainer.train_step(
            [{"scene_id": "s1", "image_paths": []},
             {"scene_id": "s2", "image_paths": []}],
            ScriptedSceneAgent())
        assert stats["num_episodes"] == 0
        assert stats["loss"] == 0.0


# ---------------------------------------------------------------------------
# Scene metrics
# ---------------------------------------------------------------------------


class TestSceneMetrics:
    def test_metrics_cover_rewards_recon_and_doppelgangers(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        eps = [
            _episode("A", 1.0, recon_result={
                "num_registered": 8,
                "mean_pose_error_deg": 4.0,
                "num_doppelgangers_present": 2,
                "num_doppelgangers_filtered": 2,
            }),
            _episode("A", 0.0, recon_result={
                "num_registered": 4,
                "mean_pose_error_deg": 8.0,
                "num_doppelgangers_present": 0,
                "num_doppelgangers_filtered": 0,
            }),
        ]
        eps[0].doppelganger_checks = {"a__b": {}, "c__d": {}}
        m = trainer._scene_metrics(eps)
        assert m["scene/mean_registered"] == pytest.approx(6.0)
        assert m["scene/mean_registered_frac"] == pytest.approx(3.0)  # 6/2 imgs
        assert m["scene/mean_pose_error_deg"] == pytest.approx(6.0)
        assert m["scene/mean_doppelganger_checks"] == pytest.approx(1.0)
        assert m["scene/mean_doppelgangers_present"] == pytest.approx(1.0)
        assert m["scene/mean_doppelgangers_filtered"] == pytest.approx(1.0)
        assert m["scene/done_rate"] == pytest.approx(1.0)
        for key in (
            "scene/registration_reward", "scene/pose_reward",
            "scene/doppelganger_reward", "scene/tool_cost",
        ):
            assert key in m

    def test_empty_episode_list(self, tmp_path):
        assert _make_trainer(tmp_path)._scene_metrics([]) == {}


# ---------------------------------------------------------------------------
# Checkpointing / resume
# ---------------------------------------------------------------------------


class _FakePeftModel(torch.nn.Linear):
    """Minimal PEFT stand-in: real parameters + save_pretrained layout."""

    def __init__(self):
        super().__init__(2, 2)

    def save_pretrained(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text(json.dumps({"r": 4}))
        torch.save(self.state_dict(), p / "adapter_model.bin")


class TestCheckpointResume:
    def test_save_writes_adapter_and_trainer_state(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer._model = _FakePeftModel()
        trainer.current_epoch = 3
        trainer._global_step = 7

        trainer.save_lora_checkpoint("latest")  # rollout_agent=None -> no vLLM
        ckpt = trainer.ckpt_dir / "latest"
        assert (ckpt / "adapter_config.json").exists()
        state = json.loads((ckpt / "trainer_state.json").read_text())
        assert state["tag"] == "latest"
        # next epoch to resume from = current_epoch + 1
        assert state["epoch"] == 4
        assert state["global_step"] == 7

    def test_save_epoch_tag(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer._model = _FakePeftModel()
        trainer.save_lora_checkpoint(5)
        assert (trainer.ckpt_dir / "epoch_5" / "adapter_config.json").exists()

    def test_resume_restores_adapter_path_and_state(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer._model = _FakePeftModel()
        trainer.current_epoch = 2
        trainer._global_step = 11
        trainer.save_lora_checkpoint(3)

        fresh = _make_trainer(tmp_path)
        start_epoch = fresh.resume(trainer.ckpt_dir / "epoch_3")
        assert start_epoch == 3
        assert fresh._global_step == 11
        assert fresh.current_epoch == 3
        assert fresh.sft_adapter.endswith("epoch_3")

    def test_resume_falls_back_to_dir_name(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        ckpt = trainer.ckpt_dir / "epoch_9"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text("{}")
        start = trainer.resume(ckpt)
        assert start == 9

    def test_resume_missing_dir_raises(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(FileNotFoundError):
            trainer.resume(tmp_path / "does_not_exist")

    def test_save_without_model_is_noop(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer._model = None
        trainer.save_lora_checkpoint("latest")
        assert not (trainer.ckpt_dir / "latest" / "trainer_state.json").exists()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_reports_scene_metrics(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "scripts.run_scene_grpo.run_scene_episode",
            lambda agent, scene_id, **kw: _episode(
                scene_id, 1.0,
                recon_result={"num_registered": 3, "mean_pose_error_deg": 2.0},
            ),
        )
        val = [{"scene_id": f"v{i}", "image_paths": [], "num_images": 3}
               for i in range(3)]
        trainer = _make_trainer(tmp_path)
        trainer.val_scenes = val

        stats = trainer.evaluate(ScriptedSceneAgent())
        assert stats["num_scenes"] == 3
        assert stats["mean_reward"] == pytest.approx(1.0)
        assert stats["success_rate"] == pytest.approx(1.0)
        assert stats["mean_registered"] == pytest.approx(3.0)
        assert stats["scene/mean_pose_error_deg"] == pytest.approx(2.0)
        # eval uses the unscheduled reward config
        agent = ScriptedSceneAgent()
        trainer.evaluate(agent)
        assert agent.reward_config["pose_weight"] == pytest.approx(0.9)

    def test_evaluate_with_no_scenes(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        stats = trainer.evaluate(ScriptedSceneAgent())
        assert stats["mean_reward"] == 0.0 and stats["num_scenes"] == 0
