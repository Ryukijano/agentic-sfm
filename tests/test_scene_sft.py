"""Tests for scripts/build_scene_sft.py — scene-level oracle SFT data.

Offline: a synthetic MegaDepth-style scene (npz + real JPEGs) is loaded
through the real ``SceneDataset`` loader; oracle episodes are generated with
synthesized GT-informed tool results — no GPU, no tool server.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import parse_tool_call  # noqa: E402
from agentic_sfm.rl.scene_episode import SCENE_SYSTEM_PROMPT  # noqa: E402
from scripts.build_scene_sft import (  # noqa: E402
    ALLOWED_SCENE_TOOLS,
    build_examples,
    build_oracle_episode,
)
from scripts.run_scene_grpo import SceneDataset  # noqa: E402

N_IMAGES = 14


def _write_scene(si_dir: Path, img_root: Path, scene_id: str = "0015",
                 n_images: int = N_IMAGES, seed: int = 0) -> str:
    """Synthetic MegaDepth scene with a *banded* overlap matrix.

    Unlike ``scene_smoke_test.write_synthetic_scene`` (np.eye), off-diagonal
    overlaps are non-zero for |i-j| <= 4 so the oracle has real candidate
    pairs, plus a tail of <=0 pairs for the doppelganger branch.
    """
    from PIL import Image

    si_dir = Path(si_dir)
    img_root = Path(img_root)
    si_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    paths: list[str] = []
    for i in range(n_images):
        rel = f"{scene_id}/img_{i:04d}.jpg"
        p = img_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(p, "JPEG")
        paths.append(rel)

    poses = np.empty(n_images, dtype=object)
    intrinsics = np.empty(n_images, dtype=object)
    for i in range(n_images):
        M = np.eye(4)
        M[0, 3] = -float(i)  # cam-from-world R=I: camera center at (i, 0, 0)
        poses[i] = M
        intrinsics[i] = np.array(
            [[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]]
        )

    overlap = np.full((n_images, n_images), -1.0)
    for i in range(n_images):
        for j in range(n_images):
            if i == j:
                continue
            d = abs(i - j)
            overlap[i, j] = max(0.0, 0.85 - 0.18 * d) if d <= 4 else 0.0
    # A few explicit zero-overlap pairs act as look-alike candidates.
    overlap[0, n_images - 1] = 0.0
    overlap[n_images - 1, 0] = 0.0

    np.savez(
        si_dir / f"{scene_id}.npz",
        image_paths=np.array(paths),
        poses=poses,
        intrinsics=intrinsics,
        overlap_matrix=overlap,
    )
    return scene_id


@pytest.fixture
def scene(tmp_path):
    """Dataset + image_root for the synthetic scene."""
    si_dir = tmp_path / "scene_info"
    img_root = tmp_path / "images"
    _write_scene(si_dir, img_root)
    ds = SceneDataset.from_megadepth(
        str(si_dir), str(img_root), scenes=["0015"], max_images=N_IMAGES)
    assert len(ds.scenes) == 1
    return ds.scenes[0], str(img_root), str(si_dir)


def _check_example_schema(ex: dict) -> None:
    """Assert one SFT example matches the run_sft.py consumable schema."""
    for key in ("scene_id", "episode_idx", "image_paths", "num_images",
                "messages", "reward", "reward_components", "recon_result",
                "num_tool_calls"):
        assert key in ex, f"missing key {key}"
    assert ex["num_images"] == len(ex["image_paths"])

    msgs = ex["messages"]
    assert msgs[0] == {"role": "system", "content": SCENE_SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert isinstance(msgs[1]["content"], list)
    assert msgs[1]["content"][-1]["type"] == "text"
    n_img_items = sum(1 for c in msgs[1]["content"] if c.get("type") == "image")
    assert n_img_items == min(ex["num_images"], 8)

    # After the scene prompt: assistant / user(observation) turns, ending
    # on the assistant's done call (done has no observation).
    calls = []
    for k, msg in enumerate(msgs[2:], start=2):
        if msg["role"] == "assistant":
            tc = parse_tool_call(msg["content"])
            assert tc is not None, f"unparseable assistant turn {k}: {msg['content']!r}"
            assert tc.tool in ALLOWED_SCENE_TOOLS
            calls.append(tc.tool)
        else:
            assert msg["role"] == "user"
            assert isinstance(msg["content"], str)
            assert msg["content"].startswith("Observation:")

    assert calls[0] == "retrieve"
    assert calls[-1] == "done"
    assert "sfm_run" in calls and "inspect" in calls
    # roles strictly alternate assistant -> user(obs) except after `done`
    roles = [m["role"] for m in msgs[2:]]
    for a, b in zip(roles, roles[1:]):
        assert not (a == "assistant" and b == "assistant")
    assert len(calls) == ex["num_tool_calls"]


class TestOracleEpisode:
    def test_schema_single_episode(self, scene):
        s, img_root, _ = scene
        rng = np.random.default_rng(0)
        ex = build_oracle_episode(s, img_root, ep_idx=0, rng=rng)
        _check_example_schema(ex)

    def test_episode_variation(self, scene):
        """Different rng draws vary tool sequences / scene subset handling."""
        s, img_root, _ = scene
        rng = np.random.default_rng(7)
        tool_seqs = set()
        has_doppel = has_crop = 0
        for ep in range(8):
            ex = build_oracle_episode(s, img_root, ep_idx=ep, rng=rng)
            _check_example_schema(ex)
            tools = tuple(
                parse_tool_call(m["content"]).tool
                for m in ex["messages"]
                if m["role"] == "assistant"
            )
            tool_seqs.add(tools)
            has_doppel += "doppelganger_check" in tools
            has_crop += ("crop_and_match" in tools) or ("crop" in tools)
        assert len(tool_seqs) > 1            # trajectories actually vary
        assert has_doppel >= 1               # doppelganger branch exercised
        assert has_crop >= 1                 # crop grammar exercised

    def test_flagged_doppelganger_excluded_from_recon(self, scene):
        """When a check flags, sfm_run reports it present + filtered."""
        s, img_root, _ = scene
        rng = np.random.default_rng(0)
        for ep in range(12):
            ex = build_oracle_episode(s, img_root, ep_idx=ep, rng=rng)
            recon = ex["recon_result"]
            if recon.get("num_doppelgangers_present"):
                assert recon["num_doppelgangers_filtered"] == \
                    recon["num_doppelgangers_present"]
                return
        pytest.fail("no episode flagged a doppelganger in 12 draws")


class TestBuildExamples:
    def test_end_to_end_and_jsonl(self, scene, tmp_path):
        s, img_root, si_dir = scene
        examples = build_examples(
            scene_info_dir=si_dir,
            image_root=img_root,
            scenes=["0015"],
            num_episodes=4,
            seed=42,
            min_images=6,
            max_images=12,
            pool_images=N_IMAGES,
        )
        assert len(examples) == 4
        out = tmp_path / "scene_sft.jsonl"
        with open(out, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        for line in out.read_text().splitlines():
            ex = json.loads(line)
            _check_example_schema(ex)

    def test_deterministic_seed(self, scene):
        s, img_root, si_dir = scene
        kwargs = dict(
            scene_info_dir=si_dir, image_root=img_root, scenes=["0015"],
            num_episodes=3, seed=123, min_images=6, max_images=10,
            pool_images=N_IMAGES,
        )
        a = build_examples(**kwargs)
        b = build_examples(**kwargs)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_image_paths_resolve(self, scene):
        s, img_root, si_dir = scene
        [ex] = build_examples(
            scene_info_dir=si_dir, image_root=img_root, scenes=["0015"],
            num_episodes=1, seed=5, min_images=6, max_images=10,
            pool_images=N_IMAGES,
        )
        for p in ex["image_paths"]:
            assert Path(p).exists(), p


class TestSFTLoaderCompat:
    def test_load_example_images_scene(self, scene):
        """run_sft._load_example_images consumes scene examples."""
        from scripts.run_sft import _count_image_slots, _load_example_images

        s, img_root, _ = scene
        rng = np.random.default_rng(3)
        ex = build_oracle_episode(s, img_root, ep_idx=0, rng=rng)
        images = _load_example_images(ex)
        n_slots = _count_image_slots(ex["messages"])
        assert n_slots == min(ex["num_images"], 8)
        assert len(images) == n_slots

    def test_load_example_images_pair_still_works(self, scene, tmp_path):
        """Pair-level examples (image_a/image_b) are unaffected."""
        from scripts.run_sft import _load_example_images

        from PIL import Image
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(a)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(b)
        ex = {
            "image_a": str(a),
            "image_b": str(b),
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"}, {"type": "image"},
                    {"type": "text", "text": "hi"}]},
            ],
        }
        assert len(_load_example_images(ex)) == 2
