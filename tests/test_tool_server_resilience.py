"""Tool-server resilience: survive native COLMAP/FAISS crashes.

Job 7882190 (2026-09-16 08:17) died this way: pycolmap's SIFT matcher built a
FAISS ``FeatureDescriptorIndex`` (IVF clustering), SIGBUS'd inside
``faiss::Clustering::train_encoded``, and aborted the uvicorn process. The
trainer then rolled out against a dead server for 11 hours, logging zeros.

Three defences, each tested here:
  1. matching options never touch FAISS (brute-force matcher)
  2. COLMAP runs in a child process, so a native abort can't kill the server
  3. a watchdog kills training if the server can't be restored
"""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "tools_server_resilience", ROOT / "tools_server" / "server.py"
)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)

WORKER = ROOT / "tools_server" / "sfm_worker.py"
WATCHDOG = ROOT / "scripts" / "watch_tool_server.sh"


def _textured_dir(d: Path, n: int = 4):
    """Images sharing a shifted pattern, so SIFT finds real correspondences."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    for i in range(n):
        Image.fromarray(np.roll(base, shift=i * 40, axis=1)).save(d / f"im{i}.jpg")
    return str(d)


# ---------------------------------------------------------------------------
# 1. Matching options avoid the FAISS path
# ---------------------------------------------------------------------------

class TestMatchingOptions:
    def test_brute_force_matcher_enabled(self):
        opts = TS.colmap_matching_options()
        assert opts.sift.cpu_brute_force_matcher is True

    def test_default_would_use_faiss(self):
        """Guard the premise: the default is the FAISS-backed matcher."""
        import pycolmap

        assert pycolmap.FeatureMatchingOptions().sift.cpu_brute_force_matcher is False

    def test_worker_uses_brute_force(self):
        spec = importlib.util.spec_from_file_location("sfm_worker_mod", WORKER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._matching_options().sift.cpu_brute_force_matcher is True


# ---------------------------------------------------------------------------
# 2. Real COLMAP run in the worker (no FAISS, no crash)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestWorkerEndToEnd:
    def test_worker_reconstructs(self, tmp_path):
        pytest.importorskip("pycolmap")
        import pycolmap

        image_dir = _textured_dir(tmp_path / "imgs")
        db_path = str(tmp_path / "database.db")
        db = pycolmap.Database.open(db_path)
        db.close()

        payload = {
            "db_path": db_path,
            "image_dir": image_dir,
            "sparse_dir": str(tmp_path / "sparse"),
            "used_agent_matches": False,
            "pair_list": None,
            "num_pairs_requested": 0,
        }
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [sys.executable, str(WORKER)],
            input=json.dumps(payload), capture_output=True, text=True,
            env=env, timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-1500:]
        out = json.loads(proc.stdout)
        assert out.get("error") is None
        assert out["num_registered"] == 4
        assert out["num_points3d"] > 0
        assert Path(out["output_dir"]).exists()


# ---------------------------------------------------------------------------
# 3. A crashing worker must not take the server down
# ---------------------------------------------------------------------------

class TestCrashIsolation:
    def test_crashing_worker_returns_error_server_survives(self, tmp_path, monkeypatch):
        crashing = tmp_path / "crash_worker.py"
        crashing.write_text(
            "import os, sys\n"
            "sys.stdin.read()\n"
            "os.kill(os.getpid(), 7)\n"  # SIGBUS, as FAISS did
        )
        monkeypatch.setattr(TS, "_SFM_WORKER_PATH", crashing)

        image_dir = _textured_dir(tmp_path / "imgs", n=2)
        result = TS.tool_sfm_run(
            image_dir, pair_list=[["im0", "im1"]], output_dir=str(tmp_path / "o")
        )
        assert result["status"] == "crash"
        assert "crashed" in result["error"]
        assert result["num_registered"] == 0

        # The module is still importable/usable — the server never aborted.
        assert TS.health()["status"] == "ok"

    def test_missing_worker_path_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(TS, "_SFM_WORKER_PATH", tmp_path / "nope.py")
        image_dir = _textured_dir(tmp_path / "imgs2", n=2)
        result = TS.tool_sfm_run(
            image_dir, pair_list=[["im0", "im1"]], output_dir=str(tmp_path / "o2")
        )
        # Either an explicit error or a crash report — never a silent success.
        assert "error" in result
        assert result.get("num_registered", 0) == 0


# ---------------------------------------------------------------------------
# 4. Watchdog stops training when the server can't be restored
# ---------------------------------------------------------------------------

class TestWatchdog:
    def test_script_is_executable(self):
        assert WATCHDOG.exists()
        assert os.access(WATCHDOG, os.X_OK)

    def test_kills_protected_pid_when_unrestorable(self):
        """Health never recovers -> the watchdog must TERM the trainer.

        Uses a dead port and a throwaway child as the 'training' process, so
        the test asserts the real fail-fast contract without touching a GPU.
        """
        # Nothing is listening on this port; restart attempts will fail fast.
        dead_port = 59311
        protect = subprocess.Popen(["sleep", "300"])
        try:
            proc = subprocess.run(
                ["bash", str(WATCHDOG),
                 "--port", str(dead_port),
                 "--project-root", "/nonexistent-project",
                 "--protect-pid", str(protect.pid),
                 "--pid-file", "/tmp/watchdog-test.pid",
                 "--interval", "1",
                 "--fail-threshold", "1",
                 "--max-restarts", "1"],
                capture_output=True, text=True, timeout=120,
            )
            # The watchdog exits non-zero after killing the trainer.
            assert proc.returncode == 2, proc.stdout + proc.stderr
            deadline = time.time() + 10
            while time.time() < deadline and protect.poll() is None:
                time.sleep(0.2)
            assert protect.poll() is not None, "watchdog did not kill the trainer"
        finally:
            if protect.poll() is None:
                protect.send_signal(signal.SIGKILL)
                protect.wait()
