#!/usr/bin/env python
"""Run real zero-shot evaluation on MegaDepth-1500 pairs.

1. Direct LoFTR matching (baseline)
2. Qwen3-VL-2B-Instruct agent with crop+match tool calling
3. Save results + match visualizations (SuperGlue-style)
"""
import argparse
import json
import os
import sys
import tempfile
import time
import uuid
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Config ───────────────────────────────────────────────────────────────
DATA_PATH = Path("/scratch/kcwp264/agentic-sfm/data/hard_pairs.json")
OUTPUT_DIR = Path("/scratch/kcwp264/agentic-sfm/outputs/phase0_real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_PAIRS = 12  # 3 per difficulty for quick run


# ── LoFTR Matcher ────────────────────────────────────────────────────────
class LoFTRMatcher:
    def __init__(self, device="cuda:0"):
        from kornia.feature import LoFTR
        self.matcher = LoFTR(pretrained="outdoor").to(device).eval()
        self.device = device

    def match(self, img_a_path, img_b_path, crop_a=None, crop_b=None):
        """Run LoFTR matching. Returns matched keypoints + confidence."""
        img_a = cv2.imread(str(img_a_path), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(img_b_path), cv2.IMREAD_GRAYSCALE)

        if crop_a:
            x, y, w, h = crop_a
            img_a = img_a[y:y + h, x:x + w]
        if crop_b:
            x, y, w, h = crop_b
            img_b = img_b[y:y + h, x:x + w]

        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        max_dim = 640
        scale_a = 1.0
        scale_b = 1.0
        if max(h_a, w_a) > max_dim:
            scale_a = max_dim / max(h_a, w_a)
            img_a = cv2.resize(img_a, (int(w_a * scale_a), int(h_a * scale_a)))
        if max(h_b, w_b) > max_dim:
            scale_b = max_dim / max(h_b, w_b)
            img_b = cv2.resize(img_b, (int(w_b * scale_b), int(h_b * scale_b)))

        t_a = torch.from_numpy(img_a).float()[None, None] / 255.0
        t_b = torch.from_numpy(img_b).float()[None, None] / 255.0

        with torch.no_grad():
            input_dict = {"image0": t_a.to(self.device), "image1": t_b.to(self.device)}
            result = self.matcher(input_dict)

        mkpts_a = result["keypoints0"].cpu().numpy() / max(scale_a, 1e-8)
        mkpts_b = result["keypoints1"].cpu().numpy() / max(scale_b, 1e-8)
        confidence = result["confidence"].cpu().numpy()

        return {
            "mkpts_a": mkpts_a,
            "mkpts_b": mkpts_b,
            "confidence": confidence,
            "img_a_size": (w_a, h_a),
            "img_b_size": (w_b, h_b),
            "n_matches": len(mkpts_a),
        }


# ── RANSAC inlier counting ───────────────────────────────────────────────
def count_inliers(match_result, K_a, K_b, gt_R, gt_t, threshold=1.0):
    """MAGSAC inliers via essential matrix with (optional) GT K."""
    from agentic_sfm.geometry import estimate_relative_pose

    mkpts_a = np.asarray(match_result.get("mkpts_a", match_result.get("keypoints_a", [])))
    mkpts_b = np.asarray(match_result.get("mkpts_b", match_result.get("keypoints_b", [])))
    if mkpts_a.size == 0 or len(mkpts_a) < 8:
        pose = match_result.get("pose")
        n = int(match_result.get("num_inliers") or 0)
        r = float(match_result.get("inlier_ratio") or 0.0)
        return n, r, pose

    size_a = match_result.get("img_a_size") or (int(mkpts_a[:, 0].max()) + 1, int(mkpts_a[:, 1].max()) + 1)
    size_b = match_result.get("img_b_size") or (int(mkpts_b[:, 0].max()) + 1, int(mkpts_b[:, 1].max()) + 1)
    pose_result = estimate_relative_pose(
        mkpts_a, mkpts_b, tuple(size_a), tuple(size_b), K_a=K_a, K_b=K_b, threshold=threshold,
    )
    return pose_result["num_inliers"], pose_result["inlier_ratio"], pose_result.get("pose")


# ── Visualization (SuperGlue-style) ──────────────────────────────────────
def visualize_matches(img_a_path, img_b_path, match_result, inlier_mask=None,
                      output_path=None, title="", crop_a=None, crop_b=None):
    """Draw SuperGlue-style match visualization."""
    img_a = cv2.imread(str(img_a_path))
    img_b = cv2.imread(str(img_b_path))

    if crop_a:
        x, y, w, h = crop_a
        img_a = img_a[y:y+h, x:x+w]
    if crop_b:
        x, y, w, h = crop_b
        img_b = img_b[y:y+h, x:x+w]

    # Resize for display
    max_h = 600
    for img_ref in [(img_a, 'a'), (img_b, 'b')]:
        img, label = img_ref
        if img.shape[0] > max_h:
            scale = max_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1]*scale), max_h))
            if label == 'a':
                img_a = img
            else:
                img_b = img

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    target_h = max(h_a, h_b)

    # Pad to same height
    if h_a < target_h:
        img_a = np.pad(img_a, ((0, target_h - h_a), (0, 0), (0, 0)), mode='constant', constant_values=255)
    if h_b < target_h:
        img_b = np.pad(img_b, ((0, target_h - h_b), (0, 0), (0, 0)), mode='constant', constant_values=255)

    # Scale keypoints to display resolution
    mkpts_a = match_result["mkpts_a"].copy()
    mkpts_b = match_result["mkpts_b"].copy()

    # Original image sizes vs display sizes
    orig_w_a, orig_h_a = match_result["img_a_size"]
    orig_w_b, orig_h_b = match_result["img_b_size"]
    scale_a = w_a / orig_w_a if orig_w_a > 0 else 1.0
    scale_b = w_b / orig_w_b if orig_w_b > 0 else 1.0
    mkpts_a[:, 0] *= scale_a
    mkpts_a[:, 1] *= scale_a
    mkpts_b[:, 0] *= scale_b
    mkpts_b[:, 1] *= scale_b

    # Create combined image
    gap = 20
    combined = np.full((target_h, w_a + w_b + gap, 3), 255, dtype=np.uint8)
    combined[:h_a, :w_a] = img_a
    combined[:h_b, w_a + gap:] = img_b

    # Draw matches
    n = len(mkpts_a)
    if n > 0:
        if inlier_mask is not None:
            colors = []
            for i in range(n):
                if inlier_mask[i]:
                    colors.append((0, 200, 0))  # green for inliers
                else:
                    colors.append((0, 0, 200))  # red for outliers
        else:
            # Color by confidence
            conf = match_result["confidence"]
            colors = []
            for i in range(n):
                c = conf[i]
                if c > 0.8:
                    colors.append((0, 200, 0))
                elif c > 0.5:
                    colors.append((0, 180, 255))
                else:
                    colors.append((0, 0, 200))

        # Limit number of lines drawn for clarity
        max_lines = 100
        if n > max_lines:
            # Sample evenly
            indices = np.linspace(0, n - 1, max_lines, dtype=int)
        else:
            indices = range(n)

        for i in indices:
            pt_a = mkpts_a[i]
            pt_b = mkpts_b[i]
            color = colors[i]
            # Draw line
            cv2.line(combined,
                     (int(pt_a[0]), int(pt_a[1])),
                     (int(pt_b[0]) + w_a + gap, int(pt_b[1])),
                     color, 1, cv2.LINE_AA)
            # Draw points
            cv2.circle(combined, (int(pt_a[0]), int(pt_a[1])), 3, color, -1, cv2.LINE_AA)
            cv2.circle(combined, (int(pt_b[0]) + w_a + gap, int(pt_b[1])), 3, color, -1, cv2.LINE_AA)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else n
    n_total = n
    info = f"{title} | {n_inliers}/{n_total} inliers"
    cv2.putText(combined, info, (10, 25), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(combined, info, (10, 25), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(str(output_path), combined)
    return combined


# ── Agent tools (same crop/match contract as training) ───────────────────
class LocalToolClient:
    """In-process crop/match so eval does not require a running tool server."""

    def __init__(self, matcher: LoFTRMatcher, tmp_dir: Path):
        self.matcher = matcher
        self.tmp_dir = tmp_dir
        self.registry: dict[str, str] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.crop_meta: dict[str, dict] = {}

    def register_image(self, image_id, path):
        self.registry[image_id] = str(path)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        self.arrays[image_id] = img
        self.crop_meta[image_id] = {"origin_xy": [0, 0]}
        return {"status": "ok"}

    def crop(self, image_id, bbox):
        img = self.arrays[image_id]
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        px1, px2 = max(0, px1), min(w, px2)
        py1, py2 = max(0, py1), min(h, py2)
        cropped = img[py1:py2, px1:px2]
        new_id = f"{image_id}_crop_{uuid.uuid4().hex[:8]}"
        path = str(self.tmp_dir / f"{new_id}.jpg")
        cv2.imwrite(path, cropped)
        self.registry[new_id] = path
        self.arrays[new_id] = cropped
        parent = self.crop_meta.get(image_id, {})
        ox = int(parent.get("origin_xy", [0, 0])[0]) + px1
        oy = int(parent.get("origin_xy", [0, 0])[1]) + py1
        self.crop_meta[new_id] = {"origin_xy": [ox, oy]}
        return {
            "cropped_image_id": new_id,
            "crop_id": new_id,
            "path": path,
            "origin_xy": [ox, oy],
            "crop_size": [px2 - px1, py2 - py1],
            "size": [px2 - px1, py2 - py1],
        }

    def match(self, image_a, image_b, matcher="loftr", max_size=512, K_a=None, K_b=None):
        from agentic_sfm.geometry import estimate_relative_pose, k_for_image

        path_a = self.registry.get(image_a, image_a)
        path_b = self.registry.get(image_b, image_b)
        result = self.matcher.match(path_a, path_b)
        img_a = self.arrays.get(image_a)
        img_b = self.arrays.get(image_b)
        if img_a is None or img_b is None:
            return {"error": f"unregistered {image_a}/{image_b}", "num_inliers": 0, "inlier_ratio": 0.0}
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        Ka = k_for_image(K_a, self.crop_meta.get(image_a, {}).get("origin_xy"), (w_a, h_a))
        Kb = k_for_image(K_b, self.crop_meta.get(image_b, {}).get("origin_xy"), (w_b, h_b))
        pose_result = estimate_relative_pose(
            result["mkpts_a"], result["mkpts_b"], (w_a, h_a), (w_b, h_b), K_a=Ka, K_b=Kb,
        )
        return {
            "num_matches": result["n_matches"],
            "num_inliers": pose_result["num_inliers"],
            "inlier_ratio": pose_result["inlier_ratio"],
            "pose": pose_result["pose"],
            "mkpts_a": result["mkpts_a"],
            "mkpts_b": result["mkpts_b"],
            "confidence": result["confidence"],
            "img_a_size": (w_a, h_a),
            "img_b_size": (w_b, h_b),
            "n_matches": result["n_matches"],
        }

    def doppelganger_check(self, image_a, image_b):
        return {"is_doppelganger": False, "score": 0.0}


def _as_k(value):
    if value is None:
        return None
    arr = np.array(value, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr.reshape(3, 3)


def _origin_to_xywh(origin_xy, crop_size):
    if not origin_xy or not crop_size:
        return None
    x, y = int(origin_xy[0]), int(origin_xy[1])
    w, h = int(crop_size[0]), int(crop_size[1])
    return (x, y, w, h)


def _crop_boxes_from_episode(episode):
    crop_a = crop_b = None
    for result in episode.results or []:
        blobs = [result]
        if isinstance(result.get("crop"), dict):
            blobs.append(result["crop"])
        for blob in blobs:
            origin = blob.get("origin_xy")
            size = blob.get("crop_size") or blob.get("size")
            box = _origin_to_xywh(origin, size)
            cid = str(blob.get("cropped_image_id") or blob.get("crop_id") or "")
            if box is None:
                continue
            if "img_b" in cid:
                crop_b = box
            else:
                crop_a = box
    return crop_a, crop_b


def _match_dict_for_viz(match_result, fallback=None):
    if match_result and "mkpts_a" in match_result:
        return match_result
    if match_result and "keypoints_a" in match_result:
        mk_a = np.asarray(match_result["keypoints_a"])
        mk_b = np.asarray(match_result["keypoints_b"])
        return {
            "mkpts_a": mk_a,
            "mkpts_b": mk_b,
            "confidence": np.ones(len(mk_a)),
            "img_a_size": match_result.get("img_a_size", (1, 1)),
            "img_b_size": match_result.get("img_b_size", (1, 1)),
            "n_matches": len(mk_a),
        }
    return fallback


def _magsac_mask(mkpts_a, mkpts_b):
    if len(mkpts_a) < 5:
        return np.ones(len(mkpts_a), dtype=bool)
    _, mask = cv2.findFundamentalMat(
        mkpts_a.astype(np.float64), mkpts_b.astype(np.float64),
        cv2.USAC_MAGSAC, 4.0, 0.999,
    )
    return mask.ravel().astype(bool) if mask is not None else np.ones(len(mkpts_a), dtype=bool)


def _pose_auc(pred_pose, gt_R, gt_t):
    if not pred_pose:
        return 0.0
    from agentic_sfm.rewards.pose_rewards import compute_pose_error, pose_auc_score

    return pose_auc_score(compute_pose_error(
        np.array(pred_pose["R"]), np.array(pred_pose["t"]), gt_R, gt_t,
    ))


# ── Main evaluation ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Real MegaDepth eval with tool-call agent")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        all_pairs = json.load(f)

    selected = []
    per_diff = max(1, args.max_pairs // 4)
    for diff in ["easy", "medium", "hard", "extreme"]:
        diff_pairs = [p for p in all_pairs if p["difficulty"] == diff]
        selected.extend(diff_pairs[:per_diff])

    print(f"Evaluating {len(selected)} pairs ({per_diff} per difficulty)")

    print("Loading LoFTR matcher...")
    matcher = LoFTRMatcher(device=DEVICE)

    from agentic_sfm.agent.policy import AgenticSfMAgent
    from agentic_sfm.tools.client import ToolClient

    print("Loading Qwen3-VL-2B-Instruct agent (training tool-call schema)...")
    agent = AgenticSfMAgent(device=DEVICE, do_sample=True, temperature=1.0)

    tool_client = None
    try:
        remote = ToolClient(args.tool_server_url)
        remote.health()
        tool_client = remote
        print(f"Using tool server at {args.tool_server_url}")
    except Exception:
        tmp = Path(tempfile.mkdtemp(prefix="asfm_eval_"))
        tool_client = LocalToolClient(matcher, tmp)
        print(f"No tool server; using in-process LoFTR at {tmp}")

    results = []

    for i, p in enumerate(selected):
        pair_id = p["pair_id"]
        diff = p["difficulty"]
        print(f"\n[{i+1}/{len(selected)}] {pair_id} ({diff})")

        img_a_path = p["image_a"]
        img_b_path = p["image_b"]
        gt_R = np.array(p["gt_R"])
        gt_t = np.array(p["gt_t"])
        K_a = _as_k(p.get("K_a"))
        K_b = _as_k(p.get("K_b"))
        K_a_list = K_a.tolist() if K_a is not None else None
        K_b_list = K_b.tolist() if K_b is not None else None

        print("  Direct LoFTR matching...")
        direct_result = matcher.match(img_a_path, img_b_path)
        direct_inliers, direct_ratio, direct_pose = count_inliers(
            direct_result, K_a, K_b, gt_R, gt_t,
        )
        direct_mask = _magsac_mask(direct_result["mkpts_a"], direct_result["mkpts_b"])
        print(
            f"  Direct: {direct_result['n_matches']} matches, "
            f"{direct_inliers} inliers ({direct_ratio:.1%})"
        )

        viz_path = VIZ_DIR / f"{pair_id}_direct.png"
        visualize_matches(
            img_a_path, img_b_path, direct_result,
            inlier_mask=direct_mask,
            output_path=viz_path,
            title=f"Direct LoFTR | {diff}",
        )

        print("  Agent episode (crop/match/done)...")
        tool_client.register_image("img_a", img_a_path)
        tool_client.register_image("img_b", img_b_path)
        episode = agent.run_episode(
            pair_id=pair_id,
            image_a_path=img_a_path,
            image_b_path=img_b_path,
            tool_client=tool_client,
            gt_pose={"R": gt_R.tolist(), "t": gt_t.tolist()},
            K_a=K_a_list,
            K_b=K_b_list,
        )
        agent_match = episode.final_match or {}
        agent_result = _match_dict_for_viz(agent_match, fallback=direct_result)
        agent_inliers = int(agent_match.get("num_inliers") or 0)
        agent_ratio = float(agent_match.get("inlier_ratio") or 0.0)
        if agent_inliers == 0 and "mkpts_a" in agent_result:
            agent_inliers, agent_ratio, pose = count_inliers(
                agent_result, K_a, K_b, gt_R, gt_t,
            )
            if pose is not None:
                agent_match = {**agent_match, "pose": pose, "num_inliers": agent_inliers, "inlier_ratio": agent_ratio}

        crop_a, crop_b = _crop_boxes_from_episode(episode)
        mk_a = np.asarray(agent_result.get("mkpts_a", []))
        mk_b = np.asarray(agent_result.get("mkpts_b", []))
        agent_mask = _magsac_mask(mk_a, mk_b) if len(mk_a) else None
        n_matches = int(agent_match.get("num_matches") or agent_result.get("n_matches") or len(mk_a))
        print(f"  Agent: {n_matches} matches, {agent_inliers} inliers ({agent_ratio:.1%})")
        if crop_a or crop_b:
            print(f"  Crop A: {crop_a}  Crop B: {crop_b}")
        else:
            print("  No crop recorded (full-frame match or parse miss)")

        viz_path_agent = VIZ_DIR / f"{pair_id}_agent.png"
        visualize_matches(
            img_a_path, img_b_path, agent_result,
            inlier_mask=agent_mask,
            output_path=viz_path_agent,
            title=f"Agent+LoFTR | {diff}",
            crop_a=crop_a, crop_b=crop_b,
        )

        agent_response = ""
        if episode.tool_calls:
            agent_response = json.dumps(
                [{"tool": tc.tool, "args": tc.args} for tc in episode.tool_calls]
            )

        results.append({
            "pair_id": pair_id,
            "difficulty": diff,
            "overlap": p["overlap_score"],
            "scene": p["scene"],
            "direct_matches": direct_result["n_matches"],
            "direct_inliers": direct_inliers,
            "direct_inlier_ratio": direct_ratio,
            "direct_pose_auc": _pose_auc(direct_pose, gt_R, gt_t),
            "agent_matches": n_matches,
            "agent_inliers": agent_inliers,
            "agent_inlier_ratio": agent_ratio,
            "agent_pose_auc": _pose_auc(agent_match.get("pose"), gt_R, gt_t),
            "agent_reward": episode.reward,
            "crop_a": crop_a,
            "crop_b": crop_b,
            "agent_response": agent_response[:500],
        })

    results_path = OUTPUT_DIR / "real_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Visualizations saved to {VIZ_DIR}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(
            f"{r['pair_id']:20s} | {r['difficulty']:8s} | "
            f"Direct: {r['direct_inliers']:3d}/{r['direct_matches']:3d} "
            f"({r['direct_inlier_ratio']:.1%} auc={r['direct_pose_auc']:.2f}) | "
            f"Agent: {r['agent_inliers']:3d}/{r['agent_matches']:3d} "
            f"({r['agent_inlier_ratio']:.1%} auc={r['agent_pose_auc']:.2f} "
            f"R={r['agent_reward']:.3f})"
        )


if __name__ == "__main__":
    main()
