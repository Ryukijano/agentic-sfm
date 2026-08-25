#!/usr/bin/env python
"""Run real zero-shot evaluation on MegaDepth-1500 pairs.

1. Direct LoFTR matching (baseline)
2. Qwen3-VL-8B agent with crop+match tool calling
3. Save results + match visualizations (SuperGlue-style)
"""
import json
import os
import sys
import time
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
            img_a = img_a[y:y+h, x:x+w]
        if crop_b:
            x, y, w, h = crop_b
            img_b = img_b[y:y+h, x:x+w]

        # Resize to max 640
        for img in [img_a, img_b]:
            pass
        h_a, w_a = img_a.shape
        h_b, w_b = img_b.shape
        max_dim = 640
        if max(h_a, w_a) > max_dim:
            scale = max_dim / max(h_a, w_a)
            img_a = cv2.resize(img_a, (int(w_a*scale), int(h_a*scale)))
        if max(h_b, w_b) > max_dim:
            scale = max_dim / max(h_b, w_b)
            img_b = cv2.resize(img_b, (int(w_b*scale), int(h_b*scale)))

        t_a = torch.from_numpy(img_a).float()[None, None] / 255.0
        t_b = torch.from_numpy(img_b).float()[None, None] / 255.0

        with torch.no_grad():
            input_dict = {"image0": t_a.to(self.device), "image1": t_b.to(self.device)}
            result = self.matcher(input_dict)

        mkpts_a = result["keypoints0"].cpu().numpy()
        mkpts_b = result["keypoints1"].cpu().numpy()
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
def count_inliers(match_result, K_a, K_b, gt_R, gt_t, threshold=4.0):
    """Count RANSAC inliers using essential matrix from GT pose."""
    mkpts_a = match_result["mkpts_a"]
    mkpts_b = match_result["mkpts_b"]

    if len(mkpts_a) < 5:
        return 0, 0.0

    K_a = np.array(K_a)
    K_b = np.array(K_b)

    # Scale keypoints back to original image coordinates if cropped
    # (for now assume no crop)
    # Compute essential matrix from GT pose
    E = K_b.T @ np.cross(gt_t, np.eye(3)) @ gt_R @ K_a

    # Use cv2.findFundamentalMat with RANSAC
    F, mask = cv2.findFundamentalMat(
        mkpts_a.astype(np.float64), mkpts_b.astype(np.float64),
        cv2.USAC_MAGSAC, threshold, 0.999, 10000
    )

    if mask is None:
        return 0, 0.0

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(mkpts_a) if len(mkpts_a) > 0 else 0.0
    return inliers, inlier_ratio


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


# ── Agent (Qwen3-VL) ────────────────────────────────────────────────────
class SimpleAgent:
    """Qwen3-VL agent that decides crops for matching."""

    def __init__(self, device="cuda:0"):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_name = "Qwen/Qwen3-VL-8B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        print(f"Loading {model_name} with 4-bit quantization...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        print("Model loaded.")

    def decide_crop(self, img_a_path, img_b_path):
        """Ask the VLM to suggest crop regions for better matching."""
        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        # Resize for VLM input
        img_a.thumbnail((448, 448))
        img_b.thumbnail((448, 448))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_a},
                    {"type": "image", "image": img_b},
                    {"type": "text", "text": (
                        "You are looking at two images of the same scene from different viewpoints. "
                        "To improve feature matching, suggest a crop region for each image that "
                        "focuses on the overlapping area. "
                        "Respond in JSON format:\n"
                        '{"crop_a": [x, y, width, height], "crop_b": [x, y, width, height]}\n'
                        "where coordinates are in the original image space (0-100 percentage). "
                        "If no crop is needed, return full image bounds."
                    )}
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[img_a, img_b],
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
            )

        response = self.processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._parse_crop_response(response, img_a_path, img_b_path)

    def _parse_crop_response(self, response, img_a_path, img_b_path):
        """Parse crop suggestion from model response."""
        import re
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', response)
        crop_a = None
        crop_b = None

        if json_match:
            try:
                data = json.loads(json_match.group())
                if "crop_a" in data:
                    crop_a = self._percent_to_pixels(data["crop_a"], img_a_path)
                if "crop_b" in data:
                    crop_b = self._percent_to_pixels(data["crop_b"], img_b_path)
            except (json.JSONDecodeError, KeyError):
                pass

        return crop_a, crop_b, response

    def _percent_to_pixels(self, crop_pct, img_path):
        """Convert percentage crop to pixel coordinates."""
        img = Image.open(img_path)
        w, h = img.size
        if len(crop_pct) == 4:
            x_pct, y_pct, w_pct, h_pct = crop_pct
            x = int(x_pct / 100.0 * w)
            y = int(y_pct / 100.0 * h)
            cw = int(w_pct / 100.0 * w)
            ch = int(h_pct / 100.0 * h)
            return (x, y, cw, ch)
        return None


# ── Main evaluation ──────────────────────────────────────────────────────
def main():
    with open(DATA_PATH) as f:
        all_pairs = json.load(f)

    # Select subset: 3 per difficulty
    selected = []
    for diff in ["easy", "medium", "hard", "extreme"]:
        diff_pairs = [p for p in all_pairs if p["difficulty"] == diff]
        selected.extend(diff_pairs[:3])

    print(f"Evaluating {len(selected)} pairs ({3} per difficulty)")

    # Initialize LoFTR
    print("Loading LoFTR matcher...")
    matcher = LoFTRMatcher(device=DEVICE)

    # Initialize agent
    print("Loading Qwen3-VL-8B agent...")
    agent = SimpleAgent(device=DEVICE)

    results = []

    for i, p in enumerate(selected):
        pair_id = p["pair_id"]
        diff = p["difficulty"]
        print(f"\n[{i+1}/{len(selected)}] {pair_id} ({diff})")

        img_a_path = p["image_a"]
        img_b_path = p["image_b"]
        gt_R = np.array(p["gt_R"])
        gt_t = np.array(p["gt_t"])
        K_a = np.array(p["K_a"])
        K_b = np.array(p["K_b"])

        # ── Direct LoFTR (baseline) ──
        print("  Direct LoFTR matching...")
        direct_result = matcher.match(img_a_path, img_b_path)
        direct_inliers, direct_ratio = count_inliers(direct_result, K_a, K_b, gt_R, gt_t)

        # RANSAC mask for visualization
        mkpts_a = direct_result["mkpts_a"]
        mkpts_b = direct_result["mkpts_b"]
        if len(mkpts_a) >= 5:
            _, direct_mask = cv2.findFundamentalMat(
                mkpts_a.astype(np.float64), mkpts_b.astype(np.float64),
                cv2.USAC_MAGSAC, 4.0, 0.999
            )
            direct_mask = direct_mask.ravel().astype(bool) if direct_mask is not None else np.ones(len(mkpts_a), dtype=bool)
        else:
            direct_mask = np.ones(len(mkpts_a), dtype=bool)

        print(f"  Direct: {direct_result['n_matches']} matches, {direct_inliers} inliers ({direct_ratio:.1%})")

        # Visualize direct matching
        viz_path = VIZ_DIR / f"{pair_id}_direct.png"
        visualize_matches(
            img_a_path, img_b_path, direct_result,
            inlier_mask=direct_mask,
            output_path=viz_path,
            title=f"Direct LoFTR | {diff}"
        )

        # ── Agent + LoFTR ──
        print("  Agent crop decision...")
        crop_a, crop_b, agent_response = agent.decide_crop(img_a_path, img_b_path)

        if crop_a or crop_b:
            print(f"  Crop A: {crop_a}")
            print(f"  Crop B: {crop_b}")
            agent_result = matcher.match(img_a_path, img_b_path, crop_a=crop_a, crop_b=crop_b)
        else:
            print("  No crop suggested, using full images")
            agent_result = direct_result

        agent_inliers, agent_ratio = count_inliers(agent_result, K_a, K_b, gt_R, gt_t)

        # RANSAC mask for agent
        mkpts_a2 = agent_result["mkpts_a"]
        mkpts_b2 = agent_result["mkpts_b"]
        if len(mkpts_a2) >= 5:
            _, agent_mask = cv2.findFundamentalMat(
                mkpts_a2.astype(np.float64), mkpts_b2.astype(np.float64),
                cv2.USAC_MAGSAC, 4.0, 0.999
            )
            agent_mask = agent_mask.ravel().astype(bool) if agent_mask is not None else np.ones(len(mkpts_a2), dtype=bool)
        else:
            agent_mask = np.ones(len(mkpts_a2), dtype=bool)

        print(f"  Agent: {agent_result['n_matches']} matches, {agent_inliers} inliers ({agent_ratio:.1%})")

        # Visualize agent matching
        viz_path_agent = VIZ_DIR / f"{pair_id}_agent.png"
        visualize_matches(
            img_a_path, img_b_path, agent_result,
            inlier_mask=agent_mask,
            output_path=viz_path_agent,
            title=f"Agent+LoFTR | {diff}",
            crop_a=crop_a, crop_b=crop_b
        )

        results.append({
            "pair_id": pair_id,
            "difficulty": diff,
            "overlap": p["overlap_score"],
            "scene": p["scene"],
            "direct_matches": direct_result["n_matches"],
            "direct_inliers": direct_inliers,
            "direct_inlier_ratio": direct_ratio,
            "agent_matches": agent_result["n_matches"],
            "agent_inliers": agent_inliers,
            "agent_inlier_ratio": agent_ratio,
            "crop_a": crop_a,
            "crop_b": crop_b,
            "agent_response": agent_response[:500],
        })

    # Save results
    results_path = OUTPUT_DIR / "real_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Visualizations saved to {VIZ_DIR}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['pair_id']:20s} | {r['difficulty']:8s} | "
              f"Direct: {r['direct_inliers']:3d}/{r['direct_matches']:3d} ({r['direct_inlier_ratio']:.1%}) | "
              f"Agent: {r['agent_inliers']:3d}/{r['agent_matches']:3d} ({r['agent_inlier_ratio']:.1%})")


if __name__ == "__main__":
    main()
