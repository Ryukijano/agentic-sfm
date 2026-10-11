"""Generate publication-quality visual diagram of SFM models as tools for Agentic-SfM.

Panels:
A. Tool Tier Architecture: How the MLLM Director calls each model family.
B. Measured Performance on 30-Scene ScanNet Benchmark: LoFTR vs VGGT correspondences vs VGGT prior head.
C. Latency vs Geometric Capability Tradeoff.
D. Model Capability & Deployment Matrix.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": False,
})

PALETTE = {
    "agent": "#1E293B",       # Slate
    "retrieval": "#2563EB",   # Blue
    "matchers": "#EA580C",    # Orange/Red
    "priors": "#DB2777",      # Pink/Magenta
    "reconstruction": "#16A34A", # Green
    "bg_box": "#F8FAFC",
    "border": "#CBD5E1",
    "text": "#0F172A",
    "muted": "#475569",
}

def create_sfm_tools_figure(output_path: str):
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.22)

    # =========================================================================
    # Panel 1: Multi-Tier Model Architecture
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("A. Tool Ecosystem: 4 Tiers of SfM Models Available to Policy", loc="left", pad=12)
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Central Agent Node
    agent_box = patches.FancyBboxPatch((3.4, 7.8), 3.2, 1.7, boxstyle="round,pad=0.25",
                                      edgecolor=PALETTE["agent"], facecolor="#E2E8F0", linewidth=2.5)
    ax1.add_patch(agent_box)
    ax1.text(5.0, 8.9, "Photogrammetry Director", ha="center", va="center", weight="bold", fontsize=12, color=PALETTE["text"])
    ax1.text(5.0, 8.3, "Qwen 3.5-2B (LoRA GRPO)\nActive View-Graph Orchestrator", ha="center", va="center", fontsize=9.5, color=PALETTE["muted"])

    # 4 Tool Categories
    tools = [
        {
            "tier": "Tier 1: Retrieval",
            "models": "DINOv2-small",
            "tools": 'retrieve\nverify_doppelganger',
            "purpose": "Global visual embeddings\nTopology & cycle verification",
            "color": PALETTE["retrieval"],
            "box": (0.15, 3.4, 2.25, 3.5)
        },
        {
            "tier": "Tier 2: Matchers",
            "models": "LoFTR | LightGlue\nMASt3R",
            "tools": 'match\nbuild_initial_graph',
            "purpose": "2D/3D correspondences\nRANSAC essential matrix",
            "color": PALETTE["matchers"],
            "box": (2.65, 3.4, 2.25, 3.5)
        },
        {
            "tier": "Tier 3: Pose Priors",
            "models": "VGGT-1B | Pi3X\nMapAnything",
            "tools": 'pose_prior\ninspect_graph',
            "purpose": "1-pass feedforward poses\nEdge confidence (no BA)",
            "color": PALETTE["priors"],
            "box": (5.15, 3.4, 2.25, 3.5)
        },
        {
            "tier": "Tier 4: Backends",
            "models": "COLMAP | GLOMAP\nMASt3R-SfM",
            "tools": 'sfm_run\n(final BA engine)',
            "purpose": "Incremental / Global BA\nFinal 3D reconstruction",
            "color": PALETTE["reconstruction"],
            "box": (7.65, 3.4, 2.20, 3.5)
        }
    ]

    for t in tools:
        bx, by, bw, bh = t["box"]
        card = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.15",
                                      edgecolor=t["color"], facecolor="#FFFFFF", linewidth=2.0)
        ax1.add_patch(card)
        # Header banner
        header = patches.Rectangle((bx, by + bh - 0.75), bw, 0.75, facecolor=t["color"])
        ax1.add_patch(header)
        ax1.text(bx + bw/2, by + bh - 0.38, t["tier"], ha="center", va="center", weight="bold", color="white", fontsize=9.5)

        # Content blocks
        ax1.text(bx + bw/2, by + bh - 1.25, t["models"], ha="center", va="center", weight="bold", color=PALETTE["text"], fontsize=9.0)
        
        tool_tag = patches.FancyBboxPatch((bx + 0.15, by + bh - 2.3), bw - 0.3, 0.85, boxstyle="round,pad=0.08",
                                         facecolor="#F1F5F9", edgecolor=t["color"], linewidth=1.0)
        ax1.add_patch(tool_tag)
        ax1.text(bx + bw/2, by + bh - 1.88, f"Tool Calls:\n{t['tools']}", ha="center", va="center", style="italic", color=t["color"], fontsize=8.0, weight="bold")

        ax1.text(bx + bw/2, by + 0.5, t["purpose"], ha="center", va="center", color=PALETTE["muted"], fontsize=8.0)

        # Connection arrow
        ax1.annotate("", xy=(bx + bw/2, by + bh), xytext=(5.0, 7.8),
                     arrowprops=dict(arrowstyle="->", color=t["color"], lw=2.0, shrinkA=3, shrinkB=3))

    # Bottom Architectural Rule
    rule_box = patches.FancyBboxPatch((0.3, 0.4), 9.4, 2.3, boxstyle="round,pad=0.2",
                                      edgecolor="#CBD5E1", facecolor="#F8FAFC", linewidth=1.5)
    ax1.add_patch(rule_box)
    ax1.text(5.0, 2.3, "CRITICAL ARCHITECTURAL CONTRACT (§4.1):", ha="center", va="center", weight="bold", color="#B91C1C", fontsize=10.0)
    ax1.text(5.0, 1.45,
             "1. NEVER feed ground-truth camera poses into policy input (policy would see evaluation target).\n"
             "2. Pose-prior models (VGGT / Pi3X) return relative pose + confidence as OBSERVATIONS for decision-making.\n"
             "3. Feed-forward poses DO NOT replace final BA; COLMAP/GLOMAP always refines chosen pairs in sfm_run.",
             ha="center", va="center", fontsize=8.8, color=PALETTE["text"])


    # =========================================================================
    # Panel 2: Measured Pose Accuracy on 30-Scene Frozen ScanNet Benchmark
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("B. Measured Benchmark on 30 Frozen ScanNet Scenes (1,170 Pairs)", loc="left", pad=12)

    categories = ["AUC @ 5°", "AUC @ 10°", "AUC @ 20°"]
    x = np.arange(len(categories))
    width = 0.25

    loftr_scores = [0.143, 0.260, 0.353]
    vggt_ransac_scores = [0.059, 0.099, 0.123]
    vggt_head_scores = [0.372, 0.631, 0.772]

    rects1 = ax2.bar(x - width, loftr_scores, width, label="LoFTR (RANSAC Essential Matrix)", color=PALETTE["matchers"], alpha=0.9)
    rects2 = ax2.bar(x, vggt_ransac_scores, width, label="VGGT (RANSAC on Correspondences)", color="#FDBA74", alpha=0.95)
    rects3 = ax2.bar(x + width, vggt_head_scores, width, label="VGGT (Extrinsics Head: prior_pose)", color=PALETTE["priors"], alpha=0.95)

    ax2.set_ylabel("Relative Pose AUC (Higher = Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, weight="bold")
    ax2.set_ylim(0, 0.95)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f"{height:.1%}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, weight="bold")

    # Callout text with median angular errors
    ax2.text(0.03, 0.86, "Median Angular Error:\n• LoFTR RANSAC: 69.4°\n• VGGT RANSAC: 11.9° (862 pairs starved)\n• VGGT Prior Head: 6.8° (1,170/1,170 pairs)",
             transform=ax2.transAxes, fontsize=8.8, weight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.5))

    legend2 = ax2.legend(loc="upper right", frameon=True, framealpha=0.95, prop={"weight": "bold", "size": 8.5})


    # =========================================================================
    # Panel 3: Inference Speed vs Geometry Capability Tradeoff
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("C. Latency vs Geometric Capability Tradeoff for Candidate Tools", loc="left", pad=12)

    # Models list with label placement offsets (dx, dy)
    models = [
        ("DINOv2-small\n(retrieval)", 15, 2.5, PALETTE["retrieval"], 85, (0, -0.7)),
        ("LightGlue\n(crop-retry)", 35, 4.5, PALETTE["matchers"], 250, (0, 0.45)),
        ("LoFTR\n(outdoor)", 90, 6.0, PALETTE["matchers"], 750, (0, 0.45)),
        ("VGGT-1B\n(feedforward prior)", 180, 8.3, PALETTE["priors"], 3800, (-40, 0.55)),
        ("Pi3X\n(perm-equiv)", 240, 7.6, "#A21CAF", 3200, (30, -0.65)),
        ("MASt3R\n(3D pointmaps)", 400, 8.7, "#C2410C", 4200, (-60, 0.55)),
        ("MapAnything\n(multimodal 3D)", 650, 9.2, "#6D28D9", 6500, (0, 0.45)),
        ("GLOMAP\n(global BA backend)", 920, 9.4, PALETTE["reconstruction"], 1200, (60, -0.6)),
        ("Incremental COLMAP\n(standard BA backend)", 1550, 9.7, "#047857", 1800, (0, 0.4)),
    ]

    for name, lat, cap, col, mem, (dx, dy) in models:
        size = np.sqrt(mem) * 2.2
        ax3.scatter(lat, cap, s=size, color=col, alpha=0.85, edgecolors="#0F172A", linewidth=1.2)
        ax3.annotate(name, xy=(lat, cap), xytext=(dx, dy*30), textcoords="offset points",
                     ha="center", va="center", fontsize=8.0, weight="bold", color=PALETTE["text"],
                     arrowprops=dict(arrowstyle="-", color=PALETTE["muted"], lw=0.8, alpha=0.7) if abs(dx)+abs(dy)>0.5 else None)

    ax3.set_xscale("log")
    ax3.set_xlabel("Inference Latency per Call (ms, log scale)")
    ax3.set_ylabel("Geometric Output Power (1: Retrieval → 10: Full BA)")
    ax3.set_xlim(8, 3500)
    ax3.set_ylim(1.3, 10.8)
    ax3.grid(True, linestyle="--", alpha=0.3)

    ax3.text(0.04, 0.05, "Bubble area proportional to GPU VRAM footprint (85 MB → 6.5 GB)",
             transform=ax3.transAxes, fontsize=8.5, color=PALETTE["muted"], style="italic")


    # =========================================================================
    # Panel 4: Model Matrix
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("D. Recommended Role & Deployment Strategy in Agentic-SfM", loc="left", pad=12)
    ax4.axis("off")

    columns = ["Model", "Tool Name", "Output Modality", "Role in Agentic-SfM", "Deployment Strategy"]
    table_data = [
        ["DINOv2", "retrieve", "Embedding (384d)", "Candidate edge proposal & k-NN graph", "Resident on tool GPU (fast, 85MB)"],
        ["LoFTR", "match", "Dense 2D matches", "Primary pair correspondence generator", "Resident on tool GPU (GPU 2)"],
        ["LightGlue", "match (retry)", "Sparse keypoint matches", "Fallback for cropped / high-zoom pairs", "Loaded on demand in tools_server"],
        ["VGGT-1B", "pose_prior", "Extrinsics + Depth/Pts", "Provide 6.8° relative pose prior to agent", "Host on GPU 2; evaluate threshold 3.0"],
        ["MASt3R", "match / mast3r", "3D pointmaps + matches", "High-accuracy candidate for tough views", "Now installed; needs clean bench"],
        ["GLOMAP", "sfm_run (alt)", "Global poses + BA", "Fast Global SfM alternative to COLMAP", "Runs via C++ subprocess on CPU/GPU"],
        ["COLMAP", "sfm_run (main)", "Sparse 3D pts + cameras", "Incremental BA engine on verified graph", "Runs on $TMP_SHARED via pycolmap"],
    ]

    col_widths = [0.15, 0.16, 0.22, 0.27, 0.25]
    table = ax4.table(cellText=table_data, colLabels=columns, colWidths=col_widths, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.85)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#E2E8F0")
        if i == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold", fontsize=8.5)
        else:
            if i % 2 == 1:
                cell.set_facecolor("#F8FAFC")
            else:
                cell.set_facecolor("#FFFFFF")
            if j == 0:
                cell.set_text_props(weight="bold")

    fig.suptitle("SfM Model Harnessing Landscape for Agentic-SfM (Phase 2)", fontsize=16, y=0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    out_dir = "/scratch/kcwp264/agentic-sfm/results/figures"
    out_file = os.path.join(out_dir, "sfm_models_tool_harness.png")
    create_sfm_tools_figure(out_file)
