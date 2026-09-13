"""Shared defaults.

Policy is the official Qwen3.5 2B unified VLM: ``Qwen/Qwen3.5-2B``
(https://huggingface.co/Qwen/Qwen3.5-2B, Apache 2.0). There is no
``Qwen3.5-2B-Instruct`` SKU — this checkpoint *is* the post-trained / Instruct
model (chat + tool use). ``Qwen/Qwen3.5-2B-Base`` is pretrain-only and is not
used here.

Architecture (``model_type`` ``qwen3_5``): hybrid 3:1 Gated DeltaNet + gated
attention, plus the Qwen3-VL vision tower. Default inference is non-thinking
(``enable_thinking=False``). Needs Transformers >= 5.2 and vLLM >= 0.17.
"""

from __future__ import annotations

DEFAULT_POLICY_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_MATCHER = "loftr"
MIN_TRANSFORMERS = (5, 2)
MIN_VLLM = (0, 17)

# Qwen3.5 text backbone: gated attention + Gated DeltaNet + MLP.
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

REWARD_TOTAL_KEYS = (
    "format_reward",
    "invalid_penalty",
    "inlier_reward",
    "pose_reward",
    "tool_cost",
    "accumulative_tool_reward",
    # NTEP process rewards (arXiv 2609.03493) — 0 unless use_ntep_rewards=True
    "ntep_intent_reward",
    "ntep_redundancy_penalty",
)


def _version_tuple(version: str) -> tuple[int, int]:
    parts = version.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def assert_qwen35_runtime() -> None:
    """Fail fast if the env cannot load Qwen3.5 (``qwen3_5``)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        tf_ver = version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError("transformers is not installed; policy needs >= 5.2") from exc
    try:
        vllm_ver = version("vllm")
    except PackageNotFoundError:
        vllm_ver = None

    if _version_tuple(tf_ver) < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"Policy needs transformers>={MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]} "
            f"(qwen3_5), found {tf_ver}. Upgrade the agentic-sfm env before training."
        )
    if vllm_ver is not None and _version_tuple(vllm_ver) < MIN_VLLM:
        raise RuntimeError(
            f"Policy needs vllm>={MIN_VLLM[0]}.{MIN_VLLM[1]} "
            f"(qwen3_5), found {vllm_ver}. Upgrade the agentic-sfm env before rollout."
        )
