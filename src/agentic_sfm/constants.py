"""Shared defaults.

Policy is the small open-weight Qwen3-VL VLM: ``Qwen/Qwen3-VL-2B-Instruct``
(https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct, Apache 2.0). Qwen3-VL is a
standard attention + MLP transformer (model_type ``qwen3_vl``) — no Gated
DeltaNet modules, unlike the Qwen3.5 line.

The Instruct variant has no thinking mode, so ``enable_thinking=False`` is a
no-op (kept in the chat-template call with a try/except fallback). Needs
Transformers >= 5.0 and vLLM >= 0.17.
"""

from __future__ import annotations

DEFAULT_POLICY_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MATCHER = "loftr"
MIN_TRANSFORMERS = (5, 0)
MIN_VLLM = (0, 17)

# Qwen3-VL text backbone is a standard transformer (attention + MLP).
# No Gated DeltaNet modules (those were Qwen3.5-specific).
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
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
)


def _version_tuple(version: str) -> tuple[int, int]:
    parts = version.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def assert_qwen35_runtime() -> None:
    """Fail fast if the env cannot load the policy VLM.

    Name kept for backward compatibility with existing slurm scripts; the check
    now validates the Qwen3-VL (``qwen3_vl``) runtime instead of Qwen3.5.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        tf_ver = version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError("transformers is not installed; policy needs >= 5.0") from exc
    try:
        vllm_ver = version("vllm")
    except PackageNotFoundError:
        vllm_ver = None

    if _version_tuple(tf_ver) < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"Policy needs transformers>={MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]} "
            f"(qwen3_vl), found {tf_ver}. Upgrade the agentic-sfm env before training."
        )
    if vllm_ver is not None and _version_tuple(vllm_ver) < MIN_VLLM:
        raise RuntimeError(
            f"Policy needs vllm>={MIN_VLLM[0]}.{MIN_VLLM[1]} "
            f"(qwen3_vl), found {vllm_ver}. Upgrade the agentic-sfm env before rollout."
        )
