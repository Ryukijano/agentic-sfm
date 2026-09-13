"""RL modules for agentic SfM training."""
from agentic_sfm.rl.fission import FissionGRPO, FissionConfig
from agentic_sfm.rl.off_policy import OffPolicyReplayBuffer
from agentic_sfm.rl.rc_grpo import (
    inject_reward_condition,
    sample_reward_conditioned_group,
    compute_rc_advantages,
    prepare_rctp_training_data,
    HIGH_REWARD_TOKEN,
    LOW_REWARD_TOKEN,
)
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    run_scene_episode,
)

__all__ = [
    "FissionGRPO",
    "FissionConfig",
    "OffPolicyReplayBuffer",
    "inject_reward_condition",
    "sample_reward_conditioned_group",
    "compute_rc_advantages",
    "prepare_rctp_training_data",
    "HIGH_REWARD_TOKEN",
    "LOW_REWARD_TOKEN",
    "SceneRolloutEpisode",
    "run_scene_episode",
]
