import os

import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import \
    PointNavResNetPolicy
from habitat_baselines.rl.ppo.policy import (PointNavBaselinePolicy,
                                             PointNavContextGoalPolicy,
                                             PointNavContextPolicy)


def load_pretrained_weights(actor_critic, cfg, load_context_encoder=False):
    # load teacher's pretrained visual encoder
    pretrained_state = torch.load(
        cfg.RL.DDPPO.teacher_pretrained_weights, map_location="cpu"
    )
    prefix = "actor_critic.net.visual_encoder."
    actor_critic.net.visual_encoder.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in pretrained_state["state_dict"].items()
            if k.startswith(prefix)
        }
    )
    prefix = "actor_critic.net.tgt_encoder."
    actor_critic.net.tgt_encoder.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in pretrained_state["state_dict"].items()
            if k.startswith(prefix)
        }
    )

    # if cfg.FREEZE_POLICY:
    prefix = "actor_critic.net.state_encoder."
    actor_critic.net.state_encoder.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in pretrained_state["state_dict"].items()
            if k.startswith(prefix)
        }
    )
    prefix = "actor_critic.action_distribution."
    actor_critic.action_distribution.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in pretrained_state["state_dict"].items()
            if k.startswith(prefix)
        }
    )
    prefix = "actor_critic.critic."
    actor_critic.critic.load_state_dict(
        {
            k[len(prefix) :]: v
            for k, v in pretrained_state["state_dict"].items()
            if k.startswith(prefix)
        }
    )

    if load_context_encoder:
        # key = "context_encoder."
        key = "waypoint_encoder."
        prefix = "actor_critic.net." + key
        actor_critic.net.waypoint_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )

    if not cfg.RL.DDPPO.train_encoder:
        for param in actor_critic.net.visual_encoder.parameters():
            param.requires_grad_(False)
        for param in actor_critic.net.tgt_encoder.parameters():
            param.requires_grad_(False)
    if cfg.FREEZE_POLICY:
        for param in actor_critic.net.state_encoder.parameters():
            param.requires_grad_(False)
        for param in actor_critic.action_distribution.parameters():
            param.requires_grad_(False)
        for param in actor_critic.critic.parameters():
            param.requires_grad_(False)
    if load_context_encoder:
        for param in actor_critic.net.waypoint_encoder.parameters():
            param.requires_grad_(False)
    return actor_critic


class WaypointTeacher:
    def __init__(self, rl_cfg: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))

        observation_space = Dict(
            {
                "spot_left_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "context_waypoint": Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                ),
            }
        )
        action_space = Box(-1.0, 1.0, (2,))
        action_space.n = 2

        self.actor_critic = PointNavContextPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=512,
            num_recurrent_layers=1,
            rnn_type="GRU",
            tgt_hidden_size=512,
            tgt_encoding="linear_2",
            context_hidden_size=512,
            use_prev_action=False,
            policy_config=rl_cfg.POLICY,
            cnn_type="cnn_2d",
        )
        self.actor_critic.to(self.device)

        pretrained_state = torch.load(
            rl_cfg.DDPPO.teacher_pretrained_weights, map_location="cpu"
        )
        self.actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in pretrained_state["state_dict"].items()
            }
        )
        print("LOADED: ", rl_cfg.DDPPO.teacher_pretrained_weights)


class MapStudent:
    def __init__(self, config: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        map_res = config.TASK_CONFIG.TASK.CONTEXT_MAP_SENSOR.MAP_RESOLUTION
        if config.TASK_CONFIG.TASK.CONTEXT_MAP_SENSOR.MULTI_CHANNEL:
            map_shape = (map_res, map_res, 3)
        elif config.TASK_CONFIG.TASK.CONTEXT_MAP_SENSOR.SECOND_CHANNEL:
            map_shape = (map_res, map_res, 2)
        else:
            map_shape = (map_res, map_res, 1)
        print("USING MAP SHAPE: ", map_shape)
        observation_space = Dict(
            {
                "spot_left_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "context_map": Box(
                    low=0.0, high=1.0, shape=map_shape, dtype=np.float32
                ),
            }
        )

        action_space = Box(-1.0, 1.0, (2,))
        action_space.n = 2

        # policy = baseline_registry.get_policy(self.config.RL.POLICY.name)

        self.policy_name = config.RL.POLICY.name
        policy_cls = baseline_registry.get_policy(self.policy_name)
        self.actor_critic = policy_cls.from_config(
            config, observation_space, action_space
        )

        self.actor_critic.to(self.device)
        if config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
            print(
                "USING PRETRAINED WEIGHTS! ",
                config.RL.DDPPO.pretrained_weights,
            )

        else:
            load_context_encoder = False
            if config.RL.PPO.use_waypoint_encoder:
                load_context_encoder = True

            self.actor_critic = load_pretrained_weights(
                self.actor_critic,
                config,
                load_context_encoder=load_context_encoder,
            )


class MapPretrainedStudent:
    def __init__(self, config: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))
        map_res = config.TASK_CONFIG.TASK.CONTEXT_MAP_SENSOR.MAP_RESOLUTION
        if config.TASK_CONFIG.TASK.CONTEXT_MAP_SENSOR.MULTI_CHANNEL:
            map_shape = (map_res, map_res, 3)
        elif config.TASK_CONFIG.TASK.CONTEXT_MAP_SENSOR.SECOND_CHANNEL:
            map_shape = (map_res, map_res, 2)
        else:
            map_shape = (map_res, map_res, 1)
        print("USING MAP SHAPE: ", map_shape)
        observation_space = Dict(
            {
                "spot_left_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "context_map": Box(
                    low=0.0, high=1.0, shape=map_shape, dtype=np.float32
                ),
            }
        )

        action_space = Box(-1.0, 1.0, (2,))
        action_space.n = 2

        # policy = baseline_registry.get_policy(self.config.RL.POLICY.name)

        self.policy_name = config.RL.POLICY.name
        policy_cls = baseline_registry.get_policy(self.policy_name)
        self.actor_critic = policy_cls.from_config(
            config, observation_space, action_space
        )

        self.actor_critic.to(self.device)

        load_context_encoder = True

        self.actor_critic = load_pretrained_weights(
            self.actor_critic,
            config,
            load_context_encoder=load_context_encoder,
        )
        pretrained_planner = torch.load(
            config.RL.DDPPO.pretrained_planner,
            map_location="cpu",
        )

        if load_context_encoder:
            prefix = "encoder."
            self.actor_critic.net.map_cnn.cnn.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_planner.items()
                    if k.startswith(prefix)
                }
            )
            prefix = "mlp."
            self.actor_critic.net.context_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_planner.items()
                    if k.startswith(prefix)
                }
            )
            for param in self.actor_critic.net.map_cnn.parameters():
                param.requires_grad_(False)
            for param in self.actor_critic.net.context_encoder.parameters():
                param.requires_grad_(False)
        #
        # if not cfg.RL.DDPPO.train_encoder:
        #     for param in actor_critic.net.visual_encoder.parameters():
        #         param.requires_grad_(False)


class WaypointStudent:
    def __init__(self, rl_cfg: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))

        observation_space = Dict(
            {
                "spot_left_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "context_waypoint": Box(
                    low=0.0, high=1.0, shape=(2,), dtype=np.float32
                ),
            }
        )
        action_space = Box(-1.0, 1.0, (2,))
        action_space.n = 2

        self.actor_critic = PointNavContextPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=512,
            tgt_hidden_size=512,
            tgt_encoding="linear_2",
            context_hidden_size=512,
            use_prev_action=False,
            policy_config=rl_cfg.POLICY,
            cnn_type="cnn_2d",
        )
        self.actor_critic.to(self.device)
        self.actor_critic = load_pretrained_weights(self.actor_critic, rl_cfg)


class BaselineStudent:
    def __init__(self, rl_cfg: Config) -> None:
        # Assume just 1 GPU from slurm
        self.device = torch.device("cuda", int(os.environ["SLURM_LOCALID"]))

        observation_space = Dict(
            {
                "spot_left_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "spot_right_depth": Box(
                    low=0.0, high=1.0, shape=(256, 128, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Box(-1.0, 1.0, (2,))
        action_space.n = 2

        self.actor_critic = PointNavBaselinePolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=512,
            tgt_hidden_size=512,
            tgt_encoding="linear_2",
            use_prev_action=False,
            policy_config=rl_cfg.POLICY,
        )
        self.actor_critic.to(self.device)

        self.actor_critic = load_pretrained_weights(self.actor_critic, rl_cfg)


if __name__ == "__main__":
    from habitat_baselines.config.default import get_config

    config = get_config(
        "habitat_baselines/config/pointnav/behavioral_cloning.yaml"
    )
    d = WaypointTeacher(config.RL)
    g = MapStudent(config.RL)