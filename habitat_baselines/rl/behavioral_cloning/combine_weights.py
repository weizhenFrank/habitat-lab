import os
import sys
import time

import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy,
)
from habitat_baselines.rl.ppo.policy import (
    PointNavBaselinePolicy,
    PointNavContextGoalPolicy,
    PointNavContextPolicy,
)

COMBINED_PTH = "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_weights/planning/combined"


class WaypointTeacher:
    def __init__(self, rl_cfg):
        # Assume just 1 GPU from slurm
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


class MapPretrainedStudent:
    def __init__(self, config, planner_ckpt):
        # Assume just 1 GPU from slurm
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

        pretrained_state = torch.load(
            config.RL.DDPPO.teacher_pretrained_weights, map_location="cpu"
        )
        prefix = "actor_critic.net.visual_encoder."
        self.actor_critic.net.visual_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        prefix = "actor_critic.net.tgt_encoder."
        self.actor_critic.net.tgt_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )

        prefix = "actor_critic.net.state_encoder."
        self.actor_critic.net.state_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        prefix = "actor_critic.action_distribution."
        self.actor_critic.action_distribution.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        prefix = "actor_critic.critic."
        self.actor_critic.critic.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )

        prefix = "actor_critic.net.context_encoder."
        self.actor_critic.net.waypoint_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )

        print("loading: ", planner_ckpt)
        pretrained_planner = torch.load(
            planner_ckpt,
            map_location="cpu",
        )

        prefix = "encoder."
        # self.actor_critic.net.map_cnn.cnn.load_state_dict(
        #     {
        #         k[len(prefix) :]: v
        #         for k, v in pretrained_planner.items()
        #         if k.startswith(prefix)
        #     }
        # )
        self.actor_critic.net.map_cnn.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_planner.items()
                if k.startswith(prefix)
            }
        )
        prefix = "visual_fc."
        self.actor_critic.net.visual_fc.load_state_dict(
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
        checkpoint = {
            "state_dict": {
                "actor_critic." + k: v
                for k, v in self.actor_critic.state_dict().items()
            },
            "config": config,
        }
        new_ckpt_pth = os.path.join(
            COMBINED_PTH,
            f"combined_weights_{time.time()}.pth",
        )
        torch.save(
            checkpoint,
            new_ckpt_pth,
        )
        print("saved checkpoint: ", new_ckpt_pth)


class MapGoalStudent:
    def __init__(self, config, planner_ckpt):
        # Assume just 1 GPU from slurm
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

        pretrained_state = torch.load(
            config.RL.DDPPO.teacher_pretrained_weights, map_location="cpu"
        )
        prefix = "actor_critic.net.visual_encoder."
        self.actor_critic.net.visual_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        prefix = "actor_critic.net.tgt_encoder."
        self.actor_critic.net.tgt_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )

        prefix = "actor_critic.net.state_encoder."
        self.actor_critic.net.state_encoder.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        prefix = "actor_critic.action_distribution."
        self.actor_critic.action_distribution.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )
        prefix = "actor_critic.critic."
        self.actor_critic.critic.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_state["state_dict"].items()
                if k.startswith(prefix)
            }
        )

        print("loading: ", planner_ckpt)
        pretrained_planner = torch.load(
            planner_ckpt,
            map_location="cpu",
        )

        prefix = "encoder."
        self.actor_critic.net.map_cnn.load_state_dict(
            {
                k[len(prefix) :]: v
                for k, v in pretrained_planner.items()
                if k.startswith(prefix)
            }
        )
        prefix = "visual_fc."
        self.actor_critic.net.visual_fc.load_state_dict(
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
        checkpoint = {
            "state_dict": {
                "actor_critic." + k: v
                for k, v in self.actor_critic.state_dict().items()
            },
            "config": config,
        }
        new_ckpt_pth = os.path.join(
            COMBINED_PTH,
            f"combined_weights_{time.time()}.pth",
        )
        torch.save(
            checkpoint,
            new_ckpt_pth,
        )
        print("saved checkpoint: ", new_ckpt_pth)


if __name__ == "__main__":
    from habitat_baselines.config.default import get_config

    config = get_config(sys.argv[1])
    planner_ckpt = sys.argv[2]

    # config = get_config(
    #     "habitat_baselines/config/pointnav/behavioral_cloning.yaml"
    # )
    # d = WaypointTeacher(config.RL)
    # g = MapPretrainedStudent(config, planner_ckpt)
    g = MapGoalStudent(config, planner_ckpt)
