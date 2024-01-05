#!/usr/bin/env python3

import sys
sys.path.append('/Users/weizhenliu/VScodeProjects/legged_nav')

import contextlib
import json
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from habitat_baselines.common.tensor_dict import TensorDict

from habitat import Config, VectorEnv, logger
from habitat.core.spaces import ActionSpace, EmptySpace
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch, apply_obs_transforms_obs_space,
    get_active_obs_transforms)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (EXIT, add_signal_handlers,
                                                  get_distrib_size,
                                                  init_distrib_slurm,
                                                  is_slurm_batch_job,
                                                  load_resume_state,
                                                  rank0_only, requeue_job,
                                                  save_resume_state)
from habitat_baselines.rl.ddppo.policy import \
    PointNavResNetPolicy  # noqa: F401.
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (ObservationBatchingCache,
                                            action_to_velocity_control,
                                            batch_obs, generate_video)
from habitat_baselines.utils.env_utils import construct_envs
from gym import spaces
import cv2

from spot_wrapper.spot import Spot
from spot_wrapper import deploy_spot

import argparse
import time
from collections import deque

import cv2
import numpy as np
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    draw_crosshair,
    image_response_to_cv2,
    scale_depth_img,
)
from spot_wrapper.utils.utils import color_bbox, resize_to_tallest
from spot_wrapper import depth_filter

@baseline_registry.register_trainer(name="ddppo")
@baseline_registry.register_trainer(name="ppo")

class Deploy(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    agent: PPO
    actor_critic: Policy

    def __init__(self, checkpoint_path=None, spot=None, target=None, save_dir=None):


        self.spot = spot
        self.spot.power_on()
        self.spot.stand(timeout_sec=None)
        time.sleep(5)
        self.target = target
        self.checkpoint_path = checkpoint_path
        ckpt_dict = self.load_checkpoint(map_location="cpu")
        print(f"Loading {ckpt_dict['extra_state']['step']/1000000:.2f}M step checkpoint")
        self.config = ckpt_dict["config"]
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        self.discrete_actions = (
            self.config.TASK_CONFIG.TASK.ACTIONS.VELOCITY_CONTROL.DISCRETE_ACTIONS
        )
        self.action_type = ckpt_dict["config"].TASK_CONFIG.TASK.POSSIBLE_ACTIONS[0]
        self.policy_action_space = ActionSpace({'angular_velocity':spaces.Box(-30.0, 30.0, (1,), float), 'horizontal_velocity':spaces.Box(-0.5, 0.5, (1,), float), 'linear_velocity':spaces.Box(-0.5, 0.5, (1,), float)})
        self.observation_space = spaces.Dict({'depth': spaces.Box(0.0, 1.0, (240, 320, 1), float), 'pointgoal_with_gps_compass': spaces.Box(-3.4028235e+38, 3.4028235e+38, (2,), float)})
        
        self.device = 'cpu'
        self.step_num = 0
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        super().__init__(ckpt_dict["config"])
    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.observation_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        # hack to prevent training with RGB; but still be able to evaluate / generate videos with RGB
        if "rgb" in observation_space.spaces:
            del observation_space.spaces["rgb"]

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def load_checkpoint(self, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(self.checkpoint_path, *args, **kwargs)

    def deploy(
        self,
    ) -> None:
        
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(map_location="cpu")

        config = ckpt_dict["config"]

        ppo_cfg = config.RL.PPO
        
        action_shape = 3
        action_type = torch.float
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        img, rgb, img_raw, goal, done, self.step_num = deploy_spot.command_spot(self.spot, goal_position=self.target, step_num=self.step_num, return_rgb=True)
        test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            1,
            action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            1,
            1,
            device=self.device,
            dtype=torch.bool,
        )

        self.actor_critic.eval()

        batch = self.to_tensor(img, goal)
        self.show_obs(img, rgb, destroy=True)
        while not done:

            with torch.no_grad():
                (_, actions, _, test_recurrent_hidden_states,) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    # deterministic=True,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            # import pdb; pdb.set_trace()
            image = batch['depth'].squeeze().numpy() * 255
            cv2.imwrite(os.path.join(self.save_dir, f'{self.step_num}_depth.jpg'), image)
            cv2.imwrite(os.path.join(self.save_dir, f'{self.step_num}_grey.jpg'), rgb)
            np.save(os.path.join(self.save_dir, f'{self.step_num}_raw.npy'), img_raw)

            if self.config.RL.POLICY.action_distribution_type == "gaussian":
                step_data = [
                    action_to_velocity_control(a, self.action_type)
                    for a in actions.to(device="cpu")
                ]
            elif len(self.discrete_actions) > 0:
                step_data = [
                    action_to_velocity_control(
                        torch.tensor(
                            self.discrete_actions[a.item()],
                            device="cpu",
                        ),
                        self.action_type,
                    )
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]
            
            self.show_obs(img, rgb, destroy=False, done=done)
            img, rgb, img_raw, goal, done, self.step_num = deploy_spot.command_spot(self.spot, step_data, done, goal_position=self.target, step_num=self.step_num, return_rgb=True)
            batch = self.to_tensor(img, goal)
        self.show_obs(img, rgb, destroy=False, done=done)
            

    def to_tensor(self, img, goal):
        img = torch.tensor(img, dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(-1)
        goal = torch.tensor(goal, dtype=torch.float32, device='cpu').unsqueeze(0)
        # noise = torch.rand_like(img)
        return TensorDict({'depth':img, 'pointgoal_with_gps_compass': goal})

    def show_obs(self, depth_image, rgb_image, destroy=False, done=False):
        depth_image = cv2.cvtColor((255.0 * depth_image).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # Resize the depth image to have the same height as the RGB image
        scaling_factor = rgb_image.shape[0] / depth_image.shape[0]
        depth_resized = cv2.resize(depth_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Make sure both images are 3-channel for visualization (optional)
        # rgb_colored = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)

        # Concatenate images horizontally
        concatenated_image = np.hstack((rgb_image, depth_resized))

        # Add dynamic step information on the top-left corner
        if done:
            height, width, _ = concatenated_image.shape
            (text_width, text_height), _ = cv2.getTextSize("Success!", cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            startX = int((width - text_width) / 2)
            startY = int((height + text_height) / 2)
            # cv2.putText(concatenated_image, f"Step: {self.step_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(concatenated_image, "Success!", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imwrite("/Users/weizhenliu/VScodeProjects/legged_nav/assis/captured_image.jpg", rgb_image)
        else:
            cv2.putText(concatenated_image, f"Step: {self.step_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the concatenated image
        cv2.imshow('RGB and Depth', concatenated_image)
        cv2.waitKey(1)
        # if destroy:
        #     cv2.destroyAllWindows()



if __name__ == "__main__":
    spot = Spot("Walk")

    with spot.get_lease() as lease:
        run_spot = Deploy(spot=spot, 
                    checkpoint_path="/Users/weizhenliu/Downloads/official_depth/ckpt.20.pth",
                    target=np.array([-3, 0, 3]), # right, up, back
                    save_dir = "/Users/weizhenliu/VScodeProjects/legged_nav/habitat-lab/deploy_results/back/save_raw_depth/5max/005") 
        run_spot.deploy()
