#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict, deque
from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from habitat.config import Config
from habitat.tasks.nav.nav import (
    ContextMapSensor,
    ContextWaypointSensor,
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN, SimpleCNNContext
from habitat_baselines.rl.ppo import Net, Policy
from torch import nn as nn
from torch.nn import functional as F

DEQUE_LENGTH = 150


@baseline_registry.register_policy
class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: None = None,
        num_cnns: int = 1,
        **kwargs,
    ):
        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = policy_config.action_distribution_type
        discrete_actions = self.action_distribution_type == "categorical"
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                num_cnns=num_cnns,
            ),
            dim_actions=action_space.n,  # for action distribution
            policy_config=policy_config,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            num_cnns=config.RL.POLICY.num_cnns,
        )


@baseline_registry.register_policy
class PointNavResNetContextPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: None = None,
        num_cnns: int = 1,
        tgt_hidden_size: int = 32,
        tgt_encoding: str = "linear_2",
        context_hidden_size: int = 512,
        use_prev_action: bool = True,
        cnn_type: str = "resnet",
        **kwargs,
    ):
        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = policy_config.action_distribution_type
        discrete_actions = self.action_distribution_type == "categorical"

        super().__init__(
            PointNavResNetContextNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                num_cnns=num_cnns,
                tgt_hidden_size=tgt_hidden_size,
                tgt_encoding=tgt_encoding,
                context_hidden_size=context_hidden_size,
                use_prev_action=use_prev_action,
                cnn_type=cnn_type,
            ),
            dim_actions=action_space.n,  # for action distribution
            policy_config=policy_config,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            num_cnns=config.RL.POLICY.num_cnns,
            tgt_hidden_size=config.RL.PPO.tgt_hidden_size,
            tgt_encoding=config.RL.PPO.tgt_encoding,
            context_hidden_size=config.RL.PPO.context_hidden_size,
            use_prev_action=config.RL.PPO.use_prev_action,
            cnn_type=config.RL.PPO.cnn_type,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        self.rgb_keys = [k for k in observation_space.spaces if "rgb" in k]
        self.gray_keys = [k for k in observation_space.spaces if "gray" in k]
        self.depth_keys = [k for k in observation_space.spaces if "depth" in k]

        self.using_one_gray_camera = len(self.gray_keys) == 1
        self.using_two_gray_cameras = len(self.gray_keys) == 2

        self.using_one_depth_camera = len(self.depth_keys) == 1
        self.using_two_depth_cameras = len(self.depth_keys) == 2

        self._n_input_rgb, self._n_input_depth, self._n_input_gray = [
            # sum() returns 0 for an empty list
            sum([observation_space.spaces[k].shape[2] for k in keys])
            for keys in [self.rgb_keys, self.depth_keys, self.gray_keys]
        ]
        if self.using_one_depth_camera or self.using_two_depth_cameras:
            self._n_input_depth = 1
        if self.using_one_gray_camera or self.using_two_gray_cameras:
            self._n_input_gray = 1

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            all_keys = self.rgb_keys + self.depth_keys + self.gray_keys
            spatial_size_h = observation_space.spaces[all_keys[0]].shape[0] // 2
            spatial_size_w = observation_space.spaces[all_keys[0]].shape[1] // 2
            if self.using_two_depth_cameras or self.using_two_gray_cameras:
                spatial_size_w = observation_space.spaces[all_keys[0]].shape[1]
            input_channels = (
                self._n_input_depth + self._n_input_rgb + self._n_input_gray
            )
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial_h = int(spatial_size_h * self.backbone.final_spatial_compress)
            final_spatial_w = int(spatial_size_w * self.backbone.final_spatial_compress)

            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial_h * final_spatial_w))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_gray == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            # if not already normalized
            if torch.max(rgb_observations) > 1.0:
                rgb_observations = rgb_observations.float() / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_gray > 0:
            if self.using_one_gray_camera:
                gray_observations = observations[self.gray_keys[0]]
            elif self.using_two_gray_cameras:
                gray_observations = torch.cat(
                    [
                        # Spot is cross-eyed; right is on the left on the FOV
                        observations["spot_right_gray"],
                        observations["spot_left_gray"],
                    ],
                    dim=2,
                )
            else:
                raise Exception("Not implemented")
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            gray_observations = gray_observations.permute(0, 3, 1, 2)
            # if not already normalized
            if torch.max(gray_observations) > 1.0:
                gray_observations = gray_observations.float() / 255.0  # normalize RGB
            cnn_input.append(gray_observations)

        if self._n_input_depth > 0:
            if self.using_one_depth_camera:
                depth_observations = observations[self.depth_keys[0]]
            elif self.using_two_depth_cameras:
                depth_observations = torch.cat(
                    [
                        # Spot is cross-eyed; right is on the left on the FOV
                        observations["spot_right_depth"],
                        observations["spot_left_depth"],
                    ],
                    dim=2,
                )
            else:
                raise Exception("Not implemented")

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class ResNetEncoderContext(ResNetEncoder):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            baseplanes=baseplanes,
            ngroups=ngroups,
            spatial_size=spatial_size,
            make_backbone=make_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
        )
        self._n_input_map = 1
        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(self._n_input_map)
        else:
            self.running_mean_and_var = nn.Sequential()

        spatial_size_h = observation_space.spaces["context_map"].shape[0] // 2
        spatial_size_w = observation_space.spaces["context_map"].shape[1] // 2

        input_channels = 2
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)
        final_spatial_h = int(
            np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
        )
        final_spatial_w = int(
            np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
        )

        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial_h * final_spatial_w))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial_h,
            final_spatial_w,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        observations = observations.permute(0, 3, 1, 2)
        x = F.avg_pool2d(observations, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        num_cnns: int = 1,
    ):
        super().__init__()

        self.discrete_actions = discrete_actions
        self.num_cnns = num_cnns
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.n, 32)

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            self._n_input_goal = observation_space.spaces[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ].shape[0]
            tgt_embeding_size = 32
            self.tgt_encoder = nn.Linear(self._n_input_goal + 1, tgt_embeding_size)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        if ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = ResNetEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )
            self.goal_visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.goal_visual_encoder.output_shape), hidden_size),
                nn.ReLU(True),
            )

            rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        print("NUM CNNS: ", self.num_cnns)
        if self.num_cnns == 1:
            self.visual_encoder = ResNetEncoder(
                observation_space if not force_blind_policy else spaces.Dict({}),
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )
        elif self.num_cnns == 2:
            left_obs_space, right_obs_space = [
                spaces.Dict(
                    {k: v for k, v in observation_space.spaces.items() if side not in k}
                )
                for side in ["right", "left"]
            ]
            # Left CNN
            self.visual_encoder = ResNetEncoder(
                left_obs_space if not force_blind_policy else spaces.Dict({}),
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )
            # Right CNN
            self.visual_encoder2 = ResNetEncoder(
                right_obs_space if not force_blind_policy else spaces.Dict({}),
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )
        elif self.num_cnns == 0:
            print("VISUAL ENCODER IS SIMPLE CNN")
            self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        if self.num_cnns != 0:
            dim = np.prod(self.visual_encoder.output_shape)
            if not self.visual_encoder.is_blind:
                if self.num_cnns == 2:
                    dim *= 2
                self.visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(dim, hidden_size),
                    nn.ReLU(True),
                )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            if "visual_features" in observations:
                visual_feats = observations["visual_features"]
            else:
                if self.num_cnns == 1:
                    visual_feats = self.visual_encoder(observations)
                elif self.num_cnns == 2:
                    left_visual_feats = self.visual_encoder(observations)
                    right_visual_feats = self.visual_encoder2(observations)
                    visual_feats = torch.cat(
                        [right_visual_feats, left_visual_feats], dim=1
                    )
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert goal_observations.shape[1] == 3, "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )
            x.append(self.tgt_encoder(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        if ImageGoalSensor.cls_uuid in observations:
            goal_image = observations[ImageGoalSensor.cls_uuid]
            goal_output = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)
        return out, rnn_hidden_states


class PointNavResNetContextNet(PointNavResNetNet):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. + Map
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        num_cnns: int = 1,
        tgt_hidden_size: int = 32,
        tgt_encoding: str = "linear_2",
        context_hidden_size: int = 512,
        use_prev_action: bool = True,
        cnn_type: str = "resnet",
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            backbone=backbone,
            resnet_baseplanes=resnet_baseplanes,
            normalize_visual_inputs=normalize_visual_inputs,
            force_blind_policy=force_blind_policy,
            discrete_actions=discrete_actions,
            num_cnns=num_cnns,
        )
        self.cnn_type = cnn_type
        self.use_prev_action = use_prev_action
        prev_action_embedding_size = 32 if self.use_prev_action else 0
        self.tgt_encoding = tgt_encoding
        self.tgt_embeding_size = tgt_hidden_size

        if self.tgt_encoding == "sin_cos":
            self.tgt_encoder = nn.Linear(self._n_input_goal + 1, self.tgt_embeding_size)
        elif self.tgt_encoding == "linear_1":
            self.tgt_encoder = nn.Sequential(
                nn.Linear(self._n_input_goal, self.tgt_embeding_size),
                nn.ReLU(),
            )
        elif self.tgt_encoding == "linear_2":
            self.tgt_encoder = nn.Sequential(
                nn.Linear(self._n_input_goal, self.tgt_embeding_size),
                nn.ReLU(),
                nn.Linear(self.tgt_embeding_size, self.tgt_embeding_size),
                nn.ReLU(),
            )

        if "context_map" in observation_space.keys():
            self.context_type = "map"
        elif "context_waypoint" in observation_space.keys():
            self.context_type = "waypoint"

        print("CONTEXT TYPE: ", self.context_type)
        self.context_hidden_size = context_hidden_size
        if self.context_type == "map":
            if self.cnn_type == "cnn_ans":
                self.context_encoder = SimpleCNNContext(
                    observation_space,
                    self.context_hidden_size,
                    cnn_type="cnn_ans",
                )
            elif self.cnn_type == "cnn_2d":
                self.context_encoder = SimpleCNNContext(
                    observation_space,
                    self.context_hidden_size,
                    cnn_type="cnn_2d",
                )
            elif self.cnn_type == "resnet":
                self.context_encoder = ResNetEncoderContext(
                    observation_space,
                    baseplanes=resnet_baseplanes,
                    ngroups=resnet_baseplanes // 2,
                    make_backbone=getattr(resnet, backbone),
                    normalize_visual_inputs=normalize_visual_inputs,
                )
                dim = np.prod(self.context_encoder.output_shape)
                self.context_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(dim, self.context_hidden_size),
                    nn.ReLU(True),
                )
        else:
            self.context_encoder = nn.Sequential(
                nn.Linear(2, self.context_hidden_size),
                nn.ReLU(),
                nn.Linear(self.context_hidden_size, self.context_hidden_size),
                nn.ReLU(),
            )

        self.state_encoder = build_rnn_state_encoder(
            self._hidden_size
            + prev_action_embedding_size
            + self.tgt_embeding_size
            + self.context_hidden_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        print(f"##### USING CONTEXT, HIDDEN SIZE: {self.context_hidden_size} #####")
        self.train()

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            if self.num_cnns == 0:
                visual_feats = self.visual_encoder(observations)
            else:
                visual_feats = self.visual_fc(self.visual_encoder(observations))
            x.append(visual_feats)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if self.tgt_encoding == "sin_cos" and goal_observations.shape[1] == 2:
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            goal_feats = self.tgt_encoder(goal_observations)
            x.append(goal_feats)
        if (
            ContextWaypointSensor.cls_uuid in observations
            or ContextMapSensor.cls_uuid in observations
        ):
            if self.context_type == "waypoint":
                context_feats = self.context_encoder(
                    observations[ContextWaypointSensor.cls_uuid]
                )
            elif self.context_type == "map":
                context_feats = self.context_encoder(
                    observations[ContextMapSensor.cls_uuid]
                )
            if self.cnn_type == "resnet":
                context_feats = self.context_fc(context_feats)
            x.append(context_feats)

        if self.use_prev_action:
            prev_actions = self.prev_action_embedding(prev_actions.float())
            x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(out, rnn_hidden_states, masks)
        return out, rnn_hidden_states
