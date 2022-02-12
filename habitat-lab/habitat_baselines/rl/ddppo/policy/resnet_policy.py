#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict, deque
from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import Config
from habitat.tasks.nav.nav import (
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
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.models.z_model import *
from habitat_baselines.rl.ppo import Net, Policy
from habitat_baselines.utils.common import GaussianNet

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
        action_distribution_type: str = "categorical",
        robots: list = ["A1"],
        use_z: bool = False,
        adapt_z_out: float = None,
        z_out_dim: int = 1,
        prev_window: int = 50,
        use_mlp: bool = False,
        z_enc_inputs: list = [],
        **kwargs
    ):
        discrete_actions = action_distribution_type == "categorical"
        self.adapt_z_out = adapt_z_out
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
                robots=robots,
                use_z=use_z,
                adapt_z_out=adapt_z_out,
                z_out_dim=z_out_dim,
                prev_window=prev_window,
                use_mlp=use_mlp,
                z_enc_inputs=z_enc_inputs,
            ),
            dim_actions=action_space.n,  # for action distribution
            action_distribution_type=action_distribution_type,
        )
        self.action_distribution_type = action_distribution_type

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            action_distribution_type=config.RL.POLICY.action_distribution_type,
            robots=config.TASK_CONFIG.TASK.ROBOTS,
            use_z=config.TASK_CONFIG.TASK.Z.USE_Z,
            adapt_z_out=config.TASK_CONFIG.TASK.Z.ADAPT_Z,
            z_out_dim=config.TASK_CONFIG.TASK.Z.Z_OUT_DIM,
            prev_window=config.TASK_CONFIG.TASK.Z.PREV_WINDOW,
            use_mlp=config.TASK_CONFIG.TASK.Z.USE_MLP,
            z_enc_inputs=config.TASK_CONFIG.TASK.Z.Z_ENC_INPUTS,
        )

    def get_metrics(self):
        def get_mean(deq):
            if not deq:
                return -1
            return np.mean(deq)

        ac_metrics = []
        if self.net.use_z:
            if not self.adapt_z_out:
                z_encodings_dict = {
                    k: get_mean(v) for k, v in self.net.z_deques.items()
                }

                name = "z_encodings"

                z_inputs_dict = {
                    k: v.z_in.clone().detach()
                    for k, v in self.net.z_networks.items()
                }

                z_in_name = "z_inputs"
                ac_metrics = [
                    (name, z_encodings_dict),
                    (z_in_name, z_inputs_dict),
                ]
            else:
                name = "z_encodings"
                value = {"z_out": self.net.adapt_z_out.clone().detach()}
                ac_metrics = [(name, value)]

        return ac_metrics

    def act(self, *args, **kwargs):
        self.net.acting = True
        ret = super().act(*args, **kwargs)
        self.net.acting = False
        return ret

    def to(self, device):
        super().to(device)
        if self.net.use_z:
            if not self.adapt_z_out:
                for z_net in self.net.z_networks.values():
                    z_net.to(device)
            else:
                self.net.adapt_z_out = self.net.adapt_z_out.to(device)


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
        self.depth_keys = [k for k in observation_space.spaces if "depth" in k]
        self.using_one_camera = "depth" in observation_space.spaces
        self.using_two_cameras = any(
            [k.endswith("_depth") for k in observation_space.spaces.keys()]
        )

        self._n_input_rgb, self._n_input_depth = [
            # sum() returns 0 for an empty list
            sum([observation_space.spaces[k].shape[2] for k in keys])
            for keys in [self.rgb_keys, self.depth_keys]
        ]
        if self.using_one_camera or self.using_two_cameras:
            self._n_input_depth = 1

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            all_keys = self.rgb_keys + self.depth_keys
            spatial_size_h = (
                observation_space.spaces[all_keys[0]].shape[0] // 2
            )
            spatial_size_w = (
                observation_space.spaces[all_keys[0]].shape[1] // 2
            )
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
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
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            if self.using_one_camera:
                depth_observations = observations["depth"]
            elif self.using_two_cameras:
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
        robots: list = ["A1"],
        use_z: bool = False,
        adapt_z_out: float = None,
        z_out_dim: int = 1,
        prev_window: int = 50,
        use_mlp: bool = False,
        z_enc_inputs: list = [],
    ):
        super().__init__()

        self.acting = False
        self.z_deques = OrderedDict(
            {
                "a1_z_out": deque(maxlen=DEQUE_LENGTH),
                "aliengo_z_out": deque(maxlen=DEQUE_LENGTH),
                "locobot_z_out": deque(maxlen=DEQUE_LENGTH),
            }
        )
        self.discrete_actions = discrete_actions
        self.z_network = None
        self.use_z = use_z
        self.adapt_z_out = adapt_z_out
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.n, 32)

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
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
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
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
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        if not self.visual_encoder.is_blind:
            ## 4069 = 256 * 4 * 4 [for 256 x 256 imgs]
            ## 4560 = 228 * 4 * 5 [for 320 x 240 imgs]
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(4096, hidden_size),
                nn.Linear(4560, hidden_size),
                # np.prod(self.visual_encoder.output_shape), hidden_size
                nn.ReLU(True),
            )

        self.use_mlp = use_mlp
        self.z_enc = z_enc_inputs

        print("USING Z: ", self.use_z)

        if self.use_z:
            print("Z ENC: ", self.z_enc, "Z OUT: ", self.adapt_z_out)
            z_in_dim = 1
            rnn_input_size += z_out_dim

            if not self.adapt_z_out:
                num_inputs = z_in_dim
                # 3 ROBOTS, 50 PREV STATES, 50 PREV ACTIONS, 4 URDF PARAMS, 1 LEARNABLE PARAMETER
                if "urdf" in self.z_enc:
                    num_inputs += 4
                if "prev_states" in self.z_enc:
                    num_inputs += 3 * prev_window
                if "prev_actions" in self.z_enc:
                    num_inputs += 3 * prev_window
                # num_inputs = (3 * num_prev * 2) + 4 + z_in_dim
                if self.use_mlp:
                    net_cls = ZEncoderNet
                else:
                    net_cls = ZVarEncoderNet
                self.z_networks = {
                    k: net_cls(num_inputs, z_out_dim)
                    for k in ["a1", "aliengo", "locobot"]
                }
                # self.z_network = GaussianNet(num_inputs, 1)
            else:
                self.adapt_z_out = torch.tensor(self.adapt_z_out)

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
                visual_feats = self.visual_encoder(observations)
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
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )
            x.append(self.tgt_embeding(goal_observations))

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
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

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

        if self.use_z:
            num_envs = observations["prev_states"].shape[0]
            if not self.adapt_z_out:
                z_in = []
                use_params = self.z_enc != []

                if "urdf" in self.z_enc:
                    z_in.append(observations["urdf_params"])
                if "prev_states" in self.z_enc:
                    prev_states_enc = torch.stack(
                        [
                            observations["prev_states"][:, :, 0],
                            torch.cos(-observations["prev_states"][:, :, 1]),
                            torch.sin(-observations["prev_states"][:, :, 1]),
                        ],
                        -1,
                    )
                    z_in.append(prev_states_enc.reshape(num_envs, -1))
                if "prev_actions" in self.z_enc:
                    z_in.append(
                        observations["prev_actions"].reshape(num_envs, -1)
                    )

                if z_in == []:
                    z_in = [torch.zeros(num_envs, 1)]
                z_concat = torch.cat(z_in, dim=1)

                # z_concat = torch.cat([
                #                       observations['urdf_params'],
                #                       prev_states_enc.reshape(num_envs, -1),
                #                       observations['prev_actions'].reshape(num_envs, -1)
                #                      ],
                #                      dim=1)

                robot_id_mask = observations["robot_id"].squeeze(-1)
                # robot_id_mask = observations['robot_id']
                # z_inputs = [
                #     z_concat[robot_id_mask == z_idx]
                #     for z_idx in range(3)
                # ]
                z_inputs = OrderedDict(
                    {
                        k: z_concat[robot_id_mask == z_idx]
                        for z_idx, k in enumerate(self.z_networks.keys())
                    }
                )
                z_inputs_data = list(z_inputs.values())

                if self.use_mlp:
                    z_outputs = OrderedDict(
                        {
                            z_name: z_net(
                                z_inputs[z_name], use_params=use_params
                            )
                            for z_name, z_net in self.z_networks.items()
                            if z_inputs[z_name].nelement() != 0
                        }
                    )
                else:
                    z_outputs = OrderedDict(
                        {
                            z_name: z_net(
                                z_inputs[z_name], use_params=use_params
                            ).sample()
                            for z_name, z_net in self.z_networks.items()
                            if z_inputs[z_name].nelement() != 0
                        }
                    )

                for k, v in z_outputs.items():
                    current_z_vals = [i.clone().item() for i in v.squeeze(-1)]
                    self.z_deques[k + "_z_out"].extend(current_z_vals)

                z_outputs = torch.cat(list(z_outputs.values()), dim=0)
            else:
                z_outputs = self.adapt_z_out.repeat(num_envs, 1)
            x.append(z_outputs)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )
        return out, rnn_hidden_states
