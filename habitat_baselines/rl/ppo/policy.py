#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import torch
from gym import spaces
from habitat.config import Config
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import \
    build_rnn_state_encoder
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.utils.common import CategoricalNet, GaussianNet
from torch import nn as nn
from vit_pytorch import SimpleViT, ViT


class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, policy_config=None):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        print("POLICY CONFIG: ", policy_config)
        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = policy_config.action_distribution_type

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.ACTION_DIST,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type} not supported."
            )

        self.critic = CriticHead(self.net.output_size)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


@baseline_registry.register_policy
class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 2,
        rnn_type: str = "LSTM",
        tgt_hidden_size: int = 512,
        tgt_encoding: str = "linear_2",
        use_prev_action: bool = False,
        policy_config: None = None,
        **kwargs,
    ):  
        action_dim = (
            1
            if policy_config.action_distribution_type == "categorical"
            else action_space.n
        )
        action_space.n = action_space.shape[0]
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                tgt_hidden_size=tgt_hidden_size,
                tgt_encoding=tgt_encoding,
                use_prev_action=use_prev_action,
                action_dim=action_dim,
                **kwargs,
            ),
            action_space.n,
            policy_config,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            tgt_hidden_size=config.RL.PPO.tgt_hidden_size,
            tgt_encoding=config.RL.PPO.tgt_encoding,
            use_prev_action=config.RL.PPO.use_prev_action,
            policy_config=config.RL.POLICY,
        )

@baseline_registry.register_policy
class PointNavContextPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 2,
        rnn_type: str = "LSTM",
        tgt_hidden_size: int = 512,
        tgt_encoding: str = "linear_2",
        context_hidden_size: int = 512,
        use_prev_action: bool = False,
        use_waypoint_encoder: bool = False,
        policy_config: None = None,
        cnn_type: str = "cnn_2d",
        **kwargs,
    ):
        action_dim = (
            1
            if policy_config.action_distribution_type == "categorical"
            else action_space.n
        )
        super().__init__(
            PointNavContextNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                tgt_hidden_size=tgt_hidden_size,
                tgt_encoding=tgt_encoding,
                context_hidden_size=context_hidden_size,
                use_prev_action=use_prev_action,
                use_waypoint_encoder=use_waypoint_encoder,
                cnn_type=cnn_type,
                policy_config=policy_config,
                action_dim=action_dim,
                **kwargs,
            ),
            action_space.n,
            policy_config,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            tgt_hidden_size=config.RL.PPO.tgt_hidden_size,
            tgt_encoding=config.RL.PPO.tgt_encoding,
            context_hidden_size=config.RL.PPO.context_hidden_size,
            use_prev_action=config.RL.PPO.use_prev_action,
            use_waypoint_encoder=config.RL.PPO.use_waypoint_encoder,
            cnn_type=config.RL.PPO.cnn_type,
            policy_config=config.RL.POLICY,
        )

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        tgt_hidden_size: int,
        tgt_encoding: str,
        use_prev_action: bool,
        action_dim: int = 2,
    ):
        super().__init__()
        self.tgt_embeding_size = 0
        self.prev_action_embedding_size = 0
        self._hidden_size = hidden_size
        self.use_prev_action = use_prev_action

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces
            or IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            k = (
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                if IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                in observation_space.spaces
                else IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            )
            self._n_input_goal = observation_space.spaces[k].shape[0]
            self.tgt_embeding_size = tgt_hidden_size
            self.tgt_encoding = tgt_encoding
            if self.tgt_encoding == "sin_cos":
                print("USING SIN COS")
                self.tgt_encoder = nn.Linear(
                    self._n_input_goal + 1, self.tgt_embeding_size
                )
            elif self.tgt_encoding == "linear_1":
                print("USING LINEAR 1")
                self.tgt_encoder = nn.Sequential(
                    nn.Linear(self._n_input_goal, self.tgt_embeding_size),
                    nn.ReLU(),
                )
            elif self.tgt_encoding == "linear_2":
                print("USING LINEAR 2")
                self.tgt_encoder = nn.Sequential(
                    nn.Linear(self._n_input_goal, self.tgt_embeding_size),
                    nn.ReLU(),
                    nn.Linear(self.tgt_embeding_size, self.tgt_embeding_size),
                    nn.ReLU(),
                )
            elif self.tgt_encoding == "ans_bin":
                self.embedding_angle = nn.Embedding(72, 8)
                self.embedding_dist = nn.Embedding(24, 8)
                self.tgt_embeding_size = 16

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        if self.use_prev_action:
            print("USING PREV ACTION")
            self.prev_action_embedding = nn.Linear(action_dim, 32)
            self.prev_action_embedding_size = 32

        nfeats = (
            self._hidden_size + self.prev_action_embedding_size + self.tgt_embeding_size
        )
        print(self._hidden_size + self.prev_action_embedding_size + self.tgt_embeding_size)
        if rnn_type == "LSTM" or rnn_type == "GRU":
            self.state_encoder = build_rnn_state_encoder(
                nfeats,
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

    def get_goal_features(self, x, observations):
        ## Goal vector
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if self.tgt_encoding == "ans_bin":
                dist_emb = self.embedding_dist(
                    goal_observations[:, 0].to(dtype=torch.int64)
                )
                angle_emb = self.embedding_angle(
                    goal_observations[:, 1].to(dtype=torch.int64)
                )
                x.append(dist_emb)
                x.append(angle_emb)
                return x
            if self.tgt_encoding == "sin_cos" and goal_observations.shape[1] == 2:
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            te = self.tgt_encoder(goal_observations)
            x.append(te)
        return x

    def get_prev_action_features(self, x, prev_actions):
        if self.use_prev_action:
            prev_actions = self.prev_action_embedding(prev_actions.float())
            x.append(prev_actions)
        return x

    def get_features(self, observations, prev_actions):
        x = []
        x.append(self.visual_encoder(observations))
        x = self.get_goal_features(x, observations)
        x = self.get_prev_action_features(x, prev_actions)
        x_out = torch.cat(x, dim=1)
        return x_out

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks=None,
        memory=None,
        memory_masks=None,
    ):
        x_out = self.get_features(observations, prev_actions)
        x_out, rnn_hidden_states = self.state_encoder(x_out, rnn_hidden_states, masks)
        return x_out, rnn_hidden_states

class PointNavContextNet(PointNavBaselineNet):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. + Map
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        tgt_hidden_size: int,
        tgt_encoding: str,
        context_hidden_size: int,
        use_prev_action: bool,
        use_waypoint_encoder: bool,
        cnn_type: str,
        policy_config: None = None,
        action_dim: int = 2,
    ):
        super().__init__(
            observation_space=observation_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            rnn_type=rnn_type,
            tgt_hidden_size=tgt_hidden_size,
            tgt_encoding=tgt_encoding,
            use_prev_action=use_prev_action,
            action_dim=action_dim,
        )
        self.cnn_type = cnn_type
        self.use_prev_action = use_prev_action
        self.use_waypoint_encoder = use_waypoint_encoder
        if self.use_prev_action:
            self.prev_action_embedding = nn.Linear(action_dim, 32)
        self.prev_action_embedding_size = 32 if self.use_prev_action else 0
        self.context_hidden_size = context_hidden_size
        self.rnn_type = rnn_type
        self.use_maxpool = policy_config.use_maxpool

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observation_space.spaces:
            self.tgt_encoding = tgt_encoding
            self.tgt_embeding_size = tgt_hidden_size

            if self.tgt_encoding == "sin_cos":
                self.tgt_encoder = nn.Linear(
                    self._n_input_goal + 1, self.tgt_embeding_size
                )
            elif self.tgt_encoding == "linear_2":
                self.tgt_encoder = nn.Sequential(
                    nn.Linear(self._n_input_goal, self.tgt_embeding_size),
                    nn.ReLU(),
                    nn.Linear(self.tgt_embeding_size, self.tgt_embeding_size),
                    nn.ReLU(),
                )
        if (
            "context_map" in observation_space.spaces
            or "context_map_trajectory" in observation_space.spaces
        ):
            if "resnet" in self.cnn_type:
                from habitat_baselines.rl.ddppo.policy import resnet
                from habitat_baselines.rl.ddppo.policy.resnet_policy import (
                    ResNetEncoderContext,
                )

                if "full" in self.cnn_type:
                    self.context_encoder = ResNetEncoderContext(
                        observation_space,
                        baseplanes=32,
                        ngroups=32 // 2,
                        make_backbone=getattr(resnet, self.cnn_type.split("_")[0]),
                        normalize_visual_inputs=False,
                    )
                    dim = np.prod(self.context_encoder.output_shape)
                else:
                    k = (
                        "context_map"
                        if "context_map" in observation_space.keys()
                        else "context_map_trajectory"
                    )
                    k = "context_map"
                    dim = 65536 if "resnet50" in self.cnn_type else 16384
                    dim = 4096 if observation_space[k].shape[0] == 100 else dim
                    in_channels = policy_config.in_channels
                    self.context_encoder = getattr(resnet, self.cnn_type)(
                        in_channels, 32, 32
                    )
                self.context_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(dim, self.context_hidden_size),
                    nn.ReLU(True),
                )
            elif self.cnn_type == "vit":
                self.context_encoder = SimpleViT(
                    image_size=256,
                    patch_size=16,
                    num_classes=self.context_hidden_size,
                    dim=512,
                    depth=2,
                    heads=8,
                    mlp_dim=1024,
                    channels=2,
                )
            elif self.cnn_type == "cnn_2d":
                self.context_encoder = SimpleCNNContext(
                    observation_space, hidden_size, self.cnn_type
                )
        elif "context_waypoint" in observation_space.keys():
            self.context_encoder = nn.Sequential(
                nn.Linear(2, 512),
                nn.ReLU(),
                nn.Linear(512, self.context_hidden_size),
                nn.ReLU(),
            )

        nfeats = (
            self._hidden_size
            + self.prev_action_embedding_size
            + self.tgt_embeding_size
            + self.context_hidden_size
        )
        if rnn_type == "LSTM" or rnn_type == "GRU":
            self.state_encoder = build_rnn_state_encoder(
                nfeats,
                self._hidden_size,
                rnn_type=rnn_type,
                num_layers=num_recurrent_layers,
            )

        print(
            f"##### USING CONTEXT, HIDDEN SIZE: {self.context_hidden_size}, RNN TYPE {rnn_type}, NUM LAYERS: {num_recurrent_layers} #####"
        )
        self.train()

    def get_map_features(self, x, observations):
        if "context_map" in observations:
            obs = observations["context_map"]
        else:
            obs = observations["context_map_trajectory"]

        if "cnn" not in self.cnn_type and "full" not in self.cnn_type:
            obs = obs.permute(0, 3, 1, 2)

        out = self.context_encoder(obs)

        if "resnet" in self.cnn_type:
            out = self.context_fc(out)

        if self.context_hidden_size == 3:
            out = torch.cat(
                [F.sigmoid(out[:, :1]), torch.tanh(out[:, 1:])],
                dim=1,
            )
        x.append(out)
        return x

    def get_features(self, observations, prev_actions):
        x = []
        ## Egocentric observations
        ve = self.visual_encoder(observations)

        x.append(ve)
        x = self.get_goal_features(x, observations)

        ## Map observation
        if (
            "context_map" in observations
            or "context_map_trajectory" in observations
        ):
            x = self.get_map_features(x, observations)
        ## Waypoint observation
        if "context_waypoint" in observations:
            we = self.context_encoder(observations["context_waypoint"])
            x.append(we)
        ## Previous actions
        x = self.get_prev_action_features(x, prev_actions)
        x_out = torch.cat(x, dim=1)
        return x_out

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks=None,
        memory=None,
        memory_masks=None,
    ):
        x_out = self.get_features(observations, prev_actions)
        x_out, rnn_hidden_states = self.state_encoder(x_out, rnn_hidden_states, masks)

        return x_out, rnn_hidden_states
        