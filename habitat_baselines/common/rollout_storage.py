#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch
from habitat_baselines.common.tensor_dict import TensorDict


class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        use_external_memory=False,
        external_memory_size=150 + 128,
        external_memory_capacity=150,
        external_memory_dim=1088,
        num_recurrent_layers=1,
        action_shape: int = -1,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(numsteps + 1, num_envs, 1)

        if action_shape == -1:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = 1
            else:
                action_shape = action_space.shape[0]

        self.buffers["actions"] = torch.zeros(numsteps + 1, num_envs, action_shape)
        self.buffers["prev_actions"] = torch.zeros(numsteps + 1, num_envs, action_shape)
        if discrete_actions and action_space.__class__.__name__ == "ActionSpace":
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(numsteps + 1, num_envs, 1, dtype=torch.bool)

        self.use_external_memory = use_external_memory
        self.em_size = external_memory_size
        self.em_capacity = external_memory_capacity
        self.em_dim = external_memory_dim
        # This is kept outside for for backward compatibility with _collect_rollout_step

        # use_external_memory = False
        self.buffers["external_memory"] = torch.zeros(
            self.em_size, numsteps + 1, num_envs, self.em_dim
        )
        self.buffers["external_memory_masks"] = torch.zeros(
            numsteps + 1, num_envs, self.em_size
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        next_memory_features=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
            external_memory=next_memory_features,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [0 for _ in self.current_rollout_step_idxs]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][self.current_rollout_step_idx] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                self.buffers["returns"][step] = gae + self.buffers["value_preds"][step]
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of envirrecurrent_generatoronments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_environments, num_mini_batch)
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[0 : self.current_rollout_step_idx, inds]
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]
            if self.use_external_memory:
                batch["external_memory"] = batch["external_memory"][
                    0 : self.current_rollout_step_idx, :, inds
                ]
                # batch["external_memory_masks"] = batch["external_memory_masks"][
                #     0 : self.current_rollout_step_idx, inds
                # ]

            yield batch.map(lambda v: v.flatten(0, 1))

    @property
    def external_memory(self):
        return self.buffers["external_memory"].memory

    @property
    def external_memory_masks(self):
        return self.buffers["external_memory"].masks

    @property
    def external_memory_idx(self):
        return self.buffers["external_memory"].idx


class ExternalMemory:
    def __init__(self, num_envs, total_size, capacity, dim, num_copies=1):
        r"""An external memory that keeps track of observations over time.
        Inputs:
            num_envs - number of parallel environments
            capacity - total capacity of the memory per episode
            total_size - capacity + additional buffer size for rollout updates
            dim - size of observations
            num_copies - number of copies of the data to maintain for efficient training
        """
        self.total_size = total_size
        self.capacity = capacity
        self.dim = dim

        self.ext_masks = torch.zeros(num_envs, self.total_size)
        self.ext_memory = torch.zeros(self.total_size, num_copies, num_envs, self.dim)
        self.idx = 0

    @classmethod
    def from_tensor_dict(cls, tensor_dict):
        instance = cls(1, 1, 1, 1, num_copies=1)
        instance.ext_masks = tensor_dict["masks"]
        instance.ext_memory = tensor_dict["memory"]
        instance.idx = tensor_dict["idx"].item()
        instance.total_size = tensor_dict["total_size"].item()
        instance.capacity = tensor_dict["capacity"].item()
        instance.dim = tensor_dict["dim"].item()
        return instance

    def to_tensor_dict(self):
        device = self.ext_masks.device
        t = TensorDict()
        t["ext_masks"] = self.ext_masks
        t["ext_memory"] = self.ext_memory
        t["idx"] = torch.tensor([self.idx], device=device)
        t["total_size"] = torch.tensor([self.total_size], device=device)
        t["capacity"] = torch.tensor([self.capacity], device=device)
        t["dim"] = torch.tensor([self.dim], device=device)
        return t

    def insert(self, em_features, not_done_masks):
        # Update memory storage and add new memory as a valid entry
        self.ext_memory[self.idx].copy_(em_features.unsqueeze(0))
        # Account for overflow capacity
        capacity_overflow_flag = self.ext_masks.sum(1) == self.capacity
        assert not torch.any(self.ext_masks.sum(1) > self.capacity)
        self.ext_masks[capacity_overflow_flag, self.idx - self.capacity] = 0.0
        self.ext_masks[:, self.idx] = 1.0
        # Mask out the entire memory for the next observation if episode done
        self.ext_masks *= not_done_masks
        self.idx = (self.idx + 1) % self.total_size

    def pop_at(self, idx):
        self.masks = torch.cat(
            [self.ext_masks[:idx, :], self.ext_masks[idx + 1 :, :]], dim=0
        )
        self.ext_memory = torch.cat(
            [self.ext_memory[:, :, :idx, :], self.ext_memory[:, :, idx + 1 :, :]], dim=2
        )

    def to(self, device):
        self.ext_masks = self.ext_masks.to(device)
        self.ext_memory = self.ext_memory.to(device)
