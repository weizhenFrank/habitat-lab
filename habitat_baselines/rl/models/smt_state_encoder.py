#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SMTStateEncoder(nn.Module):
    """
    The core Scene Memory Transformer block from https://arxiv.org/abs/1903.03878
    """

    def __init__(
        self,
        input_size: int,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        pose_indices: Optional[Tuple[int, int]] = None,
        pretraining: bool = False,
    ):
        r"""A Transformer for encoding the state in RL and decoding features based on
        the observation and goal encodings.

        Supports masking the hidden state during various timesteps in the forward pass

        Args:
            input_size: The input size of the SMT
            nhead: The number of encoding and decoding attention heads
            num_encoder_layers: The number of encoder layers
            num_decoder_layers: The number of decoder layers
            dim_feedforward: The hidden size of feedforward layers in the transformer
            dropout: The dropout value after each attention layer
            activation: The activation to use after each linear layer
        """

        super().__init__()
        self._input_size = input_size
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._activation = activation
        self._pose_indices = pose_indices
        self._pretraining = pretraining

        if pose_indices is not None:
            pose_dims = pose_indices[1] - pose_indices[0]
            self.pose_encoder = nn.Linear(5, 16)
            input_size += 16 - pose_dims
            self._use_pose_encoding = True
        else:
            self._use_pose_encoding = False

        self.fusion_encoder = nn.Sequential(
            nn.Linear(input_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_feedforward),
        )

        self.transformer = nn.Transformer(
            d_model=dim_feedforward,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def _convert_masks_to_transformer_format(self, memory_masks):
        r"""The memory_masks is a FloatTensor with
            -   zeros for invalid locations, and
            -   ones for valid locations.external_memory_features

        The required format is a BoolTensor with
            -   True for invalid locations, and
            -   False for valid locations
        """
        return (1 - memory_masks) > 0

    def single_forward(self, x, memory, memory_masks, goal=None):
        r"""Forward for a non-sequence input

        Args:
            x: (N, input_size) Tensor
            memory: The memory of encoded observations in the episode. It is a
                (M, N, input_size) Tensor.
            memory_masks: The masks indicating the set of valid memory locations
                for the current observations. It is a (N, M) Tensor.
            goal: (N, goal_dims) Tensor (optional)
        """
        # If memory_masks is all zeros for a data point, x_att will be NaN.
        # In these cases, just set memory_mask to ones and replace x_att with x.
        # all_zeros_mask = (memory_masks.sum(dim=1) == 0).float().unsqueeze(1)
        # memory_masks = 1.0 * all_zeros_mask + memory_masks * (1 - all_zeros_mask)

        memory_masks = torch.cat(
            [
                memory_masks,
                torch.ones([memory_masks.shape[0], 1], device=memory_masks.device),
            ],
            dim=1,
        )
        # Compress features
        x = x.unsqueeze(0)
        if x.size(1) != memory.size(1):
            print("x shape: ", x.shape, "memory shape: ", memory.shape)
            # x = x.reshape(-1, memory.size(1), x.size(2))
            memory = memory.reshape(-1, x.size(1), memory.size(2))
            # x = x.reshape(-1, memory.size(1), x.size(2))
        # memory = torch.cat([memory, x.unsqueeze(0)])
        memory = torch.cat([memory, x])
        M, bs = memory.shape[:2]

        memory = self.fusion_encoder(memory.view(M * bs, -1)).view(M, bs, -1)

        # Transformer operations
        t_masks = self._convert_masks_to_transformer_format(memory_masks)

        decode_memory = False
        if decode_memory:
            x_att = self.transformer(
                memory,
                memory,
                src_key_padding_mask=t_masks,
                tgt_key_padding_mask=t_masks,
                memory_key_padding_mask=t_masks,
            )[-1]
        else:
            x_att = self.transformer(
                memory,
                memory[-1:],
                # src_key_padding_mask=t_masks[:, : memory.size(0)],
                # memory_key_padding_mask=t_masks[:, : memory.size(0)],
                # memory_key_padding_mask=t_masks[:, -1:],
            )[-1]
        return x_att

    @property
    def hidden_state_size(self):
        return self._dim_feedforward

    def forward(self, x, memory, *args, **kwargs):
        """
        Single input case:
            Inputs:
                x - (N, input_size)
                memory - (M, N, input_size)
                memory_masks - (N, M)
        Sequential input case:
            Inputs:
                x - (T*N, input_size)
                memory - (M, N, input_size)
                memory_masks - (T*N, M)
        """
        return self.single_forward(x, memory, *args, **kwargs)
