#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the Creative Commons license found in the
# LICENSE file in the root directory of this source tree.

# Much of the basic code taken from https://github.com/kevinlu1211/pytorch-decoder-resnet-50-encoder
from abc import ABC

import numpy as np
import torch
from habitat_baselines.rl.ddppo.splitnet.building_blocks import (
    Bridge, ConvBlock, ShallowUpBlockForHourglassNet)
from torch import nn


class EncoderDecoderInterface(nn.Module, ABC):
    def __init__(self, decoder_output_info, create_decoder=True):
        super(EncoderDecoderInterface, self).__init__()
        self.decoder_output_info = decoder_output_info
        self.num_outputs = sum([x[1] for x in self.decoder_output_info])
        self.create_decoder = create_decoder

        self.encoder = None
        self.decoder = None
        self.construct_encoder()
        if self.create_decoder:
            self.construct_decoder()

    def construct_encoder(self):
        raise NotImplementedError

    def construct_decoder(self):
        raise NotImplementedError

    def input_transform(self, x):
        raise NotImplementedError

    @property
    def num_output_channels(self):
        raise NotImplementedError

    def forward(self, x, decoder_enabled):
        x = self.input_transform(x)
        deepest_visual_features = self.encoder(x)
        decoder_outputs = None
        if decoder_enabled:
            decoder_outputs = self.decoder(x)

        return deepest_visual_features, decoder_outputs


class BaseEncoderDecoder(EncoderDecoderInterface, ABC):
    def __init__(self, decoder_output_info, create_decoder=True):
        self.bridge = None
        self.out = None
        super(BaseEncoderDecoder, self).__init__(decoder_output_info, create_decoder)
        self.class_pred_layer = None
        if "semantic" in {info[0] for info in self.decoder_output_info}:
            num_classes = [
                info[1] for info in self.decoder_output_info if info[0] == "semantic"
            ][0]
            self.class_pred_layer = nn.Sequential(
                nn.Linear(128, 128), nn.ELU(inplace=True), nn.Linear(128, num_classes)
            )

    def construct_decoder(self):
        self.bridge = Bridge(128, 128)
        up_blocks = [
            ShallowUpBlockForHourglassNet(128, 128, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(128, 64, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(64, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
            ShallowUpBlockForHourglassNet(32, 32, upsampling_method="bilinear"),
        ]
        self.decoder = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(32, self.num_outputs, kernel_size=1, stride=1)

    def input_transform(self, x):
        x = x.type(torch.float32)
        x = x / 128.0 - 1
        return x

    def forward(self, x, decoder_enabled):
        x = self.input_transform(x)
        deepest_visual_features = self.encoder(x)
        # decoder Part
        decoder_outputs = None
        class_pred = None
        if decoder_enabled:
            x = self.bridge(deepest_visual_features)

            for i, block in enumerate(self.decoder, 1):
                x = block(x)
            decoder_outputs = self.out(x)

            if self.class_pred_layer is not None:
                class_pred_input = torch.mean(deepest_visual_features, dim=(2, 3))
                class_pred = self.class_pred_layer(class_pred_input)

        return deepest_visual_features, decoder_outputs, class_pred


class ShallowVisualEncoder(BaseEncoderDecoder):
    def __init__(self, decoder_output_info, create_decoder=True, use_gray=False):
        self.use_gray = use_gray
        super(ShallowVisualEncoder, self).__init__(decoder_output_info, create_decoder)

    def construct_encoder(self):
        in_channels = 1 if self.use_gray else 3
        self.encoder = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=32,
                padding=3,
                kernel_size=7,
                stride=4,
            ),
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    @property
    def num_output_channels(self):
        return 128
