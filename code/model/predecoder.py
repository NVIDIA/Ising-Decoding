# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

## Model architecture with CNN networks for pre-decoders

import torch
import torch.nn as nn
from types import SimpleNamespace


class ResidualBlock3D(nn.Module):

    def __init__(self, channels, kernel_sizes, activation):
        """
        channels: List of 4 ints = [in1, out1, out2, out3]
        kernel_sizes: List of 3 ints (or tuples) = k1, k2, k3
        """
        super(ResidualBlock3D, self).__init__()
        self.activation = activation()  # instantiate once

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                channels[0], channels[1], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2
            ),
            nn.BatchNorm3d(channels[1]),
            self.activation  # instance
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                channels[1], channels[2], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2
            ),
            nn.BatchNorm3d(channels[2]),
            self.activation  # instance
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                channels[2], channels[3], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2
            ), nn.BatchNorm3d(channels[3])
        )

        self.skip = nn.Identity()
        if channels[0] != channels[3]:
            self.skip = nn.Conv3d(channels[0], channels[3], kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.activation(out + identity)


class PreDecoderModelMemory_v1(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemory_v1, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p
        self.activation_fn = self._get_activation(cfg.model.activation)

        filters = cfg.model.num_filters
        kernel_sizes = cfg.model.kernel_size

        assert len(filters) == len(kernel_sizes), \
            "Mismatch: num_filters and kernel_size must be the same length."

        # === Configurable input and output channels ===
        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        assert filters[-1] == out_channels, \
            f"The last element of num_filters must match the configured out_channels ({out_channels}), but got {filters[-1]}"

        layers = []
        in_channels = input_channels  # 4 input channels from trainX

        for i in range(len(filters)):
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2  # keeps same shape (optional)
                )
            )
            if i < len(filters) - 1:  # last layer should not have dropout or activation
                layers.append(nn.Dropout3d(p=self.dropout_p))
                layers.append(self.activation_fn)
            in_channels = filters[i]

        self.net = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU(approximate='tanh')
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.net(x)  # x: (B, 4, T, D, D)


class PreDecoderModelMemory_v2(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemory_v2, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p
        activation_class = self._get_activation_class(cfg.model.activation)
        self.activation_fn = activation_class()

        filters = cfg.model.num_filters
        kernel_sizes = cfg.model.kernel_size

        assert (len(filters) - 2) % 3 == 0, \
            "The number of filters minus 2 (for the first and last layers) must be divisible by 3."
        assert len(filters) == len(kernel_sizes), \
            "Mismatch: num_filters and kernel_size must be the same length."
        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        assert filters[-1] == out_channels, \
            f"The last element of num_filters must match the configured out_channels ({out_channels}), but got {filters[-1]}"

        # === Initial Conv3D layer ===
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels=input_channels,
                    out_channels=filters[0],
                    kernel_size=kernel_sizes[0],
                    padding=kernel_sizes[0] // 2
                ), nn.BatchNorm3d(filters[0]), self.activation_fn
            )
        )

        # === Residual Blocks ===
        in_ch = filters[0]
        i = 1
        while i + 2 < len(filters) - 1:
            out_ch1, out_ch2, out_ch3 = filters[i], filters[i + 1], filters[i + 2]
            ks = [kernel_sizes[i], kernel_sizes[i + 1], kernel_sizes[i + 2]]
            self.layers.append(
                ResidualBlock3D(
                    channels=[in_ch, out_ch1, out_ch2, out_ch3],
                    kernel_sizes=ks,
                    activation=activation_class
                )
            )
            in_ch = out_ch3
            i += 3

        # === Final Conv3D layer ===
        self.final_conv = nn.Conv3d(
            in_channels=filters[-2],
            out_channels=out_channels,
            kernel_size=kernel_sizes[-1],
            padding=kernel_sizes[-1] // 2
        )

    def _get_activation_class(self, name):
        if name == "relu":
            return nn.ReLU
        elif name == "gelu":
            return nn.GELU
        elif name == "leakyrelu":
            return nn.LeakyReLU
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


# === Define a mock config using SimpleNamespace ===
def get_mock_config():
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.distance = 11
    cfg.n_rounds = 3
    cfg.model.dropout_p = 0.1
    cfg.model.activation = 'relu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 2
    cfg.model.num_filters = [8, 4, 2]
    cfg.model.kernel_size = [3, 3, 3]
    return cfg


# === Mock config for testing ===
def get_mock_config_v2():
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.distance = 11
    cfg.n_rounds = 3
    cfg.model.dropout_p = 0.1
    cfg.model.activation = 'relu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 2
    cfg.model.num_filters = [8, 16, 16, 8, 8, 8, 4, 2]  # (len - 2) % 3 == 0
    cfg.model.kernel_size = [3] * len(cfg.model.num_filters)
    return cfg


# === Test ===
def test_model_v2():
    cfg = get_mock_config_v2()
    model = PreDecoderModelMemory_v2(cfg)

    B, C_in, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    x = torch.randn(B, C_in, T, D, D)
    out = model(x)

    expected_shape = (B, cfg.model.out_channels, T, D, D)
    assert out.shape == expected_shape, f"❌ Output shape mismatch: expected {expected_shape}, got {out.shape}"
    print("✅ Model v2 test passed. Output shape:", out.shape)


# === Run the test ===
def test_model():
    cfg = get_mock_config()
    model = PreDecoderModelMemory_v1(cfg)

    B, C_in, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    input_tensor = torch.randn(B, C_in, T, D, D)

    output = model(input_tensor)

    expected_shape = (B, cfg.model.out_channels, T, D, D)
    assert output.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"

    print("✅ Model test passed. Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
    test_model_v2()
