# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Model architecture with CNN networks for pre-decoders

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from training.distributed import DistributedManager
from types import SimpleNamespace

try:
    from natten import NeighborhoodAttention3D
    _NATTEN_IMPORT_ERROR = None
except ImportError as exc:
    NeighborhoodAttention3D = None
    _NATTEN_IMPORT_ERROR = exc


def _get_activation_class(name):
    if name == "relu":
        return nn.ReLU
    elif name == "gelu":
        return nn.GELU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Unsupported activation: {name}")


def _normalize_3d_param(value, name):
    if isinstance(value, int):
        return (value, value, value)
    try:
        values = list(value)
    except TypeError:
        values = None
    if values is not None and len(values) == 3:
        return tuple(int(v) for v in values)
    raise ValueError(f"{name} must be an int or a length-3 sequence, got {value!r}")


def _normalize_block_dilations(dilation_cfg, num_blocks):
    if isinstance(dilation_cfg, int):
        dilations = [dilation_cfg] * num_blocks
    else:
        try:
            dilations = list(dilation_cfg)
        except TypeError as exc:
            raise ValueError(
                "dilation must be an int or a length-"
                f"{num_blocks} sequence, got {dilation_cfg!r}"
            ) from exc

    if len(dilations) != num_blocks:
        raise ValueError(f"dilation must provide exactly {num_blocks} values, got {len(dilations)}")
    return dilations


class ResidualBlock3D(nn.Module):

    def __init__(self, channels, kernel_sizes, activation, post_activation=True):
        """
        2-conv residual block with optional post-activation.
        channels: List of 3 ints = [in_ch, mid_ch, out_ch]
        kernel_sizes: List of 2 ints (or tuples) = [k1, k2]
        activation: activation class (e.g. nn.GELU), not an instance
        post_activation: if True, apply activation after skip addition;
                         if False, output raw values (use for final block)

        Forward (post_activation=True):  y = Act(x + Conv2(BN2(Act(BN1(Conv1(x))))))
        Forward (post_activation=False): y = x + Conv2(BN2(Act(BN1(Conv1(x)))))
        """
        super(ResidualBlock3D, self).__init__()

        in_ch, mid_ch, out_ch = channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2),
            nn.BatchNorm3d(mid_ch), activation()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_ch, out_ch, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2),
            nn.BatchNorm3d(out_ch)
        )

        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1)

        self.post_act = activation() if post_activation else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.post_act(out + identity)


class CascadeBottleneckBlock3D(nn.Module):
    """
    Bottleneck residual block used for the cascade Conv3D ablation.

    Unlike the full-decoder port, this variant can change channel count so the
    pre-decoder enters the fixed hidden width directly from the raw 4-channel
    input, without a separate 1x1x1 embedding stem.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bottleneck_ratio=4, num_blocks=1):
        super().__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        bottleneck_ratio = int(bottleneck_ratio)
        num_blocks = int(num_blocks)
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got {(in_channels, out_channels)}"
            )
        if bottleneck_ratio <= 0:
            raise ValueError(f"bottleneck_ratio must be positive, got {bottleneck_ratio}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        bottleneck_channels = max(1, out_channels // bottleneck_ratio)
        kernel_size = _normalize_3d_param(kernel_size, "kernel_size")
        padding = tuple(k // 2 for k in kernel_size)

        self.residual_scale = 1.0 / math.sqrt(2.0 * num_blocks)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.reduce = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1),
        )
        self.message_passing = nn.Sequential(
            nn.BatchNorm3d(bottleneck_channels),
            nn.SiLU(),
            nn.Conv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )
        self.restore = nn.Sequential(
            nn.BatchNorm3d(bottleneck_channels),
            nn.SiLU(),
            nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        identity = self.skip(x) * self.residual_scale
        out = self.reduce(x)
        out = self.message_passing(out)
        out = self.restore(out)
        return out + identity


class NATTENBlock3D(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        kernel_size,
        dilation=1,
        ffn_ratio=4.0,
        drop=0.0,
        activation_class=nn.GELU,
    ):
        super(NATTENBlock3D, self).__init__()

        if NeighborhoodAttention3D is None:
            raise ImportError(
                "PreDecoderModelMemory_v3 requires the 'natten' package. "
                "Install it with `pip install natten` and verify CUDA compatibility "
                "with the local PyTorch version."
            ) from _NATTEN_IMPORT_ERROR

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.kernel_size = _normalize_3d_param(kernel_size, "kernel_size")
        self.base_dilation = _normalize_3d_param(dilation, "dilation")
        hidden_dim = int(embed_dim * ffn_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = NeighborhoodAttention3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            is_causal=False,
            qkv_bias=True,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation_class(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def _get_effective_dilation(self, x):
        input_shape = tuple(int(dim) for dim in x.shape[1:4])
        effective_dilation = []
        for size, kernel, base_dilation in zip(input_shape, self.kernel_size, self.base_dilation):
            if size < kernel:
                raise ValueError(
                    "NeighborhoodAttention3D requires each input dimension to be at least "
                    f"the kernel size. Got input shape {input_shape} with kernel_size "
                    f"{self.kernel_size}."
                )
            max_dilation = max(1, size // kernel)
            effective_dilation.append(min(base_dilation, max_dilation))
        return tuple(effective_dilation)

    def forward(self, x):
        # x: (B, T, D, D, C) -- channels-last
        self.attn.dilation = self._get_effective_dilation(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


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
        # Allow the last Conv to be widened beyond out_channels; downstream
        # consumers see only the first out_channels (the rest is "tile padding"
        # that exists so TensorRT picks a better cutlass tile on B200).
        assert filters[-1] >= out_channels, (
            f"num_filters[-1]={filters[-1]} must be >= out_channels={out_channels}"
        )
        self.out_channels = int(out_channels)
        self._head_widened = filters[-1] > out_channels

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
        y = self.net(x)  # x: (B, in_channels, T, H, W)
        if self._head_widened:
            # Trained with COUT=filters[-1] but downstream only uses the first
            # out_channels channels; the rest are auxiliary capacity that lets
            # the head Conv use a wider cutlass tile on B200.
            y = y[:, :self.out_channels].contiguous()
        return y


class PreDecoderModelMemory_v2(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemory_v2, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds
        self.dropout_p = cfg.model.dropout_p
        activation_class = self._get_activation_class(cfg.model.activation)

        filters = cfg.model.num_filters
        kernel_sizes = cfg.model.kernel_size
        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels

        assert len(filters) % 2 == 0, \
            "num_filters length must be even (each residual block consumes a pair)."
        assert len(filters) == len(kernel_sizes), \
            "Mismatch: num_filters and kernel_size must be the same length."
        assert filters[-1] == out_channels, \
            f"The last element of num_filters must match out_channels ({out_channels}), but got {filters[-1]}"

        # === All Residual Blocks ===
        # Filter list is consumed in pairs: [mid_ch, out_ch] per block.
        # in_ch comes from the previous block's out_ch (or input_channels for the first).
        # The last block has no post-activation (raw logits output).
        self.layers = nn.ModuleList()
        in_ch = input_channels
        num_blocks = len(filters) // 2
        for b in range(num_blocks):
            mid_ch = filters[2 * b]
            out_ch = filters[2 * b + 1]
            ks = [kernel_sizes[2 * b], kernel_sizes[2 * b + 1]]
            is_last = (b == num_blocks - 1)
            self.layers.append(
                ResidualBlock3D(
                    channels=[in_ch, mid_ch, out_ch],
                    kernel_sizes=ks,
                    activation=activation_class,
                    post_activation=not is_last
                )
            )
            in_ch = out_ch

    def _get_activation_class(self, name):
        return _get_activation_class(name)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PreDecoderModelMemory_v3(nn.Module):

    def __init__(self, cfg):
        super(PreDecoderModelMemory_v3, self).__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds

        embed_dim = cfg.model.embed_dim
        num_heads = cfg.model.num_heads
        num_blocks = cfg.model.num_blocks
        kernel_size = cfg.model.kernel_size
        ffn_ratio = cfg.model.ffn_ratio
        drop = cfg.model.dropout_p
        input_channels = cfg.model.input_channels
        out_channels = cfg.model.out_channels
        activation_class = _get_activation_class(cfg.model.activation)
        dilations = _normalize_block_dilations(getattr(cfg.model, "dilation", 1), num_blocks)

        # Two-layer tokenizer: learn local motifs before the attention core,
        # while keeping the output projection pointwise.
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            activation_class(),
            nn.Conv3d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            activation_class(),
        )

        self.blocks = nn.ModuleList(
            [
                NATTENBlock3D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    ffn_ratio=ffn_ratio,
                    drop=drop,
                    activation_class=activation_class,
                ) for i in range(num_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(embed_dim)

        # Final per-position projection back to logits.
        self.output_projection = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        x = self.output_projection(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class PreDecoderModelMemory_Cascade(nn.Module):
    """
    Cascade-style Conv3D pre-decoder for the original 4-channel residual task.

    The backbone uses bottleneck residual blocks with a fixed hidden width, but
    the first block consumes the raw input channels directly so there is no
    standalone 1x1x1 embedding stem before the cascade stack.
    """

    def __init__(self, cfg):
        super().__init__()

        self.distance = cfg.distance
        self.n_rounds = cfg.n_rounds

        input_channels = int(cfg.model.input_channels)
        out_channels = int(cfg.model.out_channels)
        embed_dim = int(cfg.model.embed_dim)
        num_blocks = int(cfg.model.num_blocks)
        bottleneck_ratio = int(getattr(cfg.model, "bottleneck_ratio", 4))
        kernel_size = getattr(cfg.model, "kernel_size", 3)

        if embed_dim <= 0:
            raise ValueError(f"model.embed_dim must be positive, got {embed_dim}")
        if num_blocks <= 0:
            raise ValueError(f"model.num_blocks must be positive, got {num_blocks}")

        # Stem (first block). The paper's Model B uses a plain Conv3d stem that
        # lifts the raw input channels to embed_dim; set model.plain_stem=True for
        # that variant. When unset it defaults to a full bottleneck block as the
        # stem (kept for backward compatibility).
        if bool(getattr(cfg.model, "plain_stem", False)):
            if isinstance(kernel_size, int):
                _pad = kernel_size // 2
            else:
                _pad = tuple(k // 2 for k in kernel_size)
            self.input_block = nn.Conv3d(
                input_channels, embed_dim, kernel_size=kernel_size, padding=_pad
            )
        else:
            self.input_block = CascadeBottleneckBlock3D(
                in_channels=input_channels,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                bottleneck_ratio=bottleneck_ratio,
                num_blocks=num_blocks,
            )
        self.blocks = nn.ModuleList(
            [
                CascadeBottleneckBlock3D(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=kernel_size,
                    bottleneck_ratio=bottleneck_ratio,
                    num_blocks=num_blocks,
                ) for _ in range(max(0, num_blocks - 1))
            ]
        )
        self.output_projection = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_block(x)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(x)


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
    cfg.model.dropout_p = 0.0
    cfg.model.activation = 'gelu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 2
    # 3 blocks × 2 convs = 6 conv layers; last block has no post-activation
    cfg.model.num_filters = [16, 16, 16, 16, 8, 2]
    cfg.model.kernel_size = [3] * len(cfg.model.num_filters)
    return cfg


def get_mock_config_v3():
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.distance = 11
    cfg.n_rounds = 7
    cfg.model.dropout_p = 0.1
    cfg.model.activation = 'gelu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 4
    cfg.model.embed_dim = 64
    cfg.model.num_heads = 4
    cfg.model.num_blocks = 4
    cfg.model.kernel_size = 3
    cfg.model.dilation = 1
    cfg.model.ffn_ratio = 4.0
    return cfg


def get_mock_config_cascade():
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.distance = 11
    cfg.n_rounds = 7
    cfg.model.dropout_p = 0.05
    cfg.model.activation = 'gelu'
    cfg.model.input_channels = 4
    cfg.model.out_channels = 4
    cfg.model.embed_dim = 32
    cfg.model.num_blocks = 3
    cfg.model.bottleneck_ratio = 4
    cfg.model.kernel_size = 3
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


def test_model_v3():
    if NeighborhoodAttention3D is None:
        raise ImportError(
            "test_model_v3 requires the 'natten' package to be installed."
        ) from _NATTEN_IMPORT_ERROR

    cfg = get_mock_config_v3()
    model = PreDecoderModelMemory_v3(cfg)

    B, C_in, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    x = torch.randn(B, C_in, T, D, D)
    out = model(x)

    expected_shape = (B, cfg.model.out_channels, T, D, D)
    assert out.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {out.shape}"

    T2, D2 = 5, 15
    x2 = torch.randn(B, C_in, T2, D2, D2)
    out2 = model(x2)
    expected_shape2 = (B, cfg.model.out_channels, T2, D2, D2)
    assert out2.shape == expected_shape2, \
        f"Arbitrary-size test failed: expected {expected_shape2}, got {out2.shape}"
    print("Model v3 (NATTEN) test passed. Shapes:", out.shape, out2.shape)


def test_model_cascade():
    cfg = get_mock_config_cascade()
    model = PreDecoderModelMemory_Cascade(cfg)

    B, C_in, T, D = 2, cfg.model.input_channels, cfg.n_rounds, cfg.distance
    x = torch.randn(B, C_in, T, D, D)
    out = model(x)

    expected_shape = (B, cfg.model.out_channels, T, D, D)
    assert out.shape == expected_shape, \
        f"Output shape mismatch: expected {expected_shape}, got {out.shape}"
    print("Model cascade test passed. Shape:", out.shape)


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
    test_model_v3()
    test_model_cascade()
