# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
A3-FPN: Asymptotic Content-Aware Pyramid Attention Network for Dense Visual Prediction.

Reference: https://arxiv.org/abs/2604.10210

This module provides A3-FPN neck implementation adapted for Ultralytics YOLO models.
It replaces the standard PANet/FPN neck with content-aware multi-scale feature fusion
using deformable convolutions (DCNv4) and channel-wise attention mechanisms.

Key components:
    - Fusion (MCA): Multi-scale Context-aware Attention for Feature Fusion
    - Reassemble (ICA): Intra-scale Content-Aware Attention for Feature Reassemble
    - Resampler: Context-aware resampling via DCNv4
    - Body: Core A3-FPN architecture with top-down pathway

Example:
    >>> from ultralytics.nn.modules.a3fpn import A3FPN
    >>> neck = A3FPN(in_channels=[256, 512, 1024], out_channels=256)
    >>> feats = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
    >>> outputs = neck(feats)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Dict

__all__ = (
    "A3FPN",
    "A3FPNBody",
    "Fusion",
    "Reassemble",
    "Resampler",
    "SampleBlock",
    "CSPRepLayer",
    "RepVggBlock",
)


# ---------------------------------------------------------------------------
# Helper utilities (replace mmdetection dependencies)
# ---------------------------------------------------------------------------

def _get_norm(norm, out_channels, **kwargs) -> nn.Module:
    """Get normalization layer (replaces mmdet's get_norm)."""
    if norm is None or norm is False:
        return nn.Identity()
    if isinstance(norm, nn.Module):
        return norm
    if isinstance(norm, str):
        norm = norm.upper()
        if norm == "BN":
            return nn.BatchNorm2d(out_channels, **kwargs)
        if norm == "SYNCBN":
            return nn.SyncBatchNorm(out_channels, **kwargs)
        if norm == "GN":
            return nn.GroupNorm(32, out_channels, **kwargs)
        if norm == "LN":
            return nn.GroupNorm(1, out_channels, **kwargs)  # LayerNorm over C,H,W
        if norm == "LN2D":
            return nn.LayerNorm(out_channels, **kwargs)
    return nn.Identity()


def _get_activation(act: Union[str, nn.Module, bool] = "GELU") -> nn.Module:
    """Get activation module (replaces mmdet's get_activation)."""
    if isinstance(act, nn.Module):
        return act
    if act is None or act is False:
        return nn.Identity()
    if isinstance(act, str):
        act = act.lower()
        if act == "silu":
            return nn.SiLU()
        if act == "relu":
            return nn.ReLU()
        if act == "leaky_relu":
            return nn.LeakyReLU()
        if act == "gelu":
            return nn.GELU()
    return nn.Identity()


def _autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class A3Conv(nn.Module):
    """Standard convolution with batch norm + activation, used in A3-FPN."""

    default_act = nn.GELU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act: Union[bool, nn.Module] = True,
        bias=False,
        norm: Union[bool, str, nn.Module] = "BN",
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=bias)
        self.norm = _get_norm(norm, c2) if isinstance(norm, (str, bool)) else (
            norm if isinstance(norm, nn.Module) else nn.Identity()
        )
        self.act = self.default_act if act is True else (
            _get_activation(act) if isinstance(act, str) else (
                act if isinstance(act, nn.Module) else nn.Identity()
            )
        )

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class A3Conv2(A3Conv):
    """Dual-path Conv (3x3 + 1x1) used in A3-FPN."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True, norm="BN"):
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act, norm=norm)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, _autopad(1, p, d), groups=g, dilation=d, bias=False)

    def forward(self, x):
        return self.act(self.norm(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        return self.act(self.norm(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions into one."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class GroupBatchnorm2d(nn.Module):
    """Group-wise batch normalization for spatial feature maps."""

    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super().__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W).contiguous()
        return x * self.weight + self.bias


class RepVggBlock(nn.Module):
    """RepVGG-style block with 3x3 + 1x1 branches (deploy-time fusion)."""

    def __init__(
        self,
        ch_in,
        ch_out,
        act: Union[str, nn.Module, bool] = "GELU",
        norm: Union[str, bool, nn.Module] = "BN",
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = A3Conv(ch_in, ch_out, 3, 1, 1, act=False, norm=norm)
        self.conv2 = A3Conv(ch_in, ch_out, 1, 1, 0, act=False, norm=norm)
        self.act = _get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)

    def convert_to_deploy(self):
        """Fuse branches into a single conv for deployment."""
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        kernel, bias = self._get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def _get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(branch: A3Conv):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        if hasattr(branch.norm, "running_mean"):
            running_mean = branch.norm.running_mean
            running_var = branch.norm.running_var
            gamma = branch.norm.weight
            beta = branch.norm.bias
            eps = branch.norm.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        return kernel, torch.zeros(kernel.size(0), device=kernel.device)


class CSPRepLayer(nn.Module):
    """CSP-style layer with RepVGG blocks (from RT-DETR / A3-FPN)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=1,
        expansion=1.0,
        bias=False,
        act: Union[nn.Module, str] = nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = A3Conv(in_channels, hidden_channels, 1, 1, bias=bias, act=act, norm=norm)
        self.conv2 = A3Conv(in_channels, hidden_channels, 1, 1, bias=bias, act=act, norm=norm)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act, norm=norm)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = A3Conv(hidden_channels, out_channels, 1, 1, bias=True, act=False, norm=False)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# ---------------------------------------------------------------------------
# SampleBlock - up/down sampling helper
# ---------------------------------------------------------------------------

class SampleBlock(nn.Module):
    """Upsample or downsample feature maps between FPN levels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sample: str = "upsample",
        scale_factor: int = 2,
        upsample_method: str = "bilinear",
        align_corners: Optional[bool] = False,
        group: int = 1,
        act: Optional[Union[bool, nn.Module]] = nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
    ):
        super().__init__()
        if sample == "upsample":
            layers = []
            if in_channels != out_channels:
                layers.append(A3Conv(in_channels, out_channels, 1, act=act, norm=norm))
            layers.append(nn.Upsample(scale_factor=scale_factor, mode=upsample_method, align_corners=align_corners))
            self.sample_layer = nn.Sequential(*layers)
        elif sample == "downsample":
            g = in_channels if in_channels == out_channels else group
            self.sample_layer = A3Conv(in_channels, out_channels, scale_factor, scale_factor, 0, g=g, act=act, norm=norm)
        else:
            raise ValueError("sample must be 'upsample' or 'downsample'")

    def forward(self, feat_s: torch.Tensor) -> torch.Tensor:
        return self.sample_layer(feat_s)


# ---------------------------------------------------------------------------
# Resampler - context-aware resampling via DCNv4
# ---------------------------------------------------------------------------

class Resampler(nn.Module):
    """Context-aware resampling using DCNv4 (deformable convolution v4).

    Note:
        Requires DCNv4 to be installed. If not available, this module will
        be skipped gracefully (identity) when using_resampling=False.
    """

    def __init__(
        self,
        channels: int,
        act: Optional[Union[bool, nn.Module]] = nn.GELU(),
        dcn_norm: Optional[Union[bool, str, nn.Module]] = "LN",
        dcn_group: int = 4,
        offset_scale: float = 0.5,
        dw_kernel_size: int = 3,
        dcn_output_bias: bool = False,
        center_feature_scale: bool = False,
        remove_center: bool = False,
        without_pointwise: bool = False,
    ):
        super().__init__()
        try:
            from ultralytics.nn.modules.dcnv4 import DCNv4 as dcn_v4
            from ultralytics.nn.modules.dcnv4.functions.dcnv4_func import ext as dcn_ext
            if dcn_ext is None:
                raise ImportError("DCNv4 CUDA extension not compiled")
            self.resampling = dcn_v4(
                channels, 3, stride=1, pad=1, dilation=1, group=dcn_group,
                offset_scale=offset_scale, dw_kernel_size=dw_kernel_size,
                output_bias=dcn_output_bias, center_feature_scale=center_feature_scale,
                remove_center=remove_center, without_pointwise=without_pointwise,
                extra_offset_mask=True,
            )
            self._has_dcn = True
        except ImportError:
            self.resampling = nn.Identity()
            self._has_dcn = False

        self.norm = _get_norm(dcn_norm, channels) if dcn_norm else nn.Identity()
        self.act = nn.GELU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, context_info: torch.Tensor, sampled_feat: torch.Tensor) -> torch.Tensor:
        if self._has_dcn:
            adjusted = self.act(self.norm(self.resampling([sampled_feat, context_info])))
            return adjusted
        return sampled_feat


# ---------------------------------------------------------------------------
# Fusion (MCA) - Multi-scale Context-aware Attention
# ---------------------------------------------------------------------------

class Fusion(nn.Module):
    """Multi-scale Context-aware Attention for Feature Fusion (MCA).

    Collects supplementary content from adjacent levels to generate position-wise
    offsets and weights for context-aware resampling, and learns deep context
    reweights to improve intra-category similarity.

    Args:
        in_channels: Number of input channels per level.
        out_channels: Number of output channels.
        num_fusion: Number of feature levels to fuse (2, 3, or 4).
        compress_c: Compression channels for attention weights.
        act: Activation function.
        norm: Normalization type ("BN", "LN", etc.).
        num_blocks: Number of RepVGG blocks in CSPRepLayer.
        expansion: Expansion ratio for CSPRepLayer.
        using_resampling: Whether to use DCNv4 context-aware resampling.
        dcn_group: DCNv4 group parameter.
        dcn_config: Additional DCNv4 configuration.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_fusion=2,
        compress_c=16,
        act=nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
        num_blocks: int = 1,
        expansion: float = 1.0,
        using_resampling: Optional[bool] = False,
        dcn_group: int = 4,
        dcn_config: dict = None,
    ):
        super().__init__()
        if dcn_config is None:
            dcn_config = {}

        self.using_resampling = using_resampling
        self.num_fusion = num_fusion

        # Build per-level weight convolutions
        for i in range(num_fusion):
            setattr(
                self, f"weight_level_{i + 1}",
                A3Conv(in_channels, compress_c, 1, 1, bias=False, act=act, norm=norm),
            )

        self.weight_levels = CSPRepLayer(
            in_channels=compress_c * num_fusion, out_channels=num_fusion,
            num_blocks=num_blocks, expansion=expansion, act=act, norm=norm,
        )

        # Context-aware resampling (DCNv4)
        if using_resampling:
            self.context_conv = A3Conv(
                in_channels * num_fusion, in_channels * (num_fusion - 1),
                k=1, s=1, p=0, bias=False, act=act, norm=norm,
            )
            for i in range(num_fusion - 1):
                setattr(
                    self, f"resampler{i + 1}",
                    Resampler(channels=in_channels, act=act, dcn_group=dcn_group, **dcn_config),
                )

        # Final projection
        self.conv = (
            A3Conv(in_channels, out_channels, 1, 1, act=False, norm=False, bias=True)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tuple of (sampled_feat_1, ..., unsampled_feat). Length = num_fusion.

        Returns:
            Fused feature map.
        """
        sampled_feats = x[:-1]  # all up/down-sampled features
        unsampled_feat = x[-1]  # the feature at current level

        # Context-aware resampling (optional)
        if self.using_resampling:
            context = self.context_conv(torch.cat(x, 1))
            splits = context.split(unsampled_feat.size(1), dim=1)
            sampled_feats = [
                getattr(self, f"resampler{i + 1}")(splits[i], sampled_feats[i])
                for i in range(len(sampled_feats))
            ]

        # Compute attention weights
        weights = [
            getattr(self, f"weight_level_{i + 1}")(feat)
            for i, feat in enumerate(sampled_feats)
        ]
        weights.append(getattr(self, f"weight_level_{len(weights) + 1}")(unsampled_feat))
        levels_weight = torch.cat(weights, 1)
        levels_weight = self.weight_levels(levels_weight)
        levels_weight = torch.sigmoid(levels_weight)

        # Weighted fusion
        fused = unsampled_feat * levels_weight[:, -1:, :, :]
        for i, feat in enumerate(sampled_feats):
            fused = fused + feat * levels_weight[:, i:i + 1, :, :]

        return self.conv(fused)


# ---------------------------------------------------------------------------
# Reassemble (ICA) - Intra-scale Content-Aware Attention
# ---------------------------------------------------------------------------

class Reassemble(nn.Module):
    """Intra-scale Content-Aware Attention for Feature Reassemble (ICA).

    Strengthens intra-scale discriminative feature learning and reassembles
    redundant features based on information content and spatial variation.

    Args:
        out_channels: Number of output channels.
        group_num: Number of groups for GroupNorm.
        gate_threshold: Threshold for the gating mechanism.
        torch_gn: Whether to use torch.nn.GroupNorm (vs custom GroupBatchnorm2d).
        act: Activation function.
        norm: Normalization type after reassembly.
    """

    def __init__(
        self,
        out_channels: int,
        group_num: int = 16,
        gate_threshold: float = 0.5,
        torch_gn: bool = True,
        act=nn.GELU(),
        norm: Optional[Union[bool, str, nn.Module]] = "LN",
    ):
        super().__init__()
        if group_num is None:
            group_num = out_channels

        self.gn = (
            nn.GroupNorm(num_channels=out_channels, num_groups=group_num)
            if torch_gn
            else GroupBatchnorm2d(c_num=out_channels, group_num=group_num)
        )
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()
        self.act = _get_activation(act)
        self.norm = _get_norm(norm, out_channels) if norm else nn.Identity()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1).contiguous()
        reweigts = self.sigmoid(gn_x * w_gamma)

        # Gating: channels above threshold pass through; others are reweighted
        w1 = torch.where(reweigts > self.gate_threshold, torch.ones_like(reweigts), reweigts)
        w2 = torch.where(reweigts > self.gate_threshold, torch.zeros_like(reweigts), reweigts)
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.channel_reassemble(x_1, x_2)
        return self.act(self.norm(y))

    @staticmethod
    def channel_reassemble(x_1, x_2):
        """Reassemble by adding channel-reversed complementary features."""
        return x_1 + x_2.flip(1).contiguous()


# ---------------------------------------------------------------------------
# A3FPN_2, A3FPN_3, A3FPN_4 - Per-level building blocks
# ---------------------------------------------------------------------------

class A3FPN_2(nn.Module):
    """A3-FPN level block for 2-scale feature fusion (adjacent levels)."""

    def __init__(
        self,
        level=0,
        channel=None,
        act=nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
        compress_channel: int = 16,
        group_num: int = 16,
        num_blocks: int = 1,
        expansion: float = 1.0,
        using_resampling: Optional[bool] = False,
        dcn_group: int = 4,
        dcn_config: dict = None,
    ):
        super().__init__()
        if channel is None:
            channel = [64, 128]
        if dcn_config is None:
            dcn_config = {}

        self.level = level
        if self.level == 0:
            self.upsample = SampleBlock(
                channel[1], channel[0], sample="upsample", scale_factor=2, act=act, norm=norm,
            )
        else:
            self.downsample = SampleBlock(
                channel[0], channel[1], sample="downsample", scale_factor=2, act=act, norm=norm,
            )

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=2,
            compress_c=compress_channel, act=act, norm=norm, num_blocks=num_blocks,
            expansion=expansion, using_resampling=using_resampling, dcn_group=dcn_group,
            dcn_config=dcn_config,
        )
        self.ICA = Reassemble(
            out_channels=channel[level], group_num=group_num, act=act, norm="LN",
        )

    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
            out = self.MCA((input2, input1))
        else:
            input1 = self.downsample(input1)
            out = self.MCA((input1, input2))
        return self.ICA(out)


class A3FPN_3(nn.Module):
    """A3-FPN level block for 3-scale feature fusion."""

    def __init__(
        self,
        level=0,
        channel=None,
        act=nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
        compress_channel: int = 16,
        group_num: int = 16,
        num_blocks: int = 1,
        expansion: float = 1.0,
        using_resampling: Optional[bool] = False,
        dcn_group: Union[int, List[int], Tuple[int]] = 4,
        dcn_config: dict = None,
    ):
        super().__init__()
        if channel is None:
            channel = [64, 128, 256]
        if dcn_config is None:
            dcn_config = {}

        self.level = level
        if self.level == 0:
            self.upsample4x = SampleBlock(channel[2], channel[0], sample="upsample", scale_factor=4, act=act, norm=norm)
            self.upsample2x = SampleBlock(channel[1], channel[0], sample="upsample", scale_factor=2, act=act, norm=norm)
        elif self.level == 1:
            self.upsample2x1 = SampleBlock(channel[2], channel[1], sample="upsample", scale_factor=2, act=act, norm=norm)
            self.downsample2x1 = SampleBlock(channel[0], channel[1], sample="downsample", scale_factor=2, act=act, norm=norm)
        elif self.level == 2:
            self.downsample2x = SampleBlock(channel[1], channel[2], sample="downsample", scale_factor=2, act=act, norm=norm)
            self.downsample4x = SampleBlock(channel[0], channel[2], sample="downsample", scale_factor=4, act=act, norm=norm)

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=3,
            compress_c=compress_channel, act=act, norm=norm, num_blocks=num_blocks,
            expansion=expansion, using_resampling=using_resampling, dcn_group=dcn_group,
            dcn_config=dcn_config,
        )
        self.ICA = Reassemble(out_channels=channel[level], group_num=group_num, act=act, norm="LN")

    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            out = self.MCA((input2, input3, input1))
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
            out = self.MCA((input3, input1, input2))
        else:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
            out = self.MCA((input2, input1, input3))
        return self.ICA(out)


class A3FPN_4(nn.Module):
    """A3-FPN level block for 4-scale feature fusion."""

    def __init__(
        self,
        level=0,
        channel=None,
        act=nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
        compress_channel: int = 16,
        group_num: int = 16,
        num_blocks: int = 1,
        expansion: float = 1.0,
        using_resampling: Optional[bool] = False,
        dcn_group: Union[int, List[int], Tuple[int]] = 4,
        dcn_config: dict = None,
    ):
        super().__init__()
        if channel is None:
            channel = [256, 512, 1024, 2048]
        if dcn_config is None:
            dcn_config = {}

        self.level = level
        if self.level == 0:
            self.upsample4x = SampleBlock(channel[2], channel[0], sample="upsample", scale_factor=4, act=act, norm=norm)
            self.upsample2x = SampleBlock(channel[1], channel[0], sample="upsample", scale_factor=2, act=act, norm=norm)
            self.upsample8x = SampleBlock(channel[3], channel[0], sample="upsample", scale_factor=8, act=act, norm=norm)
        elif self.level == 1:
            self.upsample2x1 = SampleBlock(channel[2], channel[1], sample="upsample", scale_factor=2, act=act, norm=norm)
            self.upsample4x1 = SampleBlock(channel[3], channel[1], sample="upsample", scale_factor=4, act=act, norm=norm)
            self.downsample2x1 = SampleBlock(channel[0], channel[1], sample="downsample", scale_factor=2, act=act, norm=norm)
        elif self.level == 2:
            self.upsample2x2 = SampleBlock(channel[3], channel[2], sample="upsample", scale_factor=2, act=act, norm=norm)
            self.downsample2x2 = SampleBlock(channel[1], channel[2], sample="downsample", scale_factor=2, act=act, norm=norm)
            self.downsample4x2 = SampleBlock(channel[0], channel[2], sample="downsample", scale_factor=4, act=act, norm=norm)
        elif self.level == 3:
            self.downsample2x3 = SampleBlock(channel[2], channel[3], sample="downsample", scale_factor=2, act=act, norm=norm)
            self.downsample4x3 = SampleBlock(channel[1], channel[3], sample="downsample", scale_factor=4, act=act, norm=norm)
            self.downsample8x3 = SampleBlock(channel[0], channel[3], sample="downsample", scale_factor=8, act=act, norm=norm)

        self.MCA = Fusion(
            in_channels=channel[level], out_channels=channel[level], num_fusion=4,
            compress_c=compress_channel, act=act, norm=norm, num_blocks=num_blocks,
            expansion=expansion, using_resampling=using_resampling, dcn_group=dcn_group,
            dcn_config=dcn_config,
        )
        self.ICA = Reassemble(out_channels=channel[level], group_num=group_num, act=act, norm="LN")

    def forward(self, x):
        input1, input2, input3, input4 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
            input4 = self.upsample8x(input4)
            out = self.MCA((input2, input3, input4, input1))
        elif self.level == 1:
            input1 = self.downsample2x1(input1)
            input3 = self.upsample2x1(input3)
            input4 = self.upsample4x1(input4)
            out = self.MCA((input3, input4, input1, input2))
        elif self.level == 2:
            input1 = self.downsample4x2(input1)
            input2 = self.downsample2x2(input2)
            input4 = self.upsample2x2(input4)
            out = self.MCA((input4, input2, input1, input3))
        elif self.level == 3:
            input1 = self.downsample8x3(input1)
            input2 = self.downsample4x3(input2)
            input3 = self.downsample2x3(input3)
            out = self.MCA((input3, input2, input1, input4))
        return self.ICA(out)


# ---------------------------------------------------------------------------
# A3FPNBody - Core A3-FPN architecture
# ---------------------------------------------------------------------------

class A3FPNBody(nn.Module):
    """Core A3-FPN body with top-down multi-scale feature fusion.

    This implements the horizontally-spread column network that enables
    asymptotically global feature interaction across all hierarchical levels.
    Supports 3 or 4 input levels.

    Args:
        channels: List of input channels for each scale (length 3 or 4).
        num_levels: Number of input levels (3 or 4).
        act: Activation function.
        norm: Normalization type.
        compress_channel: Compression channels per level for attention.
        group_num: GroupNorm groups per level for ICA.
        num_repblocks: Number of RepVGG blocks in CSP layers.
        expansion: Expansion ratio for CSP layers.
        using_resampling: Whether to enable DCNv4 resampling per fusion scale.
        dcn_groups: DCNv4 group per level.
        dcn_config: Shared DCNv4 configuration.
    """

    def __init__(
        self,
        channels=None,
        num_levels: int = 4,
        act: nn.Module = nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
        compress_channel: Union[List[int], Tuple[int]] = None,
        group_num: Union[List[int], Tuple[int]] = None,
        num_repblocks: int = 1,
        expansion: float = 1.0,
        using_resampling: Union[List[bool], Tuple[bool]] = None,
        dcn_groups: Union[List[int], Tuple[int]] = None,
        dcn_config: dict = None,
    ):
        super().__init__()
        assert num_levels in (3, 4), f"num_levels must be 3 or 4, got {num_levels}"

        if channels is None:
            channels = [128, 256, 512] if num_levels == 3 else [64, 128, 256, 512]
        if compress_channel is None:
            compress_channel = [16, 16, 32] if num_levels == 3 else [16, 16, 32, 32]
        if group_num is None:
            group_num = [16, 16, 32] if num_levels == 3 else [16, 16, 32, 32]
        if using_resampling is None:
            using_resampling = [False, False, False]
        if dcn_groups is None:
            dcn_groups = [16, 16, 32] if num_levels == 3 else [16, 16, 32, 32]
        if dcn_config is None:
            dcn_config = {}

        self.num_levels = num_levels

        def _g(lst, rev_idx):
            """Get parameter from end of list with fallback."""
            i = len(lst) - rev_idx
            return lst[i] if 0 <= i < len(lst) else lst[-1]

        # --- 2-level fusions: always between the deepest 2 levels ---
        self.a3fpn_2_level0 = A3FPN_2(
            level=0, channel=channels[-2:], act=act, norm=norm,
            compress_channel=_g(compress_channel, 2), group_num=_g(group_num, 2),
            num_blocks=num_repblocks, expansion=expansion,
            using_resampling=using_resampling[0], dcn_group=_g(dcn_groups, 2),
            dcn_config=dcn_config,
        )
        self.a3fpn_2_level1 = A3FPN_2(
            level=1, channel=channels[-2:], act=act, norm=norm,
            compress_channel=_g(compress_channel, 1), group_num=_g(group_num, 1),
            num_blocks=num_repblocks, expansion=expansion,
            using_resampling=using_resampling[0], dcn_group=_g(dcn_groups, 1),
            dcn_config=dcn_config,
        )

        # --- 3-level fusions ---
        for lvl in range(3):
            setattr(self, f"a3fpn_3_level{lvl}", A3FPN_3(
                level=lvl, channel=channels[-3:], act=act, norm=norm,
                compress_channel=_g(compress_channel, 3 - lvl),
                group_num=_g(group_num, 3 - lvl),
                num_blocks=num_repblocks, expansion=expansion,
                using_resampling=using_resampling[1],
                dcn_group=_g(dcn_groups, 3 - lvl),
                dcn_config=dcn_config,
            ))

        # --- 4-level fusions (only when num_levels >= 4) ---
        if num_levels >= 4:
            for lvl in range(4):
                setattr(self, f"a3fpn_4_level{lvl}", A3FPN_4(
                    level=lvl, channel=channels[-4:], act=act, norm=norm,
                    compress_channel=_g(compress_channel, 4 - lvl),
                    group_num=_g(group_num, 4 - lvl),
                    num_blocks=num_repblocks, expansion=expansion,
                    using_resampling=using_resampling[2],
                    dcn_group=_g(dcn_groups, 4 - lvl),
                    dcn_config=dcn_config,
                ))

    def forward(self, x):
        """Forward pass: top-down multi-scale fusion.

        Args:
            x: Tuple of 3 or 4 feature maps from backbone.

        Returns:
            Tuple of fused feature maps at same scales.
        """
        if self.num_levels == 3:
            x0, x1, x2 = x

            # 2-level: P4 <-> P5
            out_p4 = self.a3fpn_2_level0((x1, x2))  # x2 up → P4 scale
            x2_new = self.a3fpn_2_level1((x1, x2))  # x1 down → P5 scale

            # 3-level: P3, out_p4, x2_new
            x0_out = self.a3fpn_3_level0((x0, out_p4, x2_new))  # → P3
            x1_out = self.a3fpn_3_level1((x0, out_p4, x2_new))  # → P4
            x2_out = self.a3fpn_3_level2((x0, out_p4, x2_new))  # → P5

            return x0_out, x1_out, x2_out

        else:  # num_levels == 4
            x0, x1, x2, x3 = x

            # 2-level: P5 <-> P6
            output2 = self.a3fpn_2_level0((x2, x3))
            x3_new = self.a3fpn_2_level1((x2, x3))

            # 3-level: P4, output2, x3_new
            output1 = self.a3fpn_3_level0((x1, output2, x3_new))
            x2_new = self.a3fpn_3_level1((x1, output2, x3_new))
            x3_new = self.a3fpn_3_level2((x1, output2, x3_new))

            # 4-level: P3, output1, x2_new, x3_new
            x0_out = self.a3fpn_4_level0((x0, output1, x2_new, x3_new))
            x1_out = self.a3fpn_4_level1((x0, output1, x2_new, x3_new))
            x2_out = self.a3fpn_4_level2((x0, output1, x2_new, x3_new))
            x3_out = self.a3fpn_4_level3((x0, output1, x2_new, x3_new))

            return x0_out, x1_out, x2_out, x3_out


# ---------------------------------------------------------------------------
# A3FPN - Main neck module for Ultralytics integration
# ---------------------------------------------------------------------------

class A3FPN(nn.Module):
    """A3-FPN Neck for YOLO-style models.

    Replaces the standard PANet/FPN neck with the A3-FPN asymptotic
    content-aware pyramid attention network. Accepts backbone features
    and outputs fused multi-scale features for the detection head.

    Args:
        in_channels: Input channels from backbone (e.g., [256, 512, 1024] for P3/4/5).
        out_channels: Output channels for each scale.
        num_outs: Number of output scales (default matches input).
        squeeze: Channel squeeze ratio per input level.
        act: Activation function.
        norm: Normalization type ("BN", "SYNCBN", "LN", etc.).
        compress_channel: Compression channels per level for attention.
        group_num: GroupNorm groups per level for ICA.
        num_repblocks: Number of RepVGG blocks in CSP layers.
        expansion: Expansion ratio for CSP layers.
        using_resampling: Whether to use DCNv4 resampling.
        dcn_groups: DCNv4 group parameter per level.
        dcn_config: DCNv4 configuration dictionary.
        end_level: Number of output levels (can be > num_outs for extra downsampling).
        init_weights: Whether to initialize weights.

    Example:
        >>> neck = A3FPN(in_channels=[256, 512, 1024], out_channels=256)
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = neck(x)
        >>> print([o.shape for o in outputs])
        [torch.Size([1, 256, 80, 80]), torch.Size([1, 256, 40, 40]), torch.Size([1, 256, 20, 20])]
    """

    def __init__(
        self,
        in_channels=None,
        out_channels=256,
        num_outs=None,
        squeeze: Union[List[int], int] = 1,
        act=nn.GELU(),
        norm: Union[str, bool, nn.Module] = "BN",
        compress_channel: Union[List[int], Tuple[int]] = None,
        group_num: Union[List[int], Tuple[int]] = None,
        num_repblocks: int = 1,
        expansion: float = 1.0,
        using_resampling: Union[bool, List[bool], Tuple[bool]] = False,
        dcn_groups: Union[int, List[int], Tuple[int]] = 4,
        dcn_config: dict = None,
        end_level: int = None,
        init_weights: bool = True,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024, 2048]
        if num_outs is None:
            num_outs = len(in_channels)
        if end_level is None:
            end_level = num_outs
        assert end_level >= num_outs, "end_level must be >= num_outs"
        if dcn_config is None:
            dcn_config = {}

        n_levels = len(in_channels)
        assert n_levels in (3, 4), f"in_channels must have length 3 or 4, got {n_levels}"

        # Broadcast scalar parameters to per-level lists
        if isinstance(squeeze, int):
            squeeze = [squeeze] * n_levels
        if isinstance(dcn_groups, int):
            dcn_groups = [dcn_groups] * n_levels
        if isinstance(using_resampling, bool):
            using_resampling = [using_resampling] * 3  # 2/3/4-level fusion flags

        # Default per-level parameters
        if compress_channel is None:
            compress_channel = [16] * n_levels
        if group_num is None:
            group_num = [16] * n_levels

        in_channels_reduced = [ch // squeeze[i] for i, ch in enumerate(in_channels)]

        self.num_outs = num_outs
        self.end_level = end_level
        self.n_levels = n_levels

        # Input squeeze convolutions
        for i in range(n_levels):
            setattr(
                self, f"conv{i}",
                A3Conv(in_channels[i], in_channels_reduced[i], 1, act=act, norm=norm, bias=False)
                if squeeze[i] != 1 else nn.Identity(),
            )

        # Main A3-FPN body (supports both 3-level and 4-level)
        self.a3fpn_body = A3FPNBody(
            channels=in_channels_reduced,
            num_levels=n_levels,
            act=act, norm=norm,
            compress_channel=compress_channel,
            group_num=group_num,
            num_repblocks=num_repblocks, expansion=expansion,
            using_resampling=using_resampling,
            dcn_groups=dcn_groups,
            dcn_config=dcn_config,
        )

        # Output projection convolutions
        for i in range(n_levels):
            setattr(
                self, f"conv_out{i}",
                A3Conv(in_channels_reduced[i], out_channels, 3, p=1, act=act, norm=norm),
            )

        # Extra downsampling for additional output levels
        self.num_outs = num_outs
        self.end_level = end_level
        self.n_levels = n_levels

        # Need extra downsampling if we want more outputs than input levels
        if num_outs > n_levels:
            self.conv_down = A3Conv(out_channels, out_channels, 3, 2, 1, act=act, norm=norm)
        if end_level > num_outs:
            self.extra_convs = nn.ModuleList()
            for i in range(self.end_level - self.num_outs - 1):
                self.extra_convs.append(A3Conv(out_channels, out_channels, 3, 2, 1, act=act, norm=norm))

        if init_weights:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """Initialize weights for linear and normalization layers."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: List of feature maps from backbone, ordered from largest to smallest
               (e.g., [P3, P4, P5] or [P3, P4, P5, P6]).

        Returns:
            List of output feature maps at each scale.
        """
        assert len(x) == self.n_levels, f"Expected {self.n_levels} input features, got {len(x)}"

        # Apply squeeze convolutions
        squeezed = [getattr(self, f"conv{i}")(x[i]) for i in range(self.n_levels)]

        # A3-FPN body forward (handles both 3 and 4 levels)
        outs = self.a3fpn_body(tuple(squeezed))

        # Output projection
        outs = [getattr(self, f"conv_out{i}")(outs[i]) for i in range(self.n_levels)]

        # Extra downsampling if needed
        if len(outs) < self.num_outs:
            outs.append(self.conv_down(outs[-1]))
        if hasattr(self, "extra_convs"):
            for conv in self.extra_convs:
                outs.append(conv(outs[-1]))

        return outs[:self.end_level]

    def deploy(self):
        """Convert to deployment mode (fuse RepVGG blocks)."""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
