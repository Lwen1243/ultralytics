# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DCNv4 - Deformable Convolution v4 for Ultralytics YOLO.

This module provides the DCNv4 operators used by A3-FPN's context-aware resampling.
The CUDA extension must be compiled separately. See setup instructions below.

Usage:
    from ultralytics.nn.modules.dcnv4 import DCNv4

Setup (CUDA required):
    cd ultralytics/nn/modules/dcnv4
    python setup.py install

Original: https://github.com/OpenGVLab/DCNv4
"""

from .functions import DCNv4Function, FlashDeformAttnFunction
from .modules import DCNv4, FlashDeformAttn

__all__ = ("DCNv4", "FlashDeformAttn", "DCNv4Function", "FlashDeformAttnFunction")
