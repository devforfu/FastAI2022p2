from functools import singledispatch
from enum import Enum, auto
import torch
import matplotlib.pyplot as plt
import numpy as np


@singledispatch
def view(tensor):
    pass


@view.register
def view(tensor: torch.Tensor, axes=None):
    viewer = {
        TensorLayout.RGBChannelFirst: _view_chw,
        TensorLayout.RGBChannelLast: _view_hwc,
        TensorLayout.GrayscaleChannelFirst: _view_1hw,
        TensorLayout.GrayscaleChannelLast: _view_hw1,
        TensorLayout.Grayscale: _view_hw,
        TensorLayout.MultiChannel: _view_grid_hw,
    }.get(get_tensor_layout(tensor))
    assert viewer is not None, f"tensor shape is not supported: {tensor.shape}"
    return viewer(tensor, axes=axes)


class TensorLayout(Enum):
    RGBChannelFirst = auto()
    RGBChannelLast = auto()
    GrayscaleChannelFirst = auto()
    GrayscaleChannelLast = auto()
    Grayscale = auto()
    MultiChannel = auto()
    Unknown = auto()


def get_tensor_layout(tensor: torch.Tensor) -> TensorLayout:
    if tensor.ndim == 3:
        shape = tensor.shape
        if shape[0] == 1:
            return TensorLayout.GrayscaleChannelFirst
        elif shape[0] == 3:
            return TensorLayout.RGBChannelFirst
        elif shape[-1] == 1:
            return TensorLayout.GrayscaleChannelLast
        elif shape[-1] == 3:
            return TensorLayout.RGBChannelLast
        else:
            return TensorLayout.MultiChannel
    elif tensor.ndim == 2:
        return TensorLayout.Grayscale
    return TensorLayout.Unknown


def _view_chw(tensor: torch.Tensor, axes=None):
    c, h, w = tensor.shape
    _view_hwc(tensor.view((h, w, c)), axes=axes)


def _view_hwc(tensor: torch.Tensor, axes=None):
    _show_one(tensor, axes=axes)


def _view_1hw(tensor: torch.Tensor, axes=None):
    _show_one(tensor.squeeze(0), axes=axes)


def _view_hw1(tensor: torch.Tensor, axes=None):
    _show_one(tensor.squeeze(2), axes=axes)


def _view_hw(tensor: torch.Tensor, axes=None):
    _show_one(tensor, cmap="gray", axes=axes)


def _show_one(tensor: torch.Tensor, axes=None, **kwargs):
    if axes is None:
        _, ax = plt.subplots(1, 1)
    elif isinstance(axes, np.ndarray):
        ax = axes[0]
    else:
        ax = axes
    ax.imshow(tensor, **kwargs)


def _view_grid_hw(tensor: torch.Tensor, axes=None):
    n_channels = tensor.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    if axes is None:
        _, axes = plt.subplots(n_rows, n_cols)
    else:
        assert isinstance(axes, np.ndarray), "grid expects an array of axes"
        assert len(axes) == tensor.shape[0], "grid should have one axis per channel"
    for channel, ax in zip(tensor, axes.flat):
        view(channel, axes=ax)
