from __future__ import annotations

from functools import singledispatch
from dataclasses import dataclass, field, fields
from enum import Enum, auto
import torch
import matplotlib.pyplot as plt
import numpy as np


@singledispatch
def view(tensor):
    pass


@view.register
def view(tensor: torch.Tensor, axes=None, **kwargs):
    tensor = to_show(tensor)
    viewer = TensorLayout.detect(tensor)
    assert viewer is not None, f"tensor shape is not supported: {tensor.shape}"
    return viewer(tensor, axes=axes, axes_params=AxesParams.from_kwargs(kwargs))
        

def to_show(tensor: torch.Tensor):
    if tensor.device != "cpu":
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor


class TensorLayout(Enum):
    RGBChannelFirst = auto()
    RGBChannelLast = auto()
    GrayscaleChannelFirst = auto()
    GrayscaleChannelLast = auto()
    Grayscale = auto()
    MultiChannel = auto()
    Unknown = auto()
    
    @classmethod
    def detect(cls, tensor: torch.Tensor) -> TensorLayout|None:
        return {
            TensorLayout.RGBChannelFirst: _view_chw,
            TensorLayout.RGBChannelLast: _view_hwc,
            TensorLayout.GrayscaleChannelFirst: _view_1hw,
            TensorLayout.GrayscaleChannelLast: _view_hw1,
            TensorLayout.Grayscale: _view_hw,
            TensorLayout.MultiChannel: _view_grid_hw,
        }.get(get_tensor_layout(tensor))
    

@dataclass
class AxesParams:
    visible: bool = True
    titles: list[str] = field(default_factory=list)
    
    @classmethod
    def from_kwargs(cls, kwargs: dict, prefix: str = "axes_") -> "AxesParams":
        expected = {f.name for f in fields(cls)}
        params = {}
        for key, value in kwargs.items():
            key = key.removeprefix(prefix)
            if key in expected:
                params[key] = value
        return AxesParams(**params)
        

def private_attr(name: str) -> bool:
    return name.startswith("_")
        

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


def _view_chw(tensor: torch.Tensor, axes_params: AxesParams, axes=None):
    c, h, w = tensor.shape
    _view_hwc(tensor.view((h, w, c)), axes_params=axes_params, axes=axes)


def _view_hwc(tensor: torch.Tensor, axes_params: AxesParams, axes=None):
    _show_one(tensor, axes_params=axes_params, axes=axes)


def _view_1hw(tensor: torch.Tensor, axes_params: AxesParams, axes=None):
    _show_one(tensor.squeeze(0), axes_params=axes_params, axes=axes)


def _view_hw1(tensor: torch.Tensor, axes_params: AxesParams, axes=None):
    _show_one(tensor.squeeze(2), axes_params=axes_params, axes=axes)


def _view_hw(tensor: torch.Tensor, axes_params: AxesParams, axes=None):
    _show_one(tensor, cmap="gray", axes_params=axes_params, axes=axes)


def _show_one(tensor: torch.Tensor, axes_params: AxesParams, axes=None, **kwargs):
    if axes is None:
        _, ax = plt.subplots(1, 1)
    elif isinstance(axes, np.ndarray):
        ax = axes[0]
    else:
        ax = axes
    ax.imshow(tensor, **kwargs)
    if axes_params.visible:
        ax.set_axis_on()
    else:
        ax.set_axis_off()


def _view_grid_hw(tensor: torch.Tensor, axes_params: AxesParams, axes=None):
    n_channels = tensor.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))
    if axes is None:
        _, axes = plt.subplots(n_rows, n_cols)
    else:
        assert isinstance(axes, np.ndarray), "grid expects an array of axes"
        assert len(axes) == tensor.shape[0], "grid should have one axis per channel"
    for ax in axes.flat:
        ax.set_axis_off()
    titles = axes_params.titles or []
    for i, (channel, ax) in enumerate(zip(tensor, axes.flat)):
        TensorLayout.detect(channel)(channel, axes=ax, axes_params=axes_params)
        if i < len(titles):
            ax.set_title(titles[i])
