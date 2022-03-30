import numpy as np
import torch
from torch import nn


class RemoveDim(nn.Module):
    def __init__(self, dim):
        super(RemoveDim, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        return remove_dim(input_tensor, self.dim)


def remove_dim_get_shape(curr_shape, dim):
    assert dim > 0, "Axis must be greater than 0"
    curr_shape = list(curr_shape)
    axis_shape = curr_shape.pop(dim)
    curr_shape[dim - 1] *= axis_shape
    return curr_shape


def remove_dim(input_tensor, dim):
    curr_shape = list(input_tensor.shape)
    if type(dim) == int:
        if dim < 0:
            dim = len(curr_shape) + dim
        new_shape = remove_dim_get_shape(curr_shape, dim)
    else:
        dim = [dd if dd >= 0 else len(curr_shape) + dd for dd in dim]
        assert len(np.unique(dim)) == len(dim), "Repeated dims are not allowed"
        for ax in sorted(dim, reverse=True):
            curr_shape = remove_dim_get_shape(curr_shape, ax)
        new_shape = curr_shape
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor.view(new_shape)
    else:
        return input_tensor.reshape(new_shape)
