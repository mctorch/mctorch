# coding=utf-8
import math
import warnings

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch._jit_internal import List, Optional

from .utils import _multiply_tuple
from ..manifolds import create_manifold_parameter, manifold_random_


class _rConvNd(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, weight_manifold, transpose_flag):
        super(_rConvNd, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, padding_mode)
        
        self.weight_manifold = weight_manifold
        self.transpose_flag = transpose_flag
        self.weight_transform = lambda x : x
        self._init_weight_matrix()

        self.local_reset_parameters()

    def local_reset_parameters(self):
        n = self.in_channels
        if self.weight_manifold is not None:
            manifold_random_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_transform(self.weight))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _init_weight_matrix(self):
        if self.weight_manifold is not None:
            kernel_mult = _multiply_tuple(self.kernel_size)
            if self.transposed:
                weight_shape = (self.in_channels, (self.out_channels // self.groups) * kernel_mult)
                kernel_shape = (self.in_channels, self.out_channels // self.groups, *self.kernel_size)
            else:
                weight_shape = (self.out_channels, (self.in_channels // self.groups) * kernel_mult)
                kernel_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)

            self.transpose_flag, self.weight = create_manifold_parameter(
                self.weight_manifold, weight_shape, self.transpose_flag)
            if self.transpose_flag:
                self.weight_transform = lambda x : x.transpose(-2, -1).view(*kernel_shape)
            else:
                self.weight_transform = lambda x : x.view(*kernel_shape)

class rConv1d(_rConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', weight_manifold=None, transpose_flag=False):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(rConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode, weight_manifold, transpose_flag)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            self.weight_transform(self.weight), self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, self.weight_transform(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class rConv2d(_rConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', weight_manifold=None, transpose_flag=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(rConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, weight_manifold, transpose_flag)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight_transform(self.weight))

class rConv3d(_rConvNd):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', weight_manifold=None, transpose_flag=False):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(rConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, padding_mode, weight_manifold, transpose_flag)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            self.weight_transform(self.weight), self.bias, self.stride, _triple(0),
                            self.dilation, self.groups)
        return F.conv3d(input, self.weight_transform(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)