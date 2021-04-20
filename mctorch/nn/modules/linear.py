import math

import torch
from torch.nn import Linear, init
from torch.nn import functional as F

from ..parameter import Parameter
from ..manifolds import create_manifold_parameter, manifold_random_


class rLinear(torch.nn.Linear):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True,
                 weight_manifold=None, transpose_flag=False):
        super(rLinear, self).__init__(in_features, out_features, bias=bias)
        
        self.weight_manifold = weight_manifold
        self.transpose_flag = transpose_flag
        self.weight_transform = lambda x : x

        if weight_manifold is None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            self.transpose_flag, self.weight = create_manifold_parameter(
                weight_manifold, (out_features, in_features), transpose_flag)
            if self.transpose_flag:
                self.weight_transform = lambda x : x.transpose(-2, -1)

        self.local_reset_parameters()

    def local_reset_parameters(self):
        if self.weight_manifold is not None:
            manifold_random_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_transform(self.weight))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight_transform(self.weight), self.bias)
