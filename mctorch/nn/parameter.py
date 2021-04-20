import torch
import weakref
from collections import OrderedDict


class Parameter(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, manifold=None):
        if data is None:
            if manifold is not None:
                data = manifold.rand()
            else:
                data = torch.Tensor()
        return torch.nn.Parameter._make_subclass(cls, data, requires_grad)

    def __init__(self, data=None, requires_grad=True, manifold=None):
        self._manifold = manifold
        self._rgrad = None
        if manifold is not None:
            assert manifold.size() == self.size()
            self.register_rgrad_hook()

    def register_rgrad_hook(self):
        weak_self = weakref.ref(self)

        def calculate_rgrad(grad):
            var = weak_self()
            if var is None or var._manifold is None:
                return
            var._rgrad = var._manifold.egrad2rgrad(self.data, grad)

        self.register_hook(calculate_rgrad)

    @property
    def manifold(self):
        return self._manifold

    @property
    def rgrad(self):
        if self._manifold is not None:
            return self._rgrad
