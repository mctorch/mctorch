import torch
from collections import OrderedDict
import weakref


class Parameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """
    def __new__(cls, data=None, requires_grad=True, manifold=None):
        if data is None:
            if manifold is not None:
                data = manifold.rand()
            else:
                data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result
    def __init__(self, data=None, requires_grad=True, manifold=None):
        self._manifold = manifold
        self._rgrad = None
        if manifold is not None:
            assert manifold.size() == self.size()
            self.register_rgrad_hook()

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )

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
