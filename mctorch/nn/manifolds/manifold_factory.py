import torch

from .stiefel import Stiefel
from .positive_definite import PositiveDefinite
from .euclidean import Euclidean
from .hyperbolic import Hyperbolic
from .doublystochastic import DoublyStochastic
from ..parameter import Parameter

class ManifoldShapeFactory(object):
    """
    Base class for manifold shape factory. This is used by torch
    modules to determine shape of Manifolds give shape of weight matrix

    For each Manifold type it takes shape and whether to transpose the
    tensor when shape is vague (in instances when both transpose and normal
    are valid).

    To register a new factory implement a new subclass and create its object
    with manifold as parameter

    """
    factories = {}

    @staticmethod
    def _addFactory(manifold, factory):
        ManifoldShapeFactory.factories[manifold] = factory

    @staticmethod
    def create_manifold_parameter(manifold, shape, transpose=False):
        if manifold not in ManifoldShapeFactory.factories:
            raise NotImplementedError
        return ManifoldShapeFactory.factories[manifold].create(shape, transpose)

    def __init__(self, manifold):
        self.manifold = manifold
        ManifoldShapeFactory._addFactory(manifold, self)

    # TODO: change return of create to manifold param and modified tensor if any
    def create(self, shape, transpose=False):
        raise NotImplementedError


class StiefelLikeFactory(ManifoldShapeFactory):
    """
    Stiefel like factory implements shape factory where tensor constrains are
    similar to that of Stiefel.

    Constraints:
        if 3D tensor (k,h,w):
            k > 1 and h > w > 1
        if 2D tensor (h,w):
            h > w > 1
    in case of h == w both normal and tranpose are valid and user has
    flexibility to choose if he wants (h x w) or (w x h) as manifold
    """
    def create(self, shape, transpose=False):
        if len(shape) == 3:
            k, h, w = shape
        elif len(shape) == 2:
            k = 1
            h, w = shape
        else:
            raise ValueError(("Invalid shape {}, length of shape "
                             "tuple should be 2 or 3").format(shape))

        to_transpose = transpose
        to_return = None
        if h > w:
            to_transpose = False
            to_return = Parameter(manifold=self.manifold(h, w, k=k))
        elif h < w:
            to_transpose = True
            to_return = Parameter(manifold=self.manifold(w, h, k=k))
        elif h == w:
            # use this argument only in case when shape is vague
            to_transpose = transpose
            to_return = Parameter(manifold=self.manifold(w, h, k=k))

        return to_transpose, to_return


class SquareManifoldFactory(ManifoldShapeFactory):
    """
    Manifold shape factory for manifold constrained parameter which
    allows only for square shapes. For example PositiveDefinite manifold

    Constraints:
        if 3D tensor (k,n,n):
            k > 1 and n > 1
        if 2D tensor (n,n):
            n > 1
    """
    def create(self, shape, transpose=False):
        if len(shape) == 3:
            k, n, m = shape
        elif len(shape) == 2:
            k = 1
            n, m = shape
        else:
            raise ValueError(("Invalid shape {}, length of shape"
                             "tuple should be 2 or 3").format(shape))
        if n != m:
            raise ValueError(("Invalid shape {}  dimensions should "
                             "be equal").format(shape))
        return transpose, Parameter(manifold=self.manifold(n=n, k=k))


class EuclideanManifoldFactory(ManifoldShapeFactory):
    """
    Manifold factory fro euclidean just initializes parameter manifold with
    shape parameter of create without transpose
    """
    def create(self, shape, transpose=False):
        if len(shape) == 0:
            raise ValueError("Shape length cannot be 0")
        else:
            return transpose, Parameter(manifold=self.manifold(*shape))

class HyperbolicManifoldFactory(ManifoldShapeFactory):
    """
    Manifold factory for hyperbolic manifold
    shape parameter of create without transpose
    """
    def create(self, shape, transpose=False):
        if len(shape) == 1:
            k, n = 1, shape
        elif len(shape) == 2:
            k, n = shape
        else:
            raise ValueError(("Invalid shape {}, length of shape"
                             "tuple should be 1 or 2").format(shape))
        return transpose, Parameter(manifold=self.manifold(n=n, k=k))


class DSManifoldFactory(ManifoldShapeFactory):
    """
    Manifold factory for DoublyStochastic manifold
    """
    def create(self, shape, transpose=False):
        given = len(shape)
        allowed = [2, 3]
        assert given in allowed, ValueError(f"Shape should be in {allowed}")

        n, m = shape[-2], shape[-1]
        k = given == allowed[1] and shape[0] or 1

        return transpose, Parameter(manifold=self.manifold(n=n, m=m, k=k))


create_manifold_parameter = ManifoldShapeFactory.create_manifold_parameter
StiefelLikeFactory(Stiefel)
SquareManifoldFactory(PositiveDefinite)
EuclideanManifoldFactory(Euclidean)
HyperbolicManifoldFactory(Hyperbolic)
DSManifoldFactory(DoublyStochastic)

def manifold_random_(tensor):
    if not hasattr(tensor, 'manifold') or tensor.manifold is None:
        return tensor

    with torch.no_grad():
        return tensor.copy_(tensor.manifold.rand())
