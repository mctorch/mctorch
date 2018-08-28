from .stiefel import Stiefel
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
        return ManifoldShapeFactory.factories[manifold].create(
                                                    shape, transpose)

    def __init__(self, manifold):
        self.manifold = manifold
        ManifoldShapeFactory._addFactory(manifold, self)

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
            raise ValueError("Invalid shape %s, length of shape"
                             "tuple should be 2 or 3" % (shape,))

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


create_manifold_parameter = ManifoldShapeFactory.create_manifold_parameter
StiefelLikeFactory(Stiefel)
