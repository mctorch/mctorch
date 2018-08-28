from .stiefel import Stiefel
from ..parameter import Parameter


class ManifoldShapeFactory(object):
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
            to_return = Parameter(manifold=Stiefel(h, w, k=k))
        elif h < w:
            to_transpose = True
            to_return = Parameter(manifold=Stiefel(w, h, k=k))
        elif h == w:
            # use this argument only in case when shape is vague
            to_transpose = transpose
            to_return = Parameter(manifold=Stiefel(w, h, k=k))

        return to_transpose, to_return


create_manifold_parameter = ManifoldShapeFactory.create_manifold_parameter
StiefelLikeFactory(Stiefel)
