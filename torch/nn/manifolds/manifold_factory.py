from .stiefel import Stiefel
from ..parameter import Parameter


#TODO: create factory for each class of manifold, so when adding new manifold user can add new factory and register that
def create_manifold_parameter(manifold, shape, transpose=False):
    if manifold is Stiefel:
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
            # use the argument only in case when it's vague
            to_transpose = transpose
            to_return = Parameter(manifold=Stiefel(w, h, k=k))

        return to_transpose, to_return

    else:
        raise NotImplementedError
