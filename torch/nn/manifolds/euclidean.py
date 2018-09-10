import torch

from .manifold import Manifold


class Euclidean(Manifold):
    """
    Class for Euclidean manifold with shape: shape
    """

    def __init__(self, *shape):
        if len(shape) <= 0:
            raise ValueError("Need shape parameters.")

        super(Manifold, self).__init__()
        self._dim = 1
        for s in shape:
            self._dim *= s
        self._size = torch.Size(shape)

    def __str__(self):
        return "Euclidean manifold of {} shape".format(self._size)

    def rand(self):
        """
        Generate random tensor
        """
        return torch.randn(*self._size)

    def proj(self, X, U):
        return U

    def inner(self, X, G1, G2):
        return torch.sum(G1 * G2)

    def retr(self, X, G):
        """
        Retraction on euclidean is X + G
        """
        return X + G

    def ehess2rhess(self, X, egrad, ehess, H):
        return ehess

    def norm(self, X, G):
        return torch.norm(G)

    def randvec(self, X):
        U = self.rand()
        return U / self.norm(X, U)

    def transp(self, x1, x2, d):
        return d
