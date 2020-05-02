import torch

from ..utils.manifold_multi import multiprod, multitransp, multisym
from .manifold import Manifold


class PositiveDefinite(Manifold):
    """
    Class for Positive Definite manifold with shape (k x height x width)
    or (height x width)

    With k > 1 it applies product of k Positive Definites
    """

    def __init__(self, n, k=1):
        super(Manifold, self).__init__()
        if n < 1:
            raise ValueError("Need n >= 1. Value supplied was n = {}.".format(n))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = {}.".format(k))
        # Set the dimensions of the Positive Definite
        self._n = n
        self._k = k

        # Set dimension
        self._dim = self._k * 0.5 * self._n * (self._n + 1)
        if k == 1:
            self._size = torch.Size((n, n))
        else:
            self._size = torch.Size((k, n, n))

    def __str__(self):
        if self._k == 1:
            return "Positive Definite manifold ({}, {}) matrix".format(
                self._n, self._n)
        elif self._k >= 2:
            return ("Product Positive Definite manifold {} ({}, {}) "
                    "matrices").format(self._k, self._n, self._n)

    def rand(self):
        """
        Generate random positive definite point
        """
        if self._k == 1:
            d = torch.ones(self._n, 1) + torch.rand(self._n, 1)
            X = torch.randn(self._n, self._n)
            q, _ = torch.qr(X)
        else:
            d = torch.ones(self._k, self._n, 1) + torch.rand(self._k, self._n, 1)
            X = torch.zeros(self._k, self._n, self._n)
            for i in range(self._k):
                X[i], _ = torch.qr(torch.randn(self._n, self._n))

        return multiprod(X, d * multitransp(X))

    def proj(self, X, U):
        return multisym(U)

    def egrad2rgrad(self, X, U):
        return multiprod(multiprod(X, multisym(U)), X)

    def inner(self, X, G1, G2):
        G1s, _ = torch.solve(G1, X)
        G2s, _ = torch.solve(G2, X)
        return torch.sum(G1s * G2s)

    def retr(self, X, G):
        """
        Retract to positive definite
        """
        X_inv_G, _ = torch.solve(G, X)
        return multisym(X + G + .5 * multiprod(G, X_inv_G))

    def ehess2rhess(self, X, egrad, ehess, H):
        # Directional derivatives of the Riemannian gradient
        Hess = multiprod(X, multiprod(multisym(ehess), X))
        Hess += 2 * multisym(multiprod(H, multiprod(multisym(egrad), X)))

        # Correction factor for the non-constant metric
        Hess -= multisym(multiprod(H, multiprod(multisym(egrad), X)))
        return Hess

    def norm(self, X, G):
        return torch.norm(torch.solve(G, X)[0])

    def randvec(self, X):
        U = torch.randn(*X.size())
        U = self.proj(X, U)
        U = U / self.norm(X, U)
        return U

    def transp(self, x1, x2, d):
        return d
