import torch

from ..utils.manifold_multi import multiprod, multitransp, multisym
from .manifold import Manifold


class Stiefel(Manifold):
    """
    Class for Stiefel manifold with shape (k x height x width)
    or (height x width)
    """

    def __init__(self, height, width, k=1):
        if height < width or width < 1:
            raise ValueError("Need height >= width >= 1. Values supplied were"
                             "height = %d and width = %d." % (height, width))
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)
        # Set the dimensions of the Stiefel
        self._n = height
        self._p = width
        self._k = k

        # Set dimension
        self._dim = self._k * (self._n * self._p
                               - 0.5 * self._p * (self._p + 1))
        if k == 1:
            self._size = torch.Size((height, width))
        else:
            self._size = torch.Size((k, height, width))

    def __str__(self):
        if self._k == 1:
            return "Stiefel manifold St(%d, %d)" % (self._n, self._p)
        elif self._k >= 2:
            return "Product Stiefel manifold St(%d, %d)^%d" % (
                self._n, self._p, self._k)

    def rand(self):
        """
        Generate random Stiefel point using qr of random normally distributed
        matrix
        """
        if self._k == 1:
            X = torch.randn(self._n, self._p)
            q, r = torch.qr(X)
            return q

        X = torch.zeros((self._k, self._n, self._p))

        # TODO: update with batch implementation
        for i in range(self._k):
            X[i], r = torch.qr(torch.randn(self._n, self._p))
        return X

    def proj(self, X, U):
        return U - multiprod(X, multisym(multiprod(multitransp(X), U)))

    def retr(self, X, G):
        """
        Retract to the Stiefel using the qr decomposition of X + G.
        """
        if self._k == 1:
            # Calculate 'thin' qr decomposition of X + G
            q, r = torch.qr(X + G)
            # Unflip any flipped signs
            XNew = torch.matmul(q, torch.diag(
                torch.sign(torch.sign(torch.diag(r))+.5)))
        else:
            XNew = X + G
            # TODO: update with batch implementation
            for i in range(self._k):
                q, r = torch.qr(XNew[i])
                XNew[i] = torch.matmul(q, torch.diag(
                    torch.sign(torch.sign(torch.diag(r))+.5)))
        return XNew

    def ehess2rhess(self, X, egrad, ehess, H):
        # Convert Euclidean into Riemannian Hessian.
        XtG = multiprod(multitransp(X), egrad)
        symXtG = multisym(XtG)
        HsymXtG = multiprod(H, symXtG)
        return self.proj(X, ehess - HsymXtG)

    def norm(self, X, G):
        """
        Norm on the tangent space of the Stiefel is simply the Euclidean
        norm.
        """
        return torch.norm(G)

    def randvec(self, X):
        U = torch.randn(*X.size())
        U = self.proj(X, U)
        U = U / torch.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)
