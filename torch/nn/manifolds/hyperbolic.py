import torch

from ..utils.manifold_multi import multiprod, multitransp, multisym
from .manifold import Manifold


class Hyperbolic(Manifold):
    """
    Class for Hyperbolic manifold with shape (k x N) or N

    With k > 1 it applies product of k Hyperbolas
    """

    def __init__(self, n, k=1):
        if n < 2:
            raise ValueError("Need n >= 2 Value supplied was n = {}".format(n))
        if k < 1:
            raise ValueError("Need k >= 1 Value supplied was k = {}".format(k))

        super(Manifold, self).__init__()
        # Set the dimensions of the Hyperbolic manifold
        self._n = n
        self._k = k
        self.eps = 0

        # Set dimension #TODO: confirm
        self._dim = self._k * (self._n)
        if k == 1:
            self._size = torch.Size((n,))
        else:
            self._size = torch.Size((k, n))

    def __str__(self):
        if self._k == 1:
            return "Hyperbolic manifold ({})".format(self._n)
        elif self._k >= 2:
            return "Product Hyperbolic manifold ({})^{}".format(
                self._n, self._k)

    def rand(self):
        """
        Generate random Hyperbolic point in range (-0.001, 0.001)
        """
        u_range = (-0.001, 0.001)
        if self._k == 1:
            X = torch.randn(self._n)# * (u_range[1] - u_range[0]) + u_range[0]
            X[0] = torch.sqrt(1 + torch.sum(X[1:]**2))
            return X

        X = torch.randn(self._k, self._n)# * (u_range[1] - u_range[0]) + u_range[0]
        X[:, 0] = torch.sqrt(1 + torch.sum(X[:, 1:]**2, dim=1))
        return X

    def _lorentz_scalar_product(self, u, v):
        if u.shape == v.shape:
            if len(v.shape) == 1:
                val = torch.sum(u*v) - 2*u[0]*v[0]
                return val
            elif len(v.shape) == 2:
                val = torch.sum(u*v, dim=1) - 2*u[:, 0]*v[:, 0]
                return val
            raise ValueError("u, v can not be {}-dimensional".format(len(v.shape)))
        raise ValueError("u,v shape should be same")

    def proj(self, X, U):
        if self._k == 1:
            return U + self._lorentz_scalar_product(X, U) * X
        else:
            return U + self._lorentz_scalar_product(X, U).reshape(self._k, -1) * X

    def egrad2rgrad(self, X, U):
        gl = torch.diag(torch.ones(self._n))
        gl[0, 0] = -1
        if self._k == 1:
            return self.proj(X, multiprod(gl, U))
        else:
            return self.proj(X, multitransp(multiprod(gl, multitransp(U))))

    def retr(self, X, G):
        """
        retaction is same as exp
        """
        return self.exp(X, G)

    def exp(self, X, G):
        # check for multi dimenstions
        G_lnorm = self.norm(X, G)
        if self._k == 1:
            ex = torch.cosh(G_lnorm) * X + torch.sinh(G_lnorm) * (G/G_lnorm)
            if G_lnorm == 0:
                ex = X
            return ex
        else:
            G_lnorm = G_lnorm.view(-1, 1)
            ex = torch.cosh(G_lnorm) * X + torch.sinh(G_lnorm) * (G/G_lnorm)
            exclude = G_lnorm == 0
            exclude = exclude.view(-1)
            ex[exclude, :] = X[exclude, :]
            return ex

    def inner(self, X, G1, G2):
        return torch.sum(G1 * G2)

    def norm(self, X, G):
        linear_product = self._lorentz_scalar_product(G, G)
        return torch.sqrt(torch.max(linear_product,
                                    torch.ones_like(linear_product) * self.eps))

    def transp(self, x1, x2, d):
        return self.proj(x2, d)
    
    def _acosh(self, x):
        return torch.log(x+(x**2-1)**0.5)

    def dist(self, X, Y):
        # arccosh(max (1,   -<X,Y>_L) )
        linear_product = -1 * self._lorentz_scalar_product(X, Y)
        return self._acosh(torch.max(linear_product, torch.ones_like(linear_product)))

    # TODO: inner, norm transp check
