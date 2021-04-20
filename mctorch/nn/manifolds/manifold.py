class Manifold(object):
    """
    Base class for manifold constraints

    All manifolds should subclass this class:

        import torch.nn as nn
        class XManifold(nn.Manifold):
            def __init__(self, args):
                super(Manifold, self).__init__()

            def dim(self):
                ...

    All functions map to corresponding functions in
    Manopt `<http://www.manopt.org>` and its python dervivation
    pymanopt `<https://github.com/pymanopt/pymanopt>`

    All functions should be converted to torch counterparts.

    """

    def __init__(self):
        self._dim = None
        self._size = None

    def __str__(self):
        """
        Name of the manifold
        """

    def dim(self):
        """
        Dimension of the manifold
        """
        return self._dim

    def size(self):
        """
        Returns tuple denoting size of a point on manifold
        """
        return self._size

    def dist(self, X, Y):
        """
        Geodesic distance on the manifold
        """
        raise NotImplementedError

    def inner(self, X, G, H):
        """
        Inner product (Riemannian metric) on the tangent space
        """
        raise NotImplementedError

    def proj(self, X, G):
        """
        Project into the tangent space. Usually the same as egrad2rgrad
        """
        raise NotImplementedError

    def egrad2rgrad(self, X, G):
        """
        A mapping from the Euclidean gradient G into the tangent space
        to the manifold at X. For embedded manifolds, this is simply the
        projection of G on the tangent space at X.
        """
        return self.proj(X, G)

    def ehess2rhess(self, X, Hess):
        """
        Convert Euclidean into Riemannian Hessian.
        """
        raise NotImplementedError

    def retr(self, X, G):
        """
        A retraction mapping from the tangent space at X to the manifold.
        See Absil for definition of retraction.
        TODO: retr with inplace update
        """
        raise NotImplementedError

    def norm(self, X, G):
        """
        Compute the norm of a tangent vector G, which is tangent to the
        manifold at X.
        """
        raise NotImplementedError

    def rand(self):
        """
        A function which returns a random point on the manifold.
        """
        raise NotImplementedError

    def randvec(self, X):
        """
        Returns a random, unit norm vector in the tangent space at X.
        """
        raise NotImplementedError

    def transp(self, x1, x2, d):
        """
        Transports d, which is a tangent vector at x1, into the tangent
        space at x2.
        """
        raise NotImplementedError

    def exp(self, X, U):
        """
        The exponential (in the sense of Lie group theory) of a tangent
        vector U at X.
        """
        raise NotImplementedError

    def log(self, X, Y):
        """
        The logarithm (in the sense of Lie group theory) of Y. This is the
        inverse of exp.
        """
        raise NotImplementedError

    def pairmean(self, X, Y):
        """
        Computes the intrinsic mean of X and Y, that is, a point that lies
        mid-way between X and Y on the geodesic arc joining them.
        """
        raise NotImplementedError

    def zerovec(self, X):
        """
        Returns the zero tangent vector at X.
        """
        return np.zeros(X.size())
