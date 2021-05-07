import torch
import math

from ..utils.manifold_multi import multiprod, multitransp, multisym
from .manifold import Manifold

def SKnopp(A, p, q, maxiters=None, checkperiod=None):
    tol = 1e-9
    if maxiters is None:
        maxiters = A.shape[0]*A.shape[1]

    if checkperiod is None:
        checkperiod = 10

    if p.ndim < 2 and q.ndim < 2:
        p = p[None, :]
        q = q[None, :]

    C = A

    # TODO: Maybe improve this if-else by looking
    # for other broadcasting techniques
    if C.ndim < 3:
        d1 = q / torch.sum(C, axis=0)[None, :]
    else:
        d1 = q / torch.sum(C, axis=1)

    if C.ndim < 3:
        d2 = p / multiprod(d1, C.T)
    else:
        d2 = p / torch.sum(C * d1[:, None, :], axis=2)

    gap = float("inf")

    iters = 0
    while iters < maxiters:
        if C.ndim < 3:
            row = multiprod(d2, C)
        else:
            row = torch.sum(C * d2[:, :, None], axis=1)

        if iters % checkperiod == 0:
            gap = torch.max(torch.absolute(row * d1 - q))
            if torch.any(torch.isnan(gap)) or gap <= tol:
                break
        iters += 1

        d1_prev = d1
        d2_prev = d2
        d1 = q / row
        if C.ndim < 3:
            d2 = p / multiprod(d1, C.T)
        else:
            d2 = p / torch.sum(C * d1[:, None, :], axis=2)

        if torch.any(torch.isnan(d1)) or torch.any(torch.isinf(d1)) or torch.any(torch.isnan(d2)) or torch.any(torch.isinf(d2)):
            warnings.warn("""SKnopp: NanInfEncountered
                    Nan or Inf occured at iter {:d} \n""".format(iters))
            d1 = d1_prev
            d2 = d2_prev
            break

    return C * (torch.einsum('bn,bm->bnm', d2, d1))


class DoublyStochastic(Manifold):
    """
    Manifold of `k` (n x m) positive matrices

    Implementation is based on multinomialdoublystochasticgeneralfactory.m
    """

    def __init__(self, n, m, k=1, p=None, q=None, maxSKnoppIters=None, checkperiod=None):
        self._n = n
        self._m = m
        self._p = p
        self._q = q
        self._maxSKnoppIters = maxSKnoppIters
        self._checkperiod = checkperiod

        # Assume that the problem is on single manifold.
        if p is None:
            self._p = 1/n * torch.ones(n)
        if q is None:
            self._q = 1/m * torch.ones(m)
        if self._p.ndim < 2 and self._q.ndim < 2:
            self._p = self._p[None, :]
            self._q = self._q[None, :]
        # `k` doublystochastic manifolds
        assert k == self._p.shape[0], ValueError(f"k:{k} and dim 0 of p:{self._p.shape[0]} don't match")
        self._k = k

        if maxSKnoppIters is None:
            self._maxSKnoppIters = min(2000, 100 + m + n)
        if checkperiod is None:
            self._checkperiod = 10

        self._size = torch.Size((self._k, self._n, self._m))

        self._dim = self._k * (self._n - 1)*(self._m - 1)
        self._e1 = torch.ones(n)
        self._e2 = torch.ones(m)


    def __str__(self):
        return ("{:d} {:d}X{:d} matrices with positive entries such that row sum is p and column sum is q respectively.".format(self._k, self._n, self._m))


    @property
    def dim(self):
        return self._dim


    @property
    def typicaldist(self):
        return math.sqrt(self._k) * (self._m + self._n)


    def inner(self, x, u, v):
        return torch.sum(u * v/ x)


    def norm(self, x, u):
        return torch.sqrt(self.inner(x, u, u))


    def rand(self):
        Z = torch.absolute(torch.randn(self._n, self._m))
        return SKnopp(Z, self._p, self._q, self._maxSKnoppIters, self._checkperiod)


    def randvec(self, x):
        Z = torch.randn(self._k, self._n, self._m)
        Zproj = self.proj(x, Z)
        return Zproj / self.norm(x, Zproj)


    def _matvec(self, v):
        raise RuntimeError
        self._k = int(self.X.shape[0])
        v = v.reshape(self._k, int(v.shape[0]/self._k))
        vtop = torch.tensor(v[:, :self._n])
        vbottom = torch.tensor(v[:, self._n:])
        Avtop = (vtop * self._p) + torch.sum(self.X * vbottom[:, None, :], axis=2)
        Avbottom = torch.sum(self.X * vtop[:, :, None], axis=1) + (vbottom * self._q)
        Av = torch.hstack((Avtop, Avbottom))
        return Av.ravel()


    def _optimLSolve(self, x, b):
        raise RuntimeError
        self.X = x.clone()
        _dim = self._k * (self._n + self._m)
        shape = (_dim, _dim)
        sol, _iters = cg(LinearOperator(shape, matvec=self._matvec), convert2numpy(b), tol=1e-6, maxiter=100)
        sol = sol.reshape(self._k, int(sol.shape[0]/self._k))
        del self.X
        alpha, beta = sol[:, :self._n], sol[:, self._n:]
        return torch.array(alpha), proc.array(beta)


    def _lsolve(self, x, b):
        # TODO: A better way to solve it is implemented in `_optimLSolve`
        # Once Pytorch gains support for `LinearOperator`/scipy like `cg`
        # function, it can be used.
        alpha = torch.empty((self._k, self._n))
        beta = torch.empty((self._k, self._m))
        for i in range(self._k):
            A = torch.cat((torch.cat((torch.diag(self._p[i]), x[i]), dim=-1), torch.cat((x[i].T, torch.diag(self._q[i])), dim=-1)), dim=0)
            zeta = torch.linalg.solve(A, b[i])
            alpha[i], beta[i] = zeta[:self._n], zeta[self._n:]
        return alpha, beta


    def proj(self, x, v):
        assert v.ndim == 3
        b = torch.hstack((torch.sum(v, axis=2), torch.sum(v, axis=1)))
        alpha, beta = self._lsolve(x, b)
        result = v - (torch.einsum('bn,m->bnm', alpha, self._e2) + torch.einsum('n,bm->bnm', self._e1, beta))*x
        return result


    def dist(self, x, y):
        raise NotImplementedError


    def egrad2rgrad(self, x, u):
        mu = x * u
        return self.proj(x, mu)


    def ehess2rhess(self, x, egrad, ehess, u):
        gamma = egrad * x
        gamma_dot = (ehess * x) + (egrad * u)

        assert gamma.ndim == 3 and gamma_dot.ndim == 3
        b = torch.hstack((torch.sum(gamma, axis=2), torch.sum(gamma, axis=1)))
        b_dot = torch.hstack((torch.sum(gamma_dot, axis=2), torch.sum(gamma_dot, axis=1)))

        alpha, beta = self._lsolve(x, b)
        alpha_dot, beta_dot = self._lsolve(
            x,
            b_dot - torch.hstack((
                torch.einsum('bnm,bm->bn', u, beta),
                torch.einsum('bnm,bn->bm', u, alpha)
            ))
        )

        S = torch.einsum('bn,m->bnm', alpha, self._e2) + torch.einsum('n,bm->bnm', self._e1, beta)
        S_dot = torch.einsum('bn,m->bnm', alpha_dot, self._e2) + torch.einsum('n,bm->bnm', self._e1, beta_dot)
        delta_dot = gamma_dot - (S_dot*x) - (S*u)

        delta = gamma - (S*x)

        nabla = delta_dot - (0.5 * (delta * u)/x)

        return self.proj(x, nabla)


    def retr(self, x, u):
        Y = x * torch.exp(u/x)
        Y = torch.clip(Y, 1e-16, 1e16)
        return SKnopp(Y, self._p, self._q, self._maxSKnoppIters, self._checkperiod)


    def zerovec(self, x):
        return torch.zeros((self._k, self._n, self._m))


    def transp(self, x1, x2, d):
        return self.proj(x2, d)
