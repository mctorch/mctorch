import torch
from torch.optim.optimizer import Optimizer

class rASA(Optimizer):
    """
    Riemannian Adaptive Stochastic gradient

    @InProceedings{pmlr-v97-kasai19a,
      title     = 	 {{R}iemannian adaptive stochastic gradient algorithms on matrix manifolds},
      author    =    {Kasai, Hiroyuki and Jawanpuria, Pratik and Mishra, Bamdev},
      booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
      pages     = 	 {3262--3271},
      year      = 	 {2019}
    }
    """

    def __init__(
            self,
            params,
            lr=1e-2,
            norm_cov=True,
            max_op=True,
            scaling="LR",
    ):
        """
        Args:
            params (iterable): an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized.
            lr (float): learning rate
            norm_cov (bool): normalizes the covariance matrices in "LR" scaling
            max_op (bool): apply max operation to ensure non decrease sequence of weights
            scaling (str): choice of scaling in update step (either "LR" or "vec")
        """
        scaling_options = ["LR", "vec"]
        assert scaling in scaling_options, f"scaling should be one of {scaling_options}"

        defaults = dict(lr=lr, beta=0.999, norm_cov=norm_cov, max_op=max_op, scaling=scaling)
        super(rASA, self).__init__(params, defaults)


    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                Shape = list(p.data.shape)
                lShape = tuple(Shape[:-1])
                Shape.pop(-2)
                rShape = tuple(Shape)

                state = self.state[p]
                lr, beta, norm_cov, max_op, scaling = group['lr'], group['beta'], group['norm_cov'], group['max_op'], group['scaling']

                if len(state) == 0:
                    if scaling == "LR":
                        state['l_prev'] = torch.zeros(lShape)
                        state['r_prev'] = torch.zeros(rShape)
                        state['l_hat_prev'] = torch.zeros(lShape)
                        state['r_hat_prev'] = torch.zeros(rShape)
                    else:
                        state['vec_prev'] = torch.zeros(p.data.shape)
                        state['vec_hat_prev'] = torch.zeros(p.data.shape)

                rows, cols = lShape, rShape
                if scaling == "LR":
                    l_prev, r_prev, l_hat_prev, r_hat_prev = state['l_prev'], state['r_prev'], state['l_hat_prev'], state['r_hat_prev']
                    row_norm, col_norm = (1, 1) if not norm_cov else (torch.ones(rows)/rows[-1], torch.ones(cols)/cols[-1])
                else:
                    vec_prev, vec_hat_prev = state['vec_prev'], state['vec_hat_prev']

                if not hasattr(p, 'manifold') or p.manifold is None:
                    raise ValueError(f"You can define a Euclidean manifold of shape ({rows}, {cols})")
                else:
                    grad = p.rgrad.data

                    if scaling == "LR":
                        l = beta*l_prev + (1-beta)*row_norm*torch.sum(grad**2, axis=-1)
                        r = beta*r_prev + (1-beta)*col_norm*torch.sum(grad**2, axis=-2)
                        l_hat = torch.max(l, l_hat_prev)
                        r_hat = torch.max(r, r_hat_prev)

                        l_hat, r_hat = torch.pow(l_hat, 0.25), torch.pow(r_hat, 0.25)

                        adgrad = (1/l_hat).unsqueeze(-1) * (grad * (1/r_hat).unsqueeze(-2))
                    else:
                        vec = beta*vec_prev + (1-beta)*(grad**2)
                        vec_hat = torch.max(vec, vec_hat_prev)

                        adgrad = grad / (torch.pow(vec_hat, 0.5))

                    grad = p.manifold.proj(p.data, adgrad)

                    p.data.add_(p.manifold.retr(p.data, -lr * grad) - p.data)

                    if scaling == "LR":
                        state['l_prev'], state['r_prev'] = l, r
                        if max_op:
                            state['l_hat_prev'], state['r_hat_prev'] = l_hat, r_hat
                    else:
                        state['vec_prev'] = vec
                        if max_op:
                            state['vec_hat_prev'] = vec_hat
        return loss
