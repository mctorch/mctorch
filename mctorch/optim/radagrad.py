import torch
from torch.optim.optimizer import Optimizer
from torch.optim import Adagrad

class rAdagrad(Adagrad):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        super(rAdagrad, self).__init__(params, lr=lr, lr_decay=lr_decay,
                                      weight_decay=weight_decay, 
                                      initial_accumulator_value=initial_accumulator_value,
                                      eps=eps)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                if not hasattr(p, 'manifold') or p.manifold is None:
                    if group['weight_decay'] != 0:
                        if p.grad.is_sparse:
                            raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                        grad = grad.add(p, alpha=group['weight_decay'])


                    if grad.is_sparse:
                        grad = grad.coalesce()  # the update is non-linear so indices must be unique
                        grad_indices = grad._indices()
                        grad_values = grad._values()
                        size = grad.size()

                        def make_sparse(values):
                            constructor = grad.new
                            if grad_indices.dim() == 0 or values.dim() == 0:
                                return constructor().resize_as_(grad)
                            return constructor(grad_indices, values, size)
                        state['sum'].add_(make_sparse(grad_values.pow(2)))
                        std = state['sum'].sparse_mask(grad)
                        std_values = std._values().sqrt_().add_(group['eps'])
                        p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                    else:
                        state['sum'].addcmul_(grad, grad, value=1)
                        std = state['sum'].sqrt().add_(group['eps'])
                        p.addcdiv_(grad, std, value=-clr)
                else:
                    if grad.is_sparse:
                        raise RuntimeError('Adagrad with manifold doesn\'t support sparse gradients')
                    rgrad = p.rgrad.data
                    state['sum'].add_(rgrad.pow(2))
                    std = state['sum'].sqrt().add_(1e-10)
                    modified_rgrad = p.manifold.proj(p.data, rgrad / std)
                    p.data.add_(p.manifold.retr(p.data, -clr * modified_rgrad)
                                - p.data)
        return loss
