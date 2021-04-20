import torch
from torch.optim.optimizer import Optimizer


class ConjugateGradient(Optimizer):
    """Implement Conjugate Gradient algorighm

    """
    BETA_TYPES = ["FletcherReeves", "PolakRibiere", "HestenesStiefel",
                  "HagerZhang"]

    def __init__(self, params, lr=1e-2, beta_type='HestenesStiefel',
                 orth_value=float('Inf')):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta_type not in self.BETA_TYPES:
            raise ValueError("Invalid beta_type: {}".format(beta_type))

        defaults = dict(lr=lr, beta_type=beta_type, orth_value=orth_value)
        super(ConjugateGradient, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('ConjugateGradient does not support sparse gradients')

                grad = p.grad.data
                state = self.state[p]
                beta_type = group['beta_type']
                orth_value = group['orth_value']
                lr = group['lr']

                if len(state) == 0:
                    state['old_grad'] = torch.zeros_like(p.data)
                    state['ograd_ograd'] = 0.0  # shape
                    state['old_norm'] = 0.0
                    state['old_x'] = None  # p.data.copy_() not needed

                old_grad = state['old_grad']
                old_x = state['old_x']
                ograd_ograd = state['ograd_ograd']
                old_norm = state['old_norm']

                if not hasattr(p, 'manifold') or p.manifold is None:
                    # euclidean manifold case
                    grad_grad = torch.sum(grad * grad)
                    if abs(grad_grad) > 0:
                        orth_grad = torch.sum(old_grad * grad) / grad_grad
                    else:
                        orth_grad = None
                    desc_dir = -1 * old_grad

                    if ((orth_grad is not None and
                        abs(orth_grad) >= orth_value) or
                       not abs(ograd_ograd) > 0):
                        beta = 0
                        desc_dir = -1 * grad

                    else:
                        if beta_type == self.BETA_TYPES[0]:  # FletcherReeves
                            beta = grad_grad / ograd_ograd

                        elif beta_type == self.BETA_TYPES[1]:  # PolakRibiere
                            diff = grad - old_grad
                            ip_diff = torch.sum(grad * diff)
                            beta = max(0, ip_diff / ograd_ograd)

                        elif beta_type == self.BETA_TYPES[2]:  # HestenesStiefel
                            diff = grad - old_grad
                            ip_diff = torch.sum(grad * diff)
                            deno = torch.sum(diff * desc_dir)
                            if abs(deno) > 0:
                                beta = max(0, ip_diff / deno)
                            else:
                                beta = 1

                        elif beta_type == self.BETA_TYPES[3]:  # HagerZhang
                            diff = grad - old_grad
                            deno = torch.sum(diff * desc_dir)
                            numo = torch.sum(diff * grad)
                            if abs(deno) > 0:
                                numo -= (2 * torch.sum(diff * diff) *
                                         torch.sum(grad * desc_dir) / deno)
                                beta = numo / deno
                            else:
                                beta = 0  # confirm this condition
                            # Robustness according to Hager-Zhang paper
                            desc_dir_norm = torch.norm(desc_dir)
                            eta_HZ = -1 * (desc_dir_norm * min(0.01, old_norm))
                            beta = max(beta, eta_HZ)

                        else:
                            raise ValueError('Unknown beta_type: {}'
                                             .format(beta_type))
                        desc_dir = -1 * grad + beta * desc_dir

                        df0 = torch.sum(grad * desc_dir)
                        if df0 >= 0:
                            desc_dir = -1 * grad
                    p.data.add_(group['lr'], desc_dir)
                    state['old_grad'].mul_(0).add_(grad)
                    state['ograd_ograd'] = torch.sum(grad * grad).item()
                    state['old_norm'] = torch.norm(grad).item()
                    state['old_x'] = None

                else:
                    grad = p.rgrad.data  # gras is rgrad here

                    grad_grad = p.manifold.inner(p.data, grad, grad)
                    old_grad = p.manifold.transp(old_x, p.data, old_grad)
                    if abs(grad_grad) > 0:
                        orth_grad = p.manifold.inner(p.data, old_grad, grad) / grad_grad
                    else:
                        orth_grad = None
                    desc_dir = p.manifold.transp(old_x, p.data, -1 * old_grad)

                    if ((orth_grad is not None and
                        abs(orth_grad) >= orth_value) or
                       not abs(ograd_ograd) > 0):
                        beta = 0
                        desc_dir = -1 * grad

                    else:
                        if beta_type == self.BETA_TYPES[0]:  # FletcherReeves
                            beta = grad_grad / ograd_ograd

                        elif beta_type == self.BETA_TYPES[1]:  # PolakRibiere
                            diff = grad - old_grad
                            ip_diff = p.manifold.inner(p.data, grad, diff)
                            beta = max(0, ip_diff / ograd_ograd)

                        elif beta_type == self.BETA_TYPES[2]:  # HestenesStiefel
                            diff = grad - old_grad
                            ip_diff = p.manifold.inner(p.data, grad, diff)
                            deno = p.manifold.inner(p.data, diff, desc_dir)
                            if abs(deno) > 0:
                                beta = max(0, ip_diff / deno)
                            else:
                                beta = 1

                        elif beta_type == self.BETA_TYPES[3]:  # HagerZhang
                            diff = grad - old_grad
                            deno = p.manifold.inner(p.data, diff, desc_dir)
                            numo = p.manifold.inner(p.data, diff, grad)
                            if abs(deno) > 0:
                                numo -= (2 * p.manifold.inner(p.data, diff, diff) *
                                         p.manifold.inner(p.data, desc_dir, grad) / deno)
                                beta = numo / deno
                            else:
                                beta = 0
                            # Robustness according to Hager-Zhang paper
                            desc_dir_norm = p.manifold.norm(p.data, desc_dir)
                            eta_HZ = -1 * (desc_dir_norm * min(0.01, old_norm))
                            beta = max(beta, eta_HZ)

                        else:
                            raise ValueError('Unknown beta_type: {}'
                                             .format(beta_type))
                        desc_dir = -1 * grad + beta * desc_dir

                        df0 = p.manifold.inner(p.data, grad, desc_dir)
                        if df0 >= 0:
                            desc_dir = -1 * grad

                    p.data.add_(p.manifold.retr(p.data,
                                group['lr'] * desc_dir) - p.data)
                    state['old_grad'].mul_(0).add_(grad)
                    state['ograd_ograd'] = p.manifold.inner(p.data, grad, grad).item()
                    state['old_norm'] = p.manifold.norm(p.data, grad).item()
                    state['old_x'] = None
        return loss
