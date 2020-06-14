import torch
from torch.optim.optimizer import Optimizer


class SGDGen(Optimizer):
    r"""
        based on torch.optim.SGD implementation
    """

    def __init__(self, params, lr, n_workers, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, comp=None, master_comp=None,
                 error_feedback=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDGen, self).__init__(params, defaults)

        self.comp = comp
        self.error_feedback = error_feedback
        if self.error_feedback and self.comp is None:
            raise ValueError("For Error-Feedback, compression can't be None")

        self.master_comp = master_comp  # should be unbiased, Error-Feedback is not supported at the moment

        self.n_workers = n_workers
        self.grads_received = 0

    def __setstate__(self, state):
        super(SGDGen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step_local_global(self, w_id, closure=None):
        """Performs a single optimization step.

        Arguments:
            w_id: integer, id of the worker
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.grads_received += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data

                if self.error_feedback:
                    error_name = 'error_' + str(w_id)
                    if error_name not in param_state:
                        loc_grad = d_p.mul(group['lr'])
                    else:
                        loc_grad = d_p.mul(group['lr']) + param_state[error_name]

                    d_p = self.comp(loc_grad)
                    param_state[error_name] = loc_grad - d_p

                else:
                    if self.comp is not None:
                        d_p = self.comp(d_p).mul(group['lr'])
                    else:
                        d_p = d_p.mul(group['lr'])

                if 'full_grad' not in param_state or self.grads_received == 1:
                    param_state['full_grad'] = torch.clone(d_p).detach()
                else:
                    param_state['full_grad'] += torch.clone(d_p).detach()

                if self.grads_received == self.n_workers:
                    grad = param_state['full_grad'] / self.n_workers

                    if self.master_comp is not None:
                        grad = self.master_comp(grad)

                    if weight_decay != 0:
                        grad.add(p, alpha=weight_decay)
                    if momentum != 0:
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add(buf, alpha=momentum)
                        else:
                            grad = buf

                    p.data.add_(grad, alpha=-1)

        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
