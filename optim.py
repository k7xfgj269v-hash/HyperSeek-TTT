import torch


def newton_schulz(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B_mat = b * A + c * A @ A
        X = a * X + B_mat @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if g.ndim >= 2:
                    g_flat = g.reshape(g.shape[0], -1)
                    g_orth = newton_schulz(g_flat).reshape_as(g)
                    p.data.add_(g_orth, alpha=-lr)
                else:
                    p.data.add_(g, alpha=-lr)
        return loss


def split_params(model):
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'token_emb' in name or p.ndim < 2 or min(p.shape) == 1:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    return muon_params, adamw_params
