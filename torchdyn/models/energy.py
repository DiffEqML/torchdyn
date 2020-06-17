import torch
import torch.nn as nn
from torch.autograd.functional import jacobian as jac

# class Stable(nn.Module):
#     """Stable Neural Flow"""
#     def __init__(self, net, depthvar=False):
#         super().__init__()
#         self.net, self.depthvar = net, depthvar

#     def forward(self, x):
#         bs, n = x.shape[0], x.shape[1] // 2
#         out = -jac(self.net, x.requires_grad_(True), strict=True, create_graph=True)[range(bs), :, range(bs), :]
#         return out[:,:-1] if self.depthvar else out


class Stable(nn.Module):
    """Stable Neural Flow"""
    def __init__(self, net, depthvar=False):
        super().__init__()
        self.net, self.depthvar = net, depthvar

    def forward(self, x):
        bs, n = x.shape[0], x.shape[1] // 2
        x.requires_grad_(True)
        out = -torch.autograd.grad(self.net(x).sum(), x, create_graph=True)[0] #jac(self.net, x, strict=True, create_graph=True)[range(bs), :, range(bs), :]
        #self.out = out
        return out[:,:-1] if self.depthvar else out

class HNN(nn.Module):
    """Hamiltonian Neural Network"""
    def __init__(self, Hamiltonian:nn.Module, dim=1):
        super().__init__()
        self.H = Hamiltonian
        self.n = dim

    def forward(self, x):
        x.requires_grad_(True)
        gradH = torch.autograd.grad(self.H(x).sum(), x,
                                    create_graph=True)[0]
        return torch.cat([gradH[:,self.n:], -gradH[:,:self.n]], 1).to(x)

class LNN(nn.Module):
    """Lagrangian Neural Network"""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = len(x) // 2

        x = torch.autograd.Variable(x, requires_grad=True)
        q, qd = x[:n], x[n:]
        dL_dq = jacobian(self._lagrangian, (q, qd), create_graph=True)[0]
        H = hessian(self._lagrangian, (q, qd), create_graph=True)
        ddL_ddqd, ddL_dqd_dq = H[1][1], H[1][0]
        qdd = ddL_ddqd.inverse() @ (dL_dq - ddL_dqd_dq @ qd)
        return torch.cat([qd, qdd[0]])

    def _lagrangian(self, q, qd):
        x = torch.cat([q, qd], 0)
        return net(x)
    
