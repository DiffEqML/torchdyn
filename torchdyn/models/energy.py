# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import hessian, jacobian


class ConservativeLinearSNF(nn.Module):
    def __init__(self, energy, J):
        """Stable Neural Flows: https://arxiv.org/abs/2003.08063
        A generalization of Hamiltonian Neural Networks and other energy-based parametrization of Neural ODEs
        Conservative version with energy preservation. Input assumed to be of dimensions `batch, dim`

        Args:
            energy: function parametrizing the energy.
            J: network parametrizing the skew-symmetric interconnection matrix
        """
        super().__init__()
        self.energy = energy
        self.J = J

    def forward(self, x: Tensor):
        with torch.set_grad_enabled(True):
            self.n = x.shape[1] // 2
            x = x.requires_grad_(True)
            dHdx = torch.autograd.grad(self.H(x).sum(), x, create_graph=True)[0]
            dHdx = torch.einsum('ijk, ij -> ik', self._skew(x), dHdx)
        return dHdx

    def _generate_skew(self, x):
        M = self.J(x).reshape(-1, *x.shape[1:])
        return (M - M.transpose(0, 2, 1)) / 2

class GNF(nn.Module):
    def __init__(self, energy:nn.Module):
        """Gradient Neural Flows version of SNFs: https://arxiv.org/abs/2003.08063
        Args:
            energy (nn.Module): function parametrizing the energy.
        """
        super().__init__()
        self.energy = energy

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            eps = self.energy(x).sum()
            out = -torch.autograd.grad(eps, x, allow_unused=False, create_graph=True)[0]
        return out


class HNN(nn.Module):
    def __init__(self, net:nn.Module):
        """Hamiltonian Neural ODE

        Args:
            net (nn.Module): function parametrizing the vector field.
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        with torch.set_grad_enabled(True):
            n = x.shape[1] // 2
            x = x.requires_grad_(True)
            gradH = grad(self.net(x).sum(), x, create_graph=True)[0]
        return torch.cat([gradH[:, n:], -gradH[:, :n]], 1).to(x)


class LNN(nn.Module):
    def __init__(self, net):
        """Lagrangian Neural Network. 

        Args:
            net (nn.Module)
        Notes:
            LNNs are currently quite slow. Improvements will be made whenever `functorch` is either merged upstream or included 
            as a dependency.
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        self.n = n = x.shape[1]//2
        bs = x.shape[0]
        x = x.requires_grad_(True)
        qqd_batch = tuple(x[i, :] for i in range(bs))
        jac = tuple(map(partial(jacobian, self._lagrangian, create_graph=True), qqd_batch))
        hess = tuple(map(partial(hessian, self._lagrangian, create_graph=True), qqd_batch))
        qdd_batch = tuple(map(self._qdd, zip(jac, hess, qqd_batch)))
        qd, qdd = x[:, n:], torch.cat([qdd[None] for qdd in qdd_batch])
        return torch.cat([qd, qdd], 1)

    def _lagrangian(self, qqd):
        return self.net(qqd).sum()

    def _qdd(self, inp):
        n = self.n ; jac, hess, qqd = inp
        return hess[n:, n:].pinverse()@(jac[:n] - hess[n:, :n]@qqd[n:])
