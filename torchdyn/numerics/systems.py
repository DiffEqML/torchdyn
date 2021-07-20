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

import torch
import torch.nn as nn
import torch.distributions 
from torch.distributions import Uniform


class Lorenz(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(1,1)
    
    def forward(self, t, x):
        x1, x2, x3 = x[...,:1], x[...,1:2], x[...,2:]
        dx1 = 10 * (x2 - x1)
        dx2 = x1 * (28 - x3) - x2
        dx3 = x1 * x2 - 8/3 * x3
        return torch.cat([dx1, dx2, dx3], -1)


class VanDerPol(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        x1, x2 = x[...,:1], x[...,1:2]
        return torch.cat([x2, self.alpha * (1 - x1**2) * x2 - x1], -1)
        
class ODEProblem2(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, s, z):
        return 0.5*z
  
class ODEProblem3(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, s, z):
        return -0.1*z

    
class ODEProblem4(nn.Module):
    "Rabinovich-Fabrikant"
    def __init__(self):
        super().__init__()
    
    def forward(self, s, z):
        x1, x2, x3 = z[...,:1], z[...,1:2], z[...,-1:] 
        dx1 = x2 * (x3 - 1 + x1**2) + 0.87*x1
        dx2 = x1*(3*x3+1-x1**2) + 0.87*x2
        dx3 = -2*x3*(1.1+x1*x2)
        return torch.cat([dx1, dx2, dx3], -1)



class SineSystem(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, s, z):
        s = s * torch.ones_like(z)
        return torch.sin(s)


class LTISystem(nn.Module):
    def __init__(self, dim=2, randomizable=True):
        super().__init__()
        self.dim = dim
        self.randomizable = randomizable
        self.l = nn.Linear(dim, dim)
        
    def forward(self, s, x):
        return self.l(x) 
    
    def randomize_parameters(self):
        self.l = nn.Linear(self.dim, self.dim)

class FourierSystem(nn.Module):
    def __init__(self,
                 dim=2, 
                 A_dist=Uniform(-10, 10),
                 phi_dist=Uniform(-1, 1),
                 w_dist=Uniform(-20, 20),
                 randomizable=True
                 ):
        
        super().__init__()
        self.n_harmonics = n_harmonics = torch.randint(2, 20, size=(1,))
        self.A_dist = A_dist; self.A = A_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.phi_dist = phi_dist; self.phi = phi_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.w_dist = w_dist; self.w = w_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.dim = dim
        self.randomizable = randomizable
        
    def forward(self, s, x):
        if len(s.shape) == 0:
            return (self.A[:, :, 0] * torch.cos(self.w[:, :, 0]*s + self.phi[:, :, 0]) + 
                   self.A[:, :, 1] * torch.cos(self.w[:, :, 1]*s + self.phi[:, :, 1])).sum(1)[None, :]
        else:
            sol = []
            for s_ in s:
                sol += [(self.A[:, :, 0] * torch.cos(self.w[:, :, 0]*s_ + self.phi[:, :, 0]) + 
                   self.A[:, :, 1] * torch.cos(self.w[:, :, 1]*s_ + self.phi[:, :, 1])).sum(1)[None, :]]
            return torch.cat(sol)
        
    def randomize_parameters(self):
        self.A = self.A_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.phi = self.phi_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.w = self.w_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        
        
class StiffFourierSystem(nn.Module):
    def __init__(self,
                 dim=2, 
                 A_dist=Uniform(-10, 10),
                 phi_dist=Uniform(-1, 1),
                 w_dist=Uniform(-20, 20),
                 randomizable=True
                 ):
        
        super().__init__()
        self.n_harmonics = n_harmonics = torch.randint(20, 100, size=(1,))
        self.A_dist = A_dist; self.A = A_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.phi_dist = phi_dist; self.phi = phi_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.w_dist = w_dist; self.w = w_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.dim = dim
        self.randomizable = randomizable
        
    def forward(self, s, x):
        if len(s.shape) == 0:
            return (self.A[:, :, 0] * torch.cos(self.w[:, :, 0]*s + self.phi[:, :, 0]) + 
                   self.A[:, :, 1] * torch.cos(self.w[:, :, 1]*s + self.phi[:, :, 1])).sum(1)[None, :]
        else:
            sol = []
            for s_ in s:
                sol += [(self.A[:, :, 0] * torch.cos(self.w[:, :, 0]*s_ + self.phi[:, :, 0]) + 
                   self.A[:, :, 1] * torch.cos(self.w[:, :, 1]*s_ + self.phi[:, :, 1])).sum(1)[None, :]]
            return torch.cat(sol)
        
    def randomize_parameters(self):
        self.A = self.A_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.phi = self.phi_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.w = self.w_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
         
        
class CoupledFourierSystem(nn.Module):
    def __init__(self,
                 dim=2, 
                 A_dist=Uniform(-10, 10),
                 phi_dist=Uniform(-1, 1),
                 w_dist=Uniform(-20, 20),
                 randomizable=True
                 ):

        super().__init__()
        self.n_harmonics = n_harmonics = torch.randint(2, 20, size=(1,))
        self.A_dist = A_dist; self.A = A_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.phi_dist = phi_dist; self.phi = phi_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.w_dist = w_dist; self.w = w_dist.sample(torch.Size([dim, n_harmonics, 2]))
        self.dim = dim
        self.randomizable = randomizable
        self.mixing_l = nn.Linear(dim, dim)

    def forward(self, s, x):
        if len(s.shape) == 0:
            pre_sol = (self.A[:, :, 0] * torch.cos(self.w[:, :, 0]*s + self.phi[:, :, 0]) + 
                       self.A[:, :, 1] * torch.cos(self.w[:, :, 1]*s + self.phi[:, :, 1])).sum(1)[None, :]
            return self.mixing_l(pre_sol)

        else:
            sol = []
            for s_ in s:
                sol += [(self.A[:, :, 0] * torch.cos(self.w[:, :, 0]*s_ + self.phi[:, :, 0]) + 
                   self.A[:, :, 1] * torch.cos(self.w[:, :, 1]*s_ + self.phi[:, :, 1])).sum(1)[None, None, :]]
            return self.mixing_l(torch.cat(sol, 0))[:, 0, :]

    def randomize_parameters(self):
        self.A = self.A_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.phi = self.phi_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.w = self.w_dist.sample(torch.Size([self.dim, self.n_harmonics, 2]))
        self.mixing_l = nn.Linear(self.dim, self.dim)

        
class MatMulSystem(nn.Module):
    def __init__(self, dim=2, activation=nn.Tanh(), layers=5, hdim=32, randomizable=True):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Sequential(nn.Linear(dim, hdim), nn.Tanh(), 
                                 *[nn.Sequential(nn.Linear(hdim, hdim), nn.Tanh()) for i in range(4)], 
                                 nn.Linear(hdim, dim)))
        self.randomizable = randomizable
        
    def forward(self, s, x):
        return self.net(x)

    def randomize_parameters(self):
        for p in self.net.parameters():
            torch.nn.init.normal_(p, 0, 1)

            
class MatMulBoundedSystem(nn.Module):
    def __init__(self, dim=2, activation=nn.Tanh(), layers=5, hdim=32, randomizable=True):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Sequential(nn.Linear(dim, hdim), nn.Tanh(), 
                                 *[nn.Sequential(nn.Linear(hdim, hdim), nn.Tanh()) for i in range(4)], 
                                 nn.Linear(hdim, dim), 
                                 nn.Tanh()))
        self.randomizable = randomizable
        
    def forward(self, s, x):
        return self.net(x) 
    
    def randomize_parameters(self):
        for p in self.net.parameters():
            torch.nn.init.normal_(p, 0, 1)

          