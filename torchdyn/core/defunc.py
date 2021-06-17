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


class DEFuncBase(nn.Module):
    """Base differential equation vector field wrapper.

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
   """
    def __init__(self, vector_field, has_time_arg=True):
        super().__init__()
        self.nfe = 0.
        self.vf = vector_field
        # flag if vector field has `t` in call signature
        # permits upstream compatibility with a wider selection of vector fields
        # including lambda functions 
        self.has_time_arg = has_time_arg 

    def forward(self, t, x):
        self.nfe += 1
        if self.has_time_arg: return self.vf(t, x)
        else: return self.vf(x)


class DEFunc(nn.Module):
    """Differential Equation Function wrapper. Handles auxiliary tasks for NeuralDEs: time ("depth") concatenation,
    higher order dynamics and forward propagated integral losses.

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
   """
    def __init__(self, model, order=1):
        super().__init__()
        self.m, self.nfe,  = model, 0.
        self.order, self.intloss, self.sensitivity = order, None, None

    def forward(self, t, x):
        self.nfe += 1
        # set `t` depth-variable to DepthCat modules
        for _, module in self.m.named_modules():
            if hasattr(module, 't'):
                module.t = t

        # if-else to handle autograd training with integral loss propagated in x[:, 0]
        if (not self.intloss is None) and self.sensitivity == 'autograd':
            x_dyn = x[:, 1:]
            dlds = self.intloss(t, x_dyn)
            if len(dlds.shape) == 1: dlds = dlds[:, None]
            if self.order > 1: x_dyn = self.horder_forward(t, x_dyn)
            else: x_dyn = self.m(x_dyn)
            return torch.cat([dlds, x_dyn], 1).to(x_dyn)

        # regular forward
        else:
            if self.order > 1: x = self.horder_forward(t, x)
            else: x = self.m(x)
            return x

    def horder_forward(self, t, x):
        x_new = []
        size_order = x.size(1) // self.order
        for i in range(1, self.order):
            x_new += [x[:, size_order*i:size_order*(i+1)]]
        x_new += [self.m(x)]
        return torch.cat(x_new, 1).to(x)

    
class SDEFunc(nn.Module):
    def __init__(self, f, g, order=1):
        super().__init__()  
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func = f, g
        self.nfe = 0

    def forward(self, t, x):
        pass
    
    def f(self, t, x):
        """Drift."""
        self.nfe += 1
        for _, module in self.f_func.named_modules():
            if hasattr(module, 't'):
                module.t = t
        return self.f_func(x)
    
    def g(self, t, x):
        """Diffusion"""
        for _, module in self.g_func.named_modules():
            if hasattr(module, 't'):
                module.t = t
        return self.g_func(x)
