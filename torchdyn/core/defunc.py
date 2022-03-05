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

from typing import Callable
import torch
from torch import Tensor, cat
import torch.nn as nn


class DEFuncBase(nn.Module):
    def __init__(self, vector_field:Callable, has_time_arg:bool=True):
        """Basic wrapper to ensure call signature compatibility between generic torch Modules and vector fields.
        Args:
            vector_field (Callable): callable defining the dynamics / vector field / `dxdt` / forcing function 
            has_time_arg (bool, optional): Internal arg. to indicate whether the callable has `t` in its `__call__' 
                or `forward` method. Defaults to True.
        """
        super().__init__()
        self.nfe, self.vf, self.has_time_arg = 0., vector_field, has_time_arg 

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        self.nfe += 1
        if self.has_time_arg: return self.vf(t, x)
        else: return self.vf(x)


class DEFunc(nn.Module):
    def __init__(self, vector_field:Callable, order:int=1):
        """Special vector field wrapper for Neural ODEs. 
        
        Handles auxiliary tasks: time ("depth") concatenation, higher-order dynamics and forward propagated integral losses.

        Args:
            vector_field (Callable): callable defining the dynamics / vector field / `dxdt` / forcing function 
            order (int, optional): order of the differential equation. Defaults to 1.
        
        Notes:
            Currently handles the following:
            (1) assigns time tensor to each submodule requiring it (e.g. `GalLinear`). 
            (2) in case of integral losses + reverse-mode differentiation, propagates the loss in the first dimension of `x`
                and automatically splits the Tensor into `x[:, 0]` and `x[:, 1:]` for vector field computation
            (3) in case of higher-order dynamics, adjusts the vector field forward to recursively compute various orders.
        """
        super().__init__()
        self.vf, self.nfe,  = vector_field, 0.
        self.order, self.integral_loss, self.sensitivity = order, None, None
        # identify whether vector field already has time arg

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        self.nfe += 1
        # set `t` depth-variable to DepthCat modules
        for _, module in self.vf.named_modules():
            if hasattr(module, 't'):
                module.t = t

        # if-else to handle autograd training with integral loss propagated in x[:, 0]
        if (self.integral_loss is not None) and self.sensitivity == 'autograd':
            x_dyn = x[:, 1:]
            dlds = self.integral_loss(t, x_dyn)
            if len(dlds.shape) == 1: dlds = dlds[:, None]
            if self.order > 1: x_dyn = self.horder_forward(t, x_dyn)
            else: x_dyn = self.vf(t, x_dyn)
            return cat([dlds, x_dyn], 1).to(x_dyn)

        # regular forward
        else:
            if self.order > 1: x = self.higher_order_forward(t, x)
            else: x = self.vf(t, x)
            return x

    def higher_order_forward(self, t:Tensor, x:Tensor) -> Tensor:
        x_new = []
        size_order = x.size(1) // self.order
        for i in range(1, self.order):
            x_new.append(x[:, size_order*i : size_order*(i+1)])
        x_new.append(self.vf(t, x))
        return cat(x_new, dim=1).to(x)

    
class SDEFunc(nn.Module):
    def __init__(self, f:Callable, g:Callable, order:int=1):
        """"Special vector field wrapper for Neural SDEs.

        Args:
            f (Callable): callable defining the drift
            g (Callable): callable defining the diffusion term
            order (int, optional): order of the differential equation. Defaults to 1.
        """
        super().__init__()  
        self.order, self.intloss, self.sensitivity = order, None, None
        self.f_func, self.g_func = f, g
        self.nfe = 0

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        pass
    
    def f(self, t:Tensor, x:Tensor) -> Tensor:
        self.nfe += 1
        for _, module in self.f_func.named_modules():
            if hasattr(module, 't'):
                module.t = t
        return self.f_func(x)
    
    def g(self, t:Tensor, x:Tensor) -> Tensor:
        for _, module in self.g_func.named_modules():
            if hasattr(module, 't'):
                module.t = t
        return self.g_func(x)
