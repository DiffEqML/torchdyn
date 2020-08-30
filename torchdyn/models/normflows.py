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

def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]  
    return None, trJ

def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)   
    return jvp, trJ

REQUIRES_NOISE = [hutch_trace]

class CNF(nn.Module):
    """Continuous Normalizing Flow

    :param net: function parametrizing the data vector field.
    :type net: nn.Module
    :param trace_estimator: specifies the strategy to otbain Jacobian traces. Options: (autograd_trace, hutch_trace) 
    :type trace_estimator: Callable
    :param noise_dist: distribution of noise vectors sampled for stochastic trace estimators. Needs to have a `.sample` method.
    :type noise_dist: torch.distributions.Distribution
    :param order: specifies parameters of the Neural DE. 
    :type order: int
    """
    def __init__(self, net, trace_estimator=None, noise_dist=None, order=1):
        super().__init__()
        self.net, self.order = net, order # order at the CNF level will be merged with DEFunc
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace;
        self.noise_dist, self.noise = noise_dist, None
        self.intloss = None # this will allow passing `jvp` to `IntegralLoss` for efficient regularization of e.g Frob. norm
        if self.trace_estimator in REQUIRES_NOISE:
            assert self.noise_dist is not None, 'This type of trace estimator requires specification of a noise distribution'
            
    def forward(self, x):   
        with torch.set_grad_enabled(True):
            x_in = torch.autograd.Variable(x[:,1:], requires_grad=True).to(x) # first dimension reserved to divergence propagation
            
            # the neural network will handle the data-dynamics here
            if self.order > 1: self.higher_order(x_in)
            else: x_out = self.net(x_in)
                
            jvp, trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
            if jvp not None: self.intloss.jvp = jvp
                
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph
    
    def higher_order(self, x):
        # NOTE: higher-order in CNF is handled at the CNF level, to refactor
        x_new = []
        size_order = x.size(1) // self.order
        for i in range(1, self.order):
            x_new += [x[:, size_order*i:size_order*(i+1)]]
        x_new += [self.m(x)]
        return torch.cat(x_new, 1).to(x)
    