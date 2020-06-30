import torch
import torch.nn as nn

class DEFunc(nn.Module):
    """Differential Equation Function Wrapper.

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
   """
    def __init__(self, model, order=1):
        super().__init__()  
        self.m, self.nfe,  = model, 0.
        self.order, self.intloss, self.sensitivity = order, None, None

    def forward(self, s, x):
        self.nfe += 1        
        # set `s` depth-variable to DepthCat modules
        for _, module in self.m.named_modules(): 
            if hasattr(module, 's'):
                module.s = s

        # if-else to handle autograd training with integral loss propagated in x[:, 0]
        if (not self.intloss is None) and self.sensitivity == 'autograd':
            x_dyn = x[:, 1:]
            dlds = self.intloss(x_dyn)
            if len(dlds.shape) == 1: dlds = dlds[:, None]
            if self.order > 1: x_dyn = self.horder_forward(s, x_dyn)
            else: x_dyn = self.m(x_dyn)
            self.dxds = x_dyn
            return torch.cat([dlds, x_dyn], 1).to(x_dyn)
        
        # regular forward
        else:   
            if self.order > 1: x = self.horder_forward(s, x)
            else: x = self.m(x)
            self.dxds = x
            return x

    def horder_forward(self, s, x):
        # NOTE: higher-order in CNF is handled at the CNF level, to refactor
        x_new = []
        size_order = x.size(1) // self.order
        for i in range(1, self.order):
            x_new += [x[:, size_order*i:size_order*(i+1)]]
        x_new += [self.m(x)]
        return torch.cat(x_new, 1).to(x)
    