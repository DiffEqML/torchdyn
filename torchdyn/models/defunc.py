import torch
import torch.nn as nn

class DEFunc(nn.Module):
    """Differential Equation Function Wrapper.

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
   """
    def __init__(self, model, order=1, controlled=False):
        super().__init__()  
        self.m, self.nfe,  = model, 0.
        self.controlled, self.order = controlled, order
        self.intloss, self.sensitivity = None, None

    def forward(self, s, x):
        self.nfe += 1
        # depth-concatenation routine preceded by data-control
        # TO DO set Stable `depth-var`
        if (not self.intloss is None) and self.sensitivity == 'autograd':
            if self.controlled: x = torch.cat([x, self.u[:, 1:]], 1).to(x)
        else:      
            if self.controlled: x = torch.cat([x, self.u], 1).to(x)
        idx_to_set = [el[0] if 'Depth' in str(el[1]) else -1 for el in list(self.m.named_children())]
        for idx in idx_to_set:
            if int(idx) > -1: self.m[int(idx)]._set_s(s)
                
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
        x_new = []
        size_order = x.size(1) // self.order
        for i in range(1, self.order):
            x_new += [x[:, size_order*i:size_order*(i+1)]]
        x_new += [self.m(x)]
        return torch.cat(x_new, 1).to(x).to(x)
    