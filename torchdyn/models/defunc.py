import torch
import torch.nn as nn

class DEFuncTemplate(nn.Module):
    """Differential Equation Function template.

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
    :param func_type: {'stable', 'higher_order'}. Specifies special variants of the neural DE. Refer to the documentation for more information on the `stable` variant.
    :type func_type: str
    """
    def __init__(self, model, order, func_type):
        super().__init__()  
        self.m, self.nfe, self.controlled = model, 0., False
        self.func_type, self.order = func_type, order
        
    def forward(self, s, x):
        self.nfe += 1
        if self.controlled: x = torch.cat([x, self.u], 1)
        if self.func_type == 'stable': x = self.stable_forward(s, x)
        if self.func_type == 'higher_order' and self.order > 1: x = self.horder_forward(s, x)
        else: x = self.m(x)
        # save dxds for regularization purposes
        self.dxds = x
        return x
        
    def stable_forward(self, s, x):
        with torch.set_grad_enabled(True):
            x = torch.autograd.Variable(x, requires_grad=True)
            energy = self.m(x)**2
            grad = -torch.autograd.grad(energy.sum(1), x, create_graph=True)[0]
            if self.controlled: grad = grad[:, :x.size(1)//2]
        return grad
    
    def horder_forward(self, s, x):
        x_new = []
        size_order = x.size(1)//self.order
        for i in range(self.order-1):
            x_new.append(x[:, size_order*i:size_order*(i+1)])
        x_new.append(self.m(x))
        return torch.cat(x_new, 1).to(x)

    
class DEFunc(DEFuncTemplate):
    """General Differential Equation Function variant

    :param model: neural network parametrizing the vector field
    :type model: nn.Module
    :param order: order of the differential equation
    :type order: int
    :param func_type: {'stable', 'higher_order'}. Specifies special variants of the neural DE. Refer to the documentation for more information on the `stable` variant.
    :type func_type: str
    """
    def __init__(self, model, order=1, func_type='classic'):
        super().__init__(model, order, func_type)  
        
    def forward(self, s, x):
        idx_to_set = [el[0] if 'Depth' in str(el[1]) else -1 for el in list(self.m.named_children())]
        for idx in idx_to_set:
            if int(idx) > -1: self.m[int(idx)]._set_s(s)
        return super().forward(s, x)

class Augmenter(nn.Module):
    """Augmentation class. Can handle several types of augmentation strategies for Neural DEs.

    :param augment_dims: number of augmented dimensions to initialize
    :type augment_dims: int
    :param augment_idx: index of dimension to augment
    :type augment_idx: int
    :param augment_func: nn.Module applied to the input data of dimension `d` to determine the augmented initial condition of dimension `d + a`.
                        `a` is defined implicitly in `augment_func` e.g. augment_func=nn.Linear(2, 5) augments a 2 dimensional input with 3 additional dimensions.
    :type augment_func: nn.Module
    """
    def __init__(self, augment_dims: int = 5, augment_idx: int = 1, augment_func=None):
        super().__init__()
        self.augment_dims, self.augment_idx, self.augment_func = augment_dims, augment_idx, augment_func

    def forward(self, x: torch.Tensor):
        if not self.augment_func:
            new_dims = list(x.shape)
            new_dims[self.augment_idx] = self.augment_dims
            x = torch.cat([x, torch.zeros(new_dims).to(x)],
                          self.augment_idx)
        else:
            x = torch.cat([x, self.augment_func(x).to(x)],
                          self.augment_idx)
        return x

class DepthCat(nn.Module):
    """Depth variable `s` concatenation module. Allows for easy concatenation of `s` each call of the numerical solver, at specified layers of the DEFunc.

    :param idx_cat: index of the data dimension to concatenate `s` to.
    :type idx_cat: int
    """
    def __init__(self, idx_cat=1):
        super().__init__()
        self.idx_cat = idx_cat
    
    def _set_s(self, s):
        self.s = s
        
    def forward(self, x):
        s_shape = list(x.shape); s_shape[self.idx_cat] = 1
        self.s = self.s*torch.ones(s_shape).to(x)
        return torch.cat([x, self.s], self.idx_cat).to(x)