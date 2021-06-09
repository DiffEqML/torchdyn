from inspect import getfullargspec
import attr
import torch
import torch.nn as nn
from typing import List
from .sensitivity import *
from .utils import WrapFunc
from .odeint import odeint
from .sensitivity import _gather_odefunc_adjoint, _gather_odefunc_interp_adjoint

#TODO: unsure if atol/rtol is best passed here. Passing it at forward might be more flexible, but is annoying altogether
# since all odefuncs will have additional in/out params. torchdiffeq does it this way
@attr.s(init=False)
class ODEProblem(nn.Module):
    def __init__(self, solver, vf, sensalg='autograd', atol=1e-4, rtol=1e-4):
        super().__init__()
        self.sensalg, self.vf, self.solver = sensalg, vf, solver
        self.atol, self.rtol = atol, rtol

        #TODO: this fails when vf does not have parameters
        if len(tuple(vf.parameters())) > 0:
            self.vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
        else:
            self.vf_params = nn.Parameter(torch.zeros(1))

        if 't' not in getfullargspec(vf.forward).args:
            self.vf = WrapFunc(vf)

        if self.sensalg == 'adjoint':  # alias .apply as direct call to preserve consistency of call signature
            self.odefunc = _gather_odefunc_adjoint(self.vf, self.vf_params, self.solver, atol, rtol).apply
        elif self.sensalg == 'interpolated_adjoint':
            self.odefunc = _gather_odefunc_interp_adjoint(self.vf, self.vf_params, self.solver, atol, rtol).apply
        else:
            def odefunc(vf_params, x0, t_span, t_eval=[]):
                return odeint(self.vf, x=x0, t_span=t_span,
                              t_eval=t_eval, solver=self.solver, atol=atol, rtol=rtol)

            self.odefunc = odefunc

    def forward(self, x0, t_span, t_eval=[]):
        x0, t_span = prep_input(x0, t_span)
        t_eval, sol = self.odefunc(self.vf_params, x0, t_span, t_eval)
        return t_eval, sol


def prep_input(x, t_span):
    if type(t_span) == list: t_span = [t.to(x) for t in t_span]
    else: t_span = t_span.to(x)
    return x, t_span