"Contains ODE solvers. The stateful design allows users to modify or tweak Tableaus during training"
import torch
import torch.nn as nn
from torchdyn.numerics._constants import construct_rk4, construct_dopri5, construct_tsit5


class SolverTemplate(nn.Module):
    def __init__(self, order, min_factor=0.2, max_factor=10., safety=0.9):
        super().__init__()
        self.order = order
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])

    def step(self, x):
        pass


class RungeKutta4(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=4)
        self.dtype = dtype
        self.stepping_class = stepping_class='fixed'
        self.c, self.a, self.bsol, self.berr = construct_rk4(self.dtype)

    def step(self, f, x, t, dt, k1=None):
        c, a, bsol = self.c.to(x), [a.to(x) for a in self.a], self.bsol.to(x)

        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * (a[0] * k1))
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4)
        return k3, None, x_sol


class DormandPrince45(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=6)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.c, self.a, self.bsol, self.berr = construct_dopri5(self.dtype)

    def step(self, f, x, t, dt, k1=None):
        c, a, bsol, berr = self.c.to(x), [a.to(x) for a in self.a], self.bsol.to(x), self.berr.to(device)

        if k1 == None: k1 = f(t, x)

        k2 = f(t + c[0] * dt, x + dt * a[0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = f(t + c[3] * dt, x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = f(t + c[4] * dt,
               x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][
                   4] * k5)
        k7 = f(t + c[5] * dt,
               x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][
                   4] * k5 + dt * a[5][5] * k6)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        x_err = x + dt * (
                berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        return k7, x_sol, x_err


class Tsitouras45(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=6)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.c, self.a, self.bsol, self.berr = construct_tsit5(self.dtype)

    def step(self, f, x, t, dt, k1=None):
        c, a, bsol, berr = self.c.to(x), [a.to(x) for a in self.a], self.bsol.to(x), self.berr.to(device)

        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * a[0][0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = f(t + c[3] * dt, x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = f(t + c[4] * dt, x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][4] * k5)
        k7 = f(t + c[5] * dt, x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][4] * k5 + dt * a[5][5] * k6)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        x_err = x + dt * (berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        return k7, x_sol, x_err
