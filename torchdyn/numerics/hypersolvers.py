import torch
import torch.nn as nn
from torchdyn.numerics.solvers import  Euler, RungeKutta4

class HyperEuler(Euler):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.stepping_class = 'fixed'

    def step(self, f, x, t, dt, k1=None):
        _, _, x_sol = super().step(f, x, t, dt, k1)
        return None, None, x_sol + dt**2 * self.hypernet(t, x)

class HyperRungeKutta4(RungeKutta4):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet

    def step(self, f, x, t, dt, k1=None):
        _, _, x_sol = super().step(f, x, t, dt, k1)
        return None, None, x_sol + dt**5 * self.hypernet(t, x)
