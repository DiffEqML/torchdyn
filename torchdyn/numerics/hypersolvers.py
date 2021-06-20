import torch
import torch.nn as nn

class HyperEuler(SolverTemplate):
   def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.hypernet = hypernet

    def step(self, f, x, t, dt, k1=None):
        if k1 == None: k1 = f(t, x)
        x_sol = x + dt * k1 + dt**2 * self.hypernet(t, x)
        return None, None, x_sol
