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
from torchdyn.numerics.solvers.ode import Euler, Midpoint, RungeKutta4

class HyperEuler(Euler):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.stepping_class = 'fixed'
        self.op1 = self.order + 1

    def step(self, f, x, t, dt, k1=None):
        _, x_sol, _ = super().step(f, x, t, dt, k1)
        return None, x_sol + dt**(self.op1) * self.hypernet(t, x), None
    
class HyperMidpoint(Midpoint):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.stepping_class = 'fixed'
        self.op1 = self.order + 1

    def step(self, f, x, t, dt, k1=None):
        _, x_sol, _ = super().step(f, x, t, dt, k1)
        return None, x_sol + dt**(self.op1) * self.hypernet(t, x), None

class HyperRungeKutta4(RungeKutta4):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.op1 = self.order + 1

    def step(self, f, x, t, dt, k1=None):
        _, x_sol, _ = super().step(f, x, t, dt, k1)
        return None, x_sol + dt**(self.op1) * self.hypernet(t, x), None
