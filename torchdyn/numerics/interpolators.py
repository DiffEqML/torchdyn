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

"""Contains several Interpolator classes"""

import torch
from torchdyn.numerics.solvers._constants import construct_4th
class Interpolator:
    def __init__(self, order):
        self.order = order

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other interpolator tensors are on the same device with compatible dtypes"
        if self.bmid is not None: self.bmid = self.bmid.to(x) 
        return x, t_span

    def fit(self, f0, f1, x0, x1, t, dt, **kwargs):
        pass

    def evaluate(self, coefs, t0, t1, t):
        "Evaluates a generic interpolant given coefs between [t0, t1]."
        theta = (t - t0) / (t1 - t0)
        result = coefs[0] + theta * coefs[1] 
        theta_power = theta
        for coef in coefs[2:]:
            theta_power = theta_power * theta
            result += theta_power * coef
        return result


class Linear(Interpolator):
    def __init__(self):
        raise NotImplementedError


class ThirdHermite(Interpolator):
    def __init__(self):
        super().__init__(order=3)
        raise NotImplementedError


class FourthOrder(Interpolator):
    def __init__(self, dtype):
        """4th order interpolation scheme."""
        super().__init__(order=4)
        self.bmid = construct_4th(dtype)

    def fit(self, dt, f0, f1, x0, x1, x_mid, **kwargs):
        c1 = 2 * dt * (f1 - f0) - 8 * (x1 + x0) + 16 * x_mid
        c2 = dt * (5 * f0 - 3 * f1) + 18 * x0 + 14 * x1 - 32 * x_mid
        c3 = dt * (f1 - 4 * f0) - 11 * x0 - 5 * x1 + 16 * x_mid
        c4 = dt * f0
        c5 = x0
        return [c5, c4, c3, c2, c1]



INTERP_DICT = {'4th': FourthOrder}


def str_to_interp(solver_name, dtype=torch.float32):
    "Transforms string specifying desired interpolation scheme into an instance of the Interpolator class."
    interpolator = INTERP_DICT[solver_name]
    return interpolator(dtype)