"""Contains several Interpolator classes"""

import torch

class Interpolator:
    def __init__(self, order):
        self.order = order
    def fit_interpolation():
        pass
    def evaluate(self, coefs, t0, t1, t):
        pass


class Linear(Interpolator):
    def __init__(self):
        raise NotImplementedError


class ThirdHermite(Interpolator):
    def __init__(self):
        raise NotImplementedError
    # def hermite_interp(self, t, t0, t1, f0, f1, x0, x1):
    #     "Fits and evaluates Hermite interpolation, (6.7) in Hairer I."
    #     dt = t1 - t0
    #     theta = (t - t0) / (t1 - t0)
    #     return (1-theta)*x0+theta*x1 + theta*(theta - 1)*((1-2*theta)*(x1-x0)+(theta-1)*dt*f0+theta*dt*f1)

class FourthOrder(Interpolator):
    def __init__(self):
        super().__init__(order=4)

    def fit_interpolation():
        a = 2 * dt * (k4 - k1) - 8 * (x1 + x0) + 16 * x_mid
        b = dt * (5 * k1 - 3 * k4) + 18 * x0 + 14 * x1 - 32 * x_mid
        c = dt * (k4 - 4 * k1) - 11 * x0 - 5 * x1 + 16 * x_mid
        d = dt * k1
        e = x0
        return [e, d, c, b, a]

    def evaluate(self, coefs, t0, t1, t):
        x = (t - t0) / (t1 - t0)
        x = x.to(coefs[0].dtype)

        total = coefs[0] + x * coefs[1]
        x_power = x
        for coefficient in coefs[2:]:
            x_power = x_power * x
            total = total + x_power * coefficient
        return total

INTERP_DICT = {'4th': FourthOrder, 'hermite': ThirdHermite}


def str_to_interp(solver_name, dtype=torch.float32):
    "Transforms string specifying desired interpolation scheme into an instance of the Interpolator class."
    interpolator = INTERP_DICT[solver_name]
    return interpolator(dtype)