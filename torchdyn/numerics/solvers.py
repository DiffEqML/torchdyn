"""
    Contains ODE solvers, both sequential as well as time-parallel multiple shooting methods necessary for multiple-shoting layers [1]. 
    The stateful design allows users to modify or tweak each Tableau during training, ensuring compatibility with hybrid methods such as Hypersolvers [2]
    [1]: Massaroli S., Poli M. et al "Differentiable Multiple Shooting Layers." 
    [2]: Poli M., Massaroli S. et al "Hypersolvers: Toward fast continuous-depth models." NeurIPS 2020
"""

import torch
import torch.nn as nn
from torchdyn.numerics._constants import construct_rk4, construct_dopri5, construct_tsit5


class SolverTemplate(nn.Module):
    def __init__(self, order, min_factor=0.2, max_factor=2., safety=0.9):
        """[summary]

        Args:
            order ([type]): [description]
            min_factor (float, optional): [description]. Defaults to 0.2.
            max_factor ([type], optional): [description]. Defaults to 10..
            safety (float, optional): [description]. Defaults to 0.9.
        """
        super().__init__()
        self.order = order
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        device = x.device
        if self.tableau is not None:
            c, a, bsol, berr = self.tableau
            self.tableau = c.to(x), [a.to(x) for a in a], bsol.to(x), berr.to(x)
        t_span = t_span.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)     
        return x, t_span

    def step(self, f, x, t, dt, k1=None):
        pass

    def hermite_interp(self, t, t0, t1, f0, f1, x0, x1):
        "Fits and evaluates Hermite interpolation, (6.7) in Hairer I."
        dt = t1 - t0
        theta = (t - t0) / (t1 - t0)
        return (1-theta)*x0+theta*x1 + theta*(theta - 1)*((1-2*theta)*(x1-x0)+(theta-1)*dt*f0+theta*dt*f1)


class Euler(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def step(self, f, x, t, dt, k1=None):
        if k1 == None: k1 = f(t, x)
        x_sol = x + dt * k1
        return None, None, x_sol


class RungeKutta4(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=4)
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.tableau = construct_rk4(self.dtype)

    def step(self, f, x, t, dt, k1=None):
        c, a, bsol, _ = self.tableau
        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * (a[0] * k1))
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4)
        return None, None, x_sol


class AsynchronousLeapfrog(SolverTemplate):
    def __init__(self, channel_index:int=-1, stepping_class:str='fixed', dtype=torch.float32):
        super().__init__(order=2)
        self.dtype = dtype
        self.channel_index = channel_index
        self.stepping_class = stepping_class
        self.const = 1
        self.tableau = construct_rk4(self.dtype)  
        # an additional overhead, necessary to preserve a certain degree of sanity 
        # in the implementation and to avoid API bloating.
        self.x_shape = None 


    def step(self, f, xv, t, dt, k1=None):
        half_state_dim = xv.shape[-1] // 2
        x, v = xv[..., :half_state_dim], xv[..., half_state_dim:]
        if k1 == None: k1 = f(t, x)
        x1 = x + 0.5 * dt * v
        vt1 = f(t + 0.5 * dt, x1)
        v1 = 2 * self.const * (vt1 - v) + v
        x2 = x1 + 0.5 * dt * v1 
        x_sol = torch.cat([x2, v1], -1)
        if self.stepping_class == 'adaptive':
            xv_err = torch.cat([torch.zeros_like(x), v], -1)
        else:
            xv_err = None
        return None, xv_err, x_sol 


class DormandPrince45(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        """[summary]

        Args:
            dtype ([type], optional): [description]. Defaults to torch.float32.
        """
        super().__init__(order=6)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.tableau = construct_dopri5(self.dtype)

    def step(self, f, x, t, dt, k1=None):
        c, a, bsol, berr = self.tableau
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
        self.tableau = construct_tsit5(self.dtype)

    def step(self, f, x, t, dt, k1=None):
        c, a, bsol, berr = self.tableau
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

class ImplicitEuler(SolverTemplate):
    def __init__(self, dtype=torch.float32):
        super().__init__(order=1)
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.opt = torch.optim.LBFGS
        self.max_iters = 200
        self.lr = 0.1

    @staticmethod
    def _residual(f, x, t, dt, x_sol):
        f_sol = f(t, x_sol)
        return torch.sum((x_sol - x - dt*f_sol)**2)

    def step(self, f, x, t, dt, k1=None):
        x_sol = x.clone()
        x_sol = nn.Parameter(data=x_sol)
        opt = self.opt([x_sol], lr=1, max_iter=self.max_iters, max_eval=10*self.max_iters,
        tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=100, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            residual = ImplicitEuler._residual(f, x, t, dt, x_sol)
            x_sol.grad, = torch.autograd.grad(residual, x_sol, only_inputs=True, allow_unused=False)
            return residual
        opt.step(closure)
        return None, None, x_sol


class MShootingSolverTemplate(nn.Module):
    def __init__(self, coarse_method, fine_method):
        super().__init__()
        if type(coarse_method) == str: self.coarse_method = str_to_solver(coarse_method)
        if type(fine_method) == str: self.fine_method = str_to_solver(fine_method)

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        x, t_span = self.coarse_method.sync_device_dtype(x, t_span)
        x, t_span = self.fine_method.sync_device_dtype(x, t_span)  
        return x, t_span

    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        pass


class MSForward(MShootingSolverTemplate):
    """Multiple shooting solver using forward sensitivity analysis on the matching conditions of shooting parameters"""
    def __init__(self, coarse_method='euler', fine_method='rk4'):
        super().__init__(coarse_method, fine_method)

    def root_solve(self, f, x, t_span, B):
        raise NotImplementedError("Waiting for `functorch` to be merged in the stable version of Pytorch"
                                  "we need their vjp for efficient implementation of forward sensitivity"
                                  "Refer to DiffEqML/diffeqml-research/multiple-shooting-layers for a manual implementation")


class MSZero(MShootingSolverTemplate):
    def __init__(self, coarse_method='euler', fine_method='rk4'):
        """Multiple shooting solver using Parareal updates (zero-order approximation of the Jacobian)

        Args:
            coarse_method (str, optional): [description]. Defaults to 'euler'.
            fine_method (str, optional): [description]. Defaults to 'rk4'.
        """
        super().__init__(coarse_method, fine_method)

    # TODO (qol): extend to time-variant ODEs by using shifted_odeint
    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        dt, n_subinterv = t_span[1] - t_span[0], len(t_span)
        sub_t_span = torch.linspace(0, dt, fine_steps).to(x)
        i = 0
        while i <= maxiter:
            i += 1
            B_coarse = odeint_func(f, B[i-1:], sub_t_span, solver=self.coarse_method)[1][-1]
            B_fine = odeint_func(f, B[i-1:], sub_t_span, solver=self.fine_method)[1][-1]
            B_out = torch.zeros_like(B)
            B_out[:i] = B[:i]
            B_in = B[i-1]
            for m in range(i, n_subinterv):
                B_in = odeint_func(f, B_in, sub_t_span, solver=self.coarse_method)[1][-1]
                B_in = B_in - B_coarse[m-i] + B_fine[m-i]
                B_out[m] = B_in
            B = B_out
        return B


class MSBackward(MShootingSolverTemplate):
    def __init__(self, coarse_method='euler', fine_method='rk4'):
        """Multiple shooting solver using discrete adjoints for the Jacobian

        Args:
            coarse_method (str, optional): [description]. Defaults to 'euler'.
            fine_method (str, optional): [description]. Defaults to 'rk4'.
        """
        super().__init__(coarse_method, fine_method)

    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        dt, n_subinterv = t_span[1] - t_span[0], len(t_span)
        sub_t_span = torch.linspace(0, dt, fine_steps).to(x)
        i = 0
        B = B.requires_grad_(True)
        while i <= maxiter:
            print(i)
            i += 1
            B_fine = odeint_func(f, B[i-1:], sub_t_span, solver=self.fine_method)[1][-1]
            B_out = torch.zeros_like(B)
            B_out[:i] = B[:i]
            B_in = B[i-1]
            for m in range(i, n_subinterv):
                # instead of jvps here the full jacobian can be computed and the vector products
                # which involve `B_in` can be performed. Trading memory ++ for speed ++
                J_blk = torch.autograd.grad(B_fine[m-1], B, B_in - B[m-1], retain_graph=True)[0][m-1]
                B_in = B_fine[m-1] + J_blk
                B_out[m] = B_in
            del B # manually free graph
            B = B_out
        return B


class ParallelImplicitEuler(MShootingSolverTemplate):
    def __init__(self, coarse_method='euler', fine_method='euler'):
        """Parallel Implicit Eurler Method
        """
        super().__init__(coarse_method, fine_method)
        self.solver = torch.optim.LBFGS
        self.max_iters = 200
        self.lr = 0.1

    def sync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device with compatible dtypes"
        return x, t_span

    @staticmethod
    def _residual(f, B, t_span):
        dt = t_span[1:] - t_span[:-1]
        F = f(0., B[1:])
        return torch.sum((B[1:] - B[:-1] - dt[:, None, None] * F) ** 2)

    # TODO (qol): extend to time-variant ODEs by model parallelization
    def root_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        B = B.clone()
        B = nn.Parameter(data=B)
        solver = self.solver([B], lr=1, max_iter=self.max_iters, max_eval=10 * self.max_iters,
                             tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=100,
                             line_search_fn='strong_wolfe')

        def closure():
            solver.zero_grad()
            residual = ParallelImplicitEuler._residual(f, B, t_span)
            B.grad, = torch.autograd.grad(residual, B, only_inputs=True, allow_unused=False)
            return residual

        solver.step(closure)
        return B


SOLVER_DICT = {'euler': Euler,
               'rk4': RungeKutta4, 'rk-4': RungeKutta4, 'RungeKutta4': RungeKutta4,
               'dopri5': DormandPrince45, 'DormandPrince45': DormandPrince45, 'DormandPrince5': DormandPrince45,
               'tsit5': Tsitouras45, 'Tsitouras45': Tsitouras45, 'Tsitouras5': Tsitouras45,
               'ieuler': ImplicitEuler, 'implicit_euler': ImplicitEuler,
               'alf': AsynchronousLeapfrog, 'AsynchronousLeapfrog': AsynchronousLeapfrog}


MS_SOLVER_DICT = {'mszero': MSZero, 'zero': MSZero, 'parareal': MSZero, 
                  'msbackward': MSBackward, 'backward': MSBackward, 'discrete-adjoint': MSBackward,
                  'ieuler': ParallelImplicitEuler, 'parallel-implicit-euler': ParallelImplicitEuler}


def str_to_solver(solver_name, dtype=torch.float32):
    "Transforms string specifying desired solver into an instance of the Solver class."
    solver = SOLVER_DICT[solver_name]
    return solver(dtype)


def str_to_ms_solver(solver_name, dtype=torch.float32):
    "Returns MSSolver class corresponding to a given string."
    solver = MS_SOLVER_DICT[solver_name]
    return solver()
