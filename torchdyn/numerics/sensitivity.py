from inspect import getfullargspec
import torch
from torch.autograd import Function, grad
from torchcde import NaturalCubicSpline, natural_cubic_coeffs
from torchdyn.numerics.odeint import odeint


# TODO (qol): if `t_sol` contains additional solution points other than the specified ones in `t_span`
# use those to interpolate
# TODO: optimize and make conditional gradient computations w.r.t end times
def _gather_odefunc_adjoint(vf, vf_params, solver, atol, rtol, 
                            solver_adjoint, atol_adjoint, rtol_adjoint):
    class _ODEProblemFunc(Function):
        "Underlying autograd.Function. All ODEProblem forwards are the same, each backward various depending on the sensitivity algorithm"
        @staticmethod
        def forward(ctx, vf_params, x, t_span):
            t_sol, sol = odeint(vf, x, t_span, solver, atol, rtol)
            ctx.save_for_backward(sol, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])
            # initialize flattened adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            xT_nel, λT_nel, μT_nel = xT.numel(), λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape

            λT_flat = λT.flatten()
            λtT = λT_flat @ vf(t_sol[-1], xT).flatten()

            A = torch.cat([xT.flatten(), λT_flat, μT.flatten(), λtT[None]])

            def adjoint_dynamics(t, A):
                x, λ, μ = A[:xT_nel], A[xT_nel:xT_nel+λT_nel], A[-μT_nel-1:-1]
                x, λ, μ = x.reshape(xT.shape), λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    x, t = x.requires_grad_(True), t.requires_grad_(True)
                    dx = vf(t, x)
                    dλ, dt, *dμ = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -λ,
                                    allow_unused=True, retain_graph=False))

                    dμ = torch.cat([el.flatten() if el is not None else torch.zeros(1) 
                                    for el in dμ], dim=-1)
                    if dt == None: dt = torch.zeros(1).to(t)
                return torch.cat([dx.flatten(), dλ.flatten(), dμ.flatten(), dt])

            # solve the adjoint equation
            n_elements = (xT_nel, λT_nel, μT_nel)
            dLdt = torch.zeros(len(t_sol)).to(xT)
            dLdt[-1] = λtT
            for i in range(len(t_sol) - 1, 0, -1):
                t_adj_sol, A = odeint(adjoint_dynamics, A, t_sol[i - 1:i + 1].flip(0), 
                                      solver_adjoint, atol=atol_adjoint, rtol=rtol_adjoint)
                # prepare adjoint state for next interval

                #TODO: reuse vf_eval for dLdt calculations
                xt = A[-1, :xT_nel].reshape(xT_shape)
                dLdt_ = A[-1, xT_nel:xT_nel + λT_nel]@vf(t_sol[i], xt).flatten()
                A[-1, -1:] -= grad_output[0][i - 1]
                dLdt[i-1] = dLdt_

                A = torch.cat([A[-1, :xT_nel], A[-1, xT_nel:xT_nel + λT_nel], A[-1, -μT_nel-1:-1], A[-1, -1:]])
                A[xT_nel:xT_nel + λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[xT_nel:xT_nel + λT_nel], A[-μT_nel-1:-1]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            λ_tspan = torch.stack([dLdt[0], dLdt[-1]])
            return (μ, λ, λ_tspan, None, None)

    return _ODEProblemFunc


#TODO: introduce `t_span` grad as above
#TODO: introduce option to interpolate on all solution points evaluated
def _gather_odefunc_interp_adjoint(vf, vf_params, solver, atol, rtol, 
                                solver_adjoint, atol_adjoint, rtol_adjoint):
    class _ODEProblemFunc(Function):
        @staticmethod
        def forward(ctx, vf_params, x, t_span):
            t_sol, sol = odeint(vf, x, t_span, solver, atol, rtol, return_all_eval=True)
            ctx.save_for_backward(sol, t_span, t_sol)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_span, t_sol = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

            # initialize adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            λT_nel, μT_nel = λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape
            A = torch.cat([λT.flatten(), μT.flatten()])
            spline_coeffs = natural_cubic_coeffs(x=sol.permute(1, 0, 2).detach(), t=t_sol)
            x_spline = NaturalCubicSpline(coeffs=spline_coeffs, t=t_sol)

            # define adjoint dynamics
            def adjoint_dynamics(t, A):
                x = x_spline.evaluate(t)[:,0] 
                x, t = x.requires_grad_(True), t.requires_grad_(True)
                λ, μ = A[:λT_nel], A[-μT_nel:]
                λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    dx = vf(t, x)
                    dλ, dt, *dμ = tuple(grad(dx, (x, t) + tuple(vf.parameters()), -λ,
                                        allow_unused=True, retain_graph=False))
                    dμ = torch.cat([el.flatten() if el is not None else torch.zeros(1) 
                                    for el in dμ], dim=-1)
                return torch.cat([dλ.flatten(), dμ.flatten()])

            # solve the adjoint equation
            n_elements = (λT_nel, μT_nel)
            for i in range(len(t_span) - 1, 0, -1):
                t_adj_sol, A = odeint(adjoint_dynamics, A, t_span[i - 1:i + 1].flip(0), solver, atol=atol, rtol=rtol)
                # prepare adjoint state for next interval
                A = torch.cat([A[-1, :λT_nel], A[-1, -μT_nel:]])
                A[:λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[:λT_nel], A[-μT_nel:]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            return (μ, λ, None, None)

    return _ODEProblemFunc