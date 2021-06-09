from inspect import getfullargspec
import torch
from torch.autograd import Function, grad
from torchcde import NaturalCubicSpline, natural_cubic_spline_coeffs
from torchdyn.functional.odeint import odeint, backward_adjoint_odeint

# TODO: t_span and t_eval should be type-checked, so that we are sure
# gradients w.r.t t_eval are of same shape and type as t_eval
def _gather_odefunc_adjoint(vf, vf_params, solver, atol, rtol):
    class ODEFunction(Function):
        @staticmethod
        def forward(ctx, vf_params, x0, t_span, t_eval=[]):
            t_sol, sol = odeint(f=vf, x=x0, t_span=t_span, t_eval=t_eval, solver=solver, atol=atol, rtol=rtol)
            ctx.save_for_backward(sol, t_sol, t_span)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_sol, t_span = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

            # initialize adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)

            # flatten
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

                    #TODO: expand and improve edge-case checks
                    dμ = torch.cat([el.flatten() for el in dμ], dim=-1)
                    if dt == None: dt = torch.zeros(1).to(t)
                return torch.cat([dx.flatten(), dλ.flatten(), dμ.flatten(), dt])

            # solve the adjoint equation
            n_elements = (xT_nel, λT_nel, μT_nel)
            dLdt = torch.zeros(len(t_sol)).to(xT)
            dLdt[-1] = λtT
            for i in range(len(t_sol) - 1, 0, -1):
                t_adj_sol, A = backward_adjoint_odeint(adjoint_dynamics, A, -t_sol[i - 1:i + 1].flip(0),
                                              n_elements, solver, t_eval=[], atol=atol, rtol=rtol)
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
            return (μ, λ, λ_tspan, None)

    return ODEFunction


#TODO: introduce `t_span` grad as above
def _gather_odefunc_interp_adjoint(vf, vf_params, solver, atol, rtol):
    class ODEFunction(Function):
        @staticmethod
        def forward(ctx, vf_params, x0, t_span, t_eval=[]):
            t_sol, sol = odeint(f=vf, x=x0, t_span=t_span, t_eval=t_eval, solver=solver)
            ctx.save_for_backward(sol, t_sol, t_span)
            return t_sol, sol

        @staticmethod
        def backward(ctx, *grad_output):
            sol, t_sol, t_span = ctx.saved_tensors
            vf_params = torch.cat([p.contiguous().flatten() for p in vf.parameters()])

            # initialize adjoint state
            xT, λT, μT = sol[-1], grad_output[-1][-1], torch.zeros_like(vf_params)
            λT_nel, μT_nel = λT.numel(), μT.numel()
            xT_shape, λT_shape, μT_shape = xT.shape, λT.shape, μT.shape
            A = torch.cat([λT.flatten(), μT.flatten()])

            # create dense solution
            spline_coeffs = natural_cubic_spline_coeffs(t_sol.to(sol), sol.permute(1, 0, 2).detach())
            x_spline = NaturalCubicSpline(t_span, spline_coeffs)

            # define adjoint dynamics
            def adjoint_dynamics(t, A):
                x = x_spline.evaluate(t).requires_grad_(True)
                λ, μ = A[:λT_nel], A[-μT_nel:]
                λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
                with torch.set_grad_enabled(True):
                    dx = vf(t, x)
                    dλ = grad(dx, x, -λ, allow_unused=True, retain_graph=True)[0]
                    dμ = tuple(grad(dx, tuple(vf.parameters()), -λ, allow_unused=True, retain_graph=False))
                    dμ = torch.cat([el.flatten() for el in dμ], dim=-1)
                return torch.cat([dλ.flatten(), dμ.flatten()])

            # solve the adjoint equation
            n_elements = (λT_nel, μT_nel)
            for i in range(len(t_sol) - 1, 0, -1):
                t_adj_sol, A = backward_adjoint_odeint(adjoint_dynamics, A, -t_sol[i - 1:i + 1].flip(0),
                                              n_elements, solver, t_eval=[], atol=atol, rtol=rtol)
                # prepare adjoint state for next interval
                A = torch.cat([A[-1, :λT_nel], A[-1, -μT_nel:]])
                A[:λT_nel] += grad_output[-1][i - 1].flatten()

            λ, μ = A[:λT_nel], A[-μT_nel:]
            λ, μ = λ.reshape(λT.shape), μ.reshape(μT.shape)
            return (μ, λ, None, None)

    return ODEFunction