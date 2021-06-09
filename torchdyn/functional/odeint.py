from typing import List
import torch
from torchtyping import TensorType

from torchdyn.numerics.constants import *
from torchdyn.functional.utils import norm, init_step, adapt_step, EventState


# TODO: merge t_eval with t_span for clarity
def odeint(f, x, t_span, solver, t_eval=[], atol=1e-3, rtol=1e-3):
	stepping_class = solver.stepping_class

	# preprocessing steps
	if type(t_span) == List:
		print("Jagged parallel IVPs not compatible with adjoint")
		return jagged_fixed_odeint(f, x, t_span, solver)

	# `Tensor' t_span integration types
	else:
		t_shape = t_span.shape
		if stepping_class == 'fixed' and len(t_shape) == 1:
			return fixed_odeint(f, x, t_span, solver)
		elif stepping_class == 'fixed':
			# ensure compatibility with backsolve is preserved here
			return shifted_fixed_odeint(f, x, t_span, solver)
		else:
			return adaptive_odeint(f, x, t_span, solver, t_eval, atol, rtol)


# TODO: ensure compat with float32
def adaptive_odeint(f, x, t_span, solver, t_eval=[], atol=1e-3, rtol=1e-3):
	x_shape = x.shape
	dtype = x.dtype
	device = x.device

	ckpt_counter = 0
	ckpt_flag = False
	t = t_span[:1].to(device)

	solver.a = [a_.to(device) for a_ in solver.a]
	solver.c = [c_.to(device) for c_ in solver.c]
	solver.bsol = [bsol_.to(device) for bsol_ in solver.bsol]
	if solver.berr is not None: solver.berr = [berr_.to(device) for berr_ in solver.berr]
	solver.safety = solver.safety.to(device)
	solver.min_factor = solver.min_factor.to(device)
	solver.max_factor = solver.max_factor.to(device)

	################## initial step size setting ################
	k1 = f(t, x)
	dt = init_step(f, k1, x, t, solver.order, atol, rtol)

	#### init solution & time vector ####
	eval_times = [t]
	sol = [x]
	while t < t_span[-1]:
		############### checkpointing ###############################
		if t + dt > t_span[-1]:
			dt = t_span[-1] - t
		if t_eval is not None:
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				# save old dt and raise "checkpoint" flag
				dt_old, ckpt_flag = dt, True
				dt = t_eval[ckpt_counter] - t
				ckpt_counter += 1


		#print(t, flush=True, end='\r')

		f_new, x_new, x_app = solver.step(f, x, t, dt, k1=k1)

		################# compute error #############################
		error = x_new - x_app
		error_tol = atol + rtol * torch.max(x.abs(), x_new.abs())
		error_ratio = norm(error / error_tol)
		accept_step = error_ratio <= 1
		if accept_step:
			t = t + dt
			x = x_new
			sol.append(x)
			eval_times.append(t)
			k1 = f_new

		################ stepsize control ###########################
		# reset "dt" in case of ceckpoint
		if ckpt_flag:
			dt = dt_old - dt
			ckpt_flag = False

		dt = adapt_step(dt, error_ratio,
						solver.safety,
						solver.min_factor,
						solver.max_factor,
						solver.order)

	return torch.cat(eval_times), torch.stack(sol)


# TODO: update dt
def fixed_odeint(f, x: TensorType["batch", "dim1"], t_span: TensorType["t_len"], solver):
	"Solves a single IVP with fixed-step methods"
	t, T = t_span[..., 0], t_span[..., -1]
	dt = t_span[..., 1] - t
	sol = [x]

	steps = 1
	while steps <= len(t_span) - 1:
		_, _, x = solver.step(f, x, t, dt)
		sol.append(x)
		t = t + dt
		dt = t_span[steps] - t
		steps += 1

	return t_span, torch.stack(sol)


# TODO: update dt
def shifted_fixed_odeint(f, x: TensorType["n_segments", "batch", "dim1"],
						t_span: TensorType["n_segments", "t_len"], solver):
	"""Solves ``n_segments'' jagged IVPs in parallel with fixed-step methods. All subproblems
	have equal step sizes and number of solution points"""
	t, T = t_span[..., 0], t_span[..., -1]
	dt = t_span[..., 1] - t
	sol, k1 = [], f(t, x)

	not_converged = ~((t - T).abs() <= 1e-6).bool()
	while not_converged.any():
		x[:, ~not_converged] = torch.zeros_like(x[:, ~not_converged])
		k1, _, x = solver.step(f, x, t, dt[..., None], k1=k1)  # dt will be broadcasted on dim1
		sol.append(x)
		t = t + dt
		not_converged = ~((t - T).abs() <= 1e-6).bool()
	# stacking is only possible since the number of steps in each of the ``n_segments''
	# is assumed to be the same. Otherwise require jagged tensors or a []
	return torch.stack(sol)



def jagged_fixed_odeint(f, x: TensorType["n_segments", "batch", "dim1"],
						t_span: List, solver):
	"""
	Solves ``n_segments'' jagged IVPs in parallel with fixed-step methods. Each sub-IVP can vary in number
    of solution steps and step sizes

	Args:
		f:
		x:
		t_span:
		solver:

	Returns:
		A list of `len(t_span)' containing solutions of each IVP computed in parallel.
	"""
	t, T = [t_sub[0] for t_sub in t_span], [t_sub[-1] for t_sub in t_span]
	t, T = torch.stack(t), torch.stack(T)

	dt = torch.stack([t_[1] - t0 for t_, t0 in zip(t_span, t)])
	sol = [[x_] for x_ in x]
	not_converged = ~((t - T).abs() <= 1e-6).bool()
	steps = 0
	while not_converged.any():
		_, _, x = solver.step(f, x, t, dt[..., None, None])  # dt will be to x dims

		for n, sol_ in enumerate(sol):
			sol_.append(x[n])
		t = t + dt
		not_converged = ~((t - T).abs() <= 1e-6).bool()

		steps += 1
		dt = []
		for t_, tcur in zip(t_span, t):
			if steps > len(t_) - 1:
				dt.append(torch.zeros_like(tcur))  # subproblem already solved
			else:
				dt.append(t_[steps] - tcur)

		dt = torch.stack(dt)
	# prune solutions to remove noop steps
	sol = [sol_[:len(t_)] for sol_, t_ in zip(sol, t_span)]
	return [torch.stack(sol_, 0) for sol_ in sol]


def jagged_adaptive_odeint():
	raise NotImplementedError("Adaptive-step version of ``jagged_fixed_odeint''")


# mostly similar as odeint above, with semi-norm and a wrapper for `f` as lambda t, x: -f(-t, x)
# since we assume to be calling this on backward time domains only
def backward_adjoint_odeint(f, x, t_span, n_elements, solver, t_eval=[], atol=1e-3, rtol=1e-3):
	"""Modified odeint for efficient backsolve of adjoint systems. Stepsize control uses error on (x, λ), ignoring μ"""
	x_shape, dtype, device = x.shape, x.dtype, x.device
	if len(n_elements) == 3: x_nel, λ_nel, μ_nel = n_elements  # backsolve adjoint
	else: λ_nel, μ_nel = n_elements  # interpolated adjoint
	# since time is reversed, flip f
	f_ = lambda t, x: -f(-t, x)

	ckpt_counter = 0
	t = t_span[:1]

	solver.a = [a_.to(device) for a_ in solver.a]
	solver.c = [c_.to(device) for c_ in solver.c]
	solver.bsol = [bsol_.to(device) for bsol_ in solver.bsol]
	solver.berr = [berr_.to(device) for berr_ in solver.berr]
	solver.safety = solver.safety.to(device)
	solver.min_factor = solver.min_factor.to(device)
	solver.max_factor = solver.max_factor.to(device)

	################## initial step size setting ################
	k1 = f_(t, x)
	dt = init_step(f_, k1, x, t, solver.order, atol, rtol)

	#### init solution & time vector ####
	eval_times = [t]
	sol = [x]

	while t < t_span[-1]:
		############### checkpointing ###############################
		if t + dt > t_span[-1]:
			dt = t_span[-1] - t
		if t_eval is not None:
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				dt = t_eval[ckpt_counter] - t
				ckpt_counter += 1
		f_new, x_new, x_app = solver.step(f_, x, t, dt, k1=k1)
		################# compute error #############################
		error = x_new[:-μ_nel] - x_app[:-μ_nel]
		error_tol = atol + rtol * torch.max(x[:-μ_nel].abs(), x_new[:-μ_nel].abs())
		error_ratio = norm(error / error_tol)
		accept_step = error_ratio <= 1
		if accept_step:
			t = t + dt
			x = x_new
			sol.append(x)
			eval_times.append(t)
			k1 = f_new

		################ stepsize control ###########################
		dt = adapt_step(dt, error_ratio,
						solver.safety,
						solver.min_factor,
						solver.max_factor,
						solver.order)

	return torch.cat(eval_times), torch.stack(sol)


# TODO: check why for some tols `min(....)` becomes empty in internal event finder
def odeint_hybrid(f, x, t_span, j_span, solver, callbacks, t_eval=[], atol=1e-3, rtol=1e-3, event_tol=1e-4,
				  priority='jump'):
	x_shape, dtype, device = x.shape, x.dtype, x.device
	ckpt_counter = 0
	t = t_span[:1]
	jnum = 0

	###### initial jumps ###########
	event_states = EventState([False for _ in range(len(callbacks))])

	if priority == 'jump':
		new_event_states = EventState([cb.check_event(t, x) for cb in callbacks])
		triggered_events = event_states != new_event_states
		if triggered_events > 0:
			i = min([i for i, idx in enumerate(new_event_states.evid) if idx == True])
			x = callbacks[i].jump_map(t, x)
			jnum = jnum + 1

	################## initial step size setting ################
	k1 = f(t, x)
	dt = init_step(f, k1, x, t, solver.order, atol, rtol)

	#### init solution & time vector ####
	eval_times = [t]
	sol = [x]

	while t < t_span[-1] and jnum < j_span:
		############### checkpointing ###############################
		if t + dt > t_span[-1]:
			dt = t_span[-1] - t
		if t_eval is not None:
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				dt = t_eval[ckpt_counter] - t
				ckpt_counter += 1

		f_new, x_new, x_app = solver.step(f, x, t, dt, k1=k1)

		################ callback and events ########################
		new_event_states = EventState([cb.check_event(t + dt, x_new) for cb in callbacks])
		triggered_events = event_states != new_event_states

		# if event close in on switching state in [t, t + Δt]
		if triggered_events > 0:
			t_inner = t
			dt_inner = dt
			x_inner = x
			niters = 0
			max_iters = 100  # compute as function of tolerances

			while niters < max_iters and event_tol < dt_inner:
				with torch.no_grad():

					dt_inner = dt_inner / 2
					f_new, x_, _ = solver.step(f, x_inner, t_inner, dt_inner, k1=k1)

					new_event_states = EventState([cb.check_event(t_inner + dt_inner, x_)
												   for cb in callbacks])
					triggered_events = event_states != new_event_states
					niters = niters + 1

				if triggered_events == 0:
					x_inner = x_
					sol.append(x_inner.reshape(x_shape))
					t_inner = t_inner + dt_inner
					eval_times.append(t_inner.reshape(t.shape))
					dt_inner = dt
					k1 = f_new

			x = x_inner
			t = t_inner
			i = min([i for i, x in enumerate(new_event_states.evid) if x == True])

			# save state and time BEFORE jump
			sol.append(x.reshape(x_shape))
			eval_times.append(t.reshape(t.shape))

			# apply jump func.
			x = callbacks[i].jump_map(t, x)

			# save state and time AFTER jump
			sol.append(x.reshape(x_shape))
			eval_times.append(t.reshape(t.shape))

			# reset k1
			k1 = f_new

		else:

			################# compute error #############################
			error = x_new - x_app
			error_tol = atol + rtol * torch.max(x.abs(), x_new.abs())
			error_ratio = norm(error / error_tol)
			accept_step = error_ratio <= 1

			if accept_step:
				t = t + dt
				x = x_new.float()
				sol.append(x.reshape(x_shape))
				eval_times.append(t.reshape(t.shape))
				k1 = f_new

			################ stepsize control ###########################
			dt = adapt_step(dt, error_ratio,
							solver.safety,
							solver.min_factor,
							solver.max_factor,
							solver.order)

	return torch.cat(eval_times), torch.stack(sol)
