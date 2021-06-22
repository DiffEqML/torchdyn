"""
	Functional API of ODE integration routines, with specialized functions for different options
	`odeint` and `odeint_mshooting` prepare and redirect to more specialized routines, detected automatically.
"""
from inspect import getargspec
from typing import List, Union, Callable
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn

from torchdyn.numerics.solvers import AsynchronousLeapfrog, str_to_solver, str_to_ms_solver
from torchdyn.numerics.utils import norm, init_step, adapt_step, EventState


def odeint(f:Callable, x:Tensor, t_span:Union[List, Tensor], solver:Union[str, nn.Module], atol:float=1e-3, rtol:float=1e-3, 
		   verbose:bool=False, return_all_eval:bool=False):
	if t_span[1] < t_span[0]: # time is reversed
		if verbose: warn("You are integrating on a reversed time domain, adjusting the vector field automatically")
		f_ = lambda t, x: -f(-t, x)
		t_span = -t_span
	else: f_ = f

	if type(t_span) == list: t_span = torch.cat(t_span)
	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str:
		solver = str_to_solver(solver, x.dtype)
	x, t_span = solver.sync_device_dtype(x, t_span)
	stepping_class = solver.stepping_class

	# access parallel integration routines with different t_spans for each sample in `x`.
	if len(t_span.shape) > 1:
		raise NotImplementedError("Parallel routines not implemented yet, check experimental versions of `torchdyn`")
	# odeint routine with a single t_span for all samples
	elif len(t_span.shape) == 1:
		if stepping_class == 'fixed': 
			return _fixed_odeint(f_, x, t_span, solver) 
		elif stepping_class == 'adaptive':
			t = t_span[0]
			k1 = f(t, x)
			dt = init_step(f, k1, x, t, solver.order, atol, rtol)
			return _adaptive_odeint(f_, k1, x, dt, t_span, solver, atol, rtol, return_all_eval)


# TODO (qol) state augmentation for symplectic methods 
def odeint_symplectic(f:Callable, x:Tensor, t_span:Union[List, Tensor], solver:Union[str, nn.Module], atol:float=1e-3, rtol:float=1e-3, 
		   verbose:bool=False, return_all_eval:bool=False):
	if t_span[1] < t_span[0]: # time is reversed
		if verbose: warn("You are integrating on a reversed time domain, adjusting the vector field automatically")
		f_ = lambda t, x: -f(-t, x)
		t_span = -t_span
	else: f_ = f
	if type(t_span) == list: t_span = torch.cat(t_span)

	# instantiate the solver in case the user has specified preference via a `str` and ensure compatibility of device ~ dtype
	if type(solver) == str:
		solver = str_to_solver(solver, x.dtype)
	x, t_span = solver.sync_device_dtype(x, t_span)
	stepping_class = solver.stepping_class

	# additional bookkeeping for symplectic solvers
	if not hasattr(f, 'order'):
		raise RuntimeError('The system order should be specified as an attribute `order` of `vector_field`')
	if isinstance(solver, AsynchronousLeapfrog) and f.order == 2: 
		raise RuntimeError('ALF solver should be given a vector field specified as a first-order symplectic system: v = f(t, x)')
	solver.x_shape = x.shape[-1] // 2

	# access parallel integration routines with different t_spans for each sample in `x`.
	if len(t_span.shape) > 1:
		raise NotImplementedError("Parallel routines not implemented yet, check experimental versions of `torchdyn`")
	# odeint routine with a single t_span for all samples
	elif len(t_span.shape) == 1:
		if stepping_class == 'fixed': 
			return _fixed_odeint(f_, x, t_span, solver) 
		elif stepping_class == 'adaptive':
			t = t_span[0]
			if f.order == 1: 
				pos = x[..., : solver.x_shape]
				k1 = f(t, pos)
				dt = init_step(f, k1, pos, t, solver.order, atol, rtol)
			else:
				 k1 = f(t, x)
				 dt = init_step(f, k1, x, t, solver.order, atol, rtol)	
			return _adaptive_odeint(f_, k1, x, dt, t_span, solver, atol, rtol, return_all_eval)


def odeint_mshooting(f:Callable, x:Tensor, t_span:Tensor, solver:Union[str, nn.Module], atol:float=1e-3, rtol:float=1e-3,
					 fine_steps=4, B_initialization='manual', maxiter=100):
		coarse_solver, fine_solver = str_to_ms_solver(solver)
		# initialize solver
		solver = solver()

		B = solver.root_solve(f, x, t_span, B)
		return B, t_span


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


# TODO (qol): interpolation option instead of checkpoint
def _adaptive_odeint(f, k1, x, dt, t_span, solver, atol=1e-4, rtol=1e-4, return_all_eval=False):
	"""
	
	Notes:
	(1) We check if the user wants all evaluated solution points, not only those
	corresponding to times in `t_span`. This is automatically set to `True` when `odeint`
	is called for interpolated adjoints

	Args:
		f ([type]): [description]
		k1 ([type]): [description]
		x ([type]): [description]
		dt ([type]): [description]
		t_span ([type]): [description]
		solver ([type]): [description]
		atol ([type], optional): [description]. Defaults to 1e-4.
		rtol ([type], optional): [description]. Defaults to 1e-4.
		return_all_eval (bool, optional): [description]. Defaults to False.

	Returns:
		[type]: [description]
	
	"""
	t_eval = t_span[1:]
	t = t_span[:1]
	ckpt_counter, ckpt_flag = 0, False	
	eval_times, sol = [t], [x]
	while t < t_span[-1]:
		############### checkpointing ###############################
		if t + dt > t_span[-1]:
			dt = t_span[-1] - t
		if t_eval is not None:
			if (ckpt_counter < len(t_eval)) and (t + dt > t_eval[ckpt_counter]):
				# save old dt and raise "checkpoint" flag
				dt_old, ckpt_flag = dt, True
				dt = t_eval[ckpt_counter] - t				
		f_new, x_new, x_app = solver.step(f, x, t, dt, k1=k1)
		################# compute error #############################
		error = x_new - x_app
		
		error_tol = atol + rtol * torch.max(x.abs(), x_new.abs())
		error_ratio = norm(error / error_tol)
		accept_step = error_ratio <= 1
		if accept_step:
			t, x = t + dt, x_new
			if t == t_eval[ckpt_counter] or return_all_eval: # note (1)
				sol.append(x_new)
				eval_times.append(t)
				ckpt_counter += 1	
			k1 = f_new 
		################ stepsize control ###########################
		# reset "dt" in case of checkpoint
		if ckpt_flag:
			dt = dt_old - dt
			ckpt_flag = False
		dt = adapt_step(dt, error_ratio,
						solver.safety,
						solver.min_factor,
						solver.max_factor,
						solver.order)
		# TODO: insert safety mechanism for small or large steps
		# dt = max(dt, torch.tensor(1e-5).to(dt))
	return torch.cat(eval_times), torch.stack(sol)


def _fixed_odeint(f, x, t_span, solver):
	"""Solves IVPs with same `t_span`, using fixed-step methods

	Args:
		f ([type]): [description]
		x ([type]): [description]
		t_span ([type]): [description]
		solver ([type]): [description]

	Returns:
		[type]: [description]
	"""
	t, T, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
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
def _shifted_fixed_odeint(f, x, t_span):
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



def _jagged_fixed_odeint(f, x,
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
