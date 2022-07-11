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

from inspect import getfullargspec

import torch, abc
from torchdyn.core.defunc import DEFuncBase, DEFunc
import torch.nn as nn
import numpy as np
import math

def standardize_vf_call_signature(vector_field, order=1, defunc_wrap=False):
    "Ensures Callables or nn.Modules passed to `ODEProblems` and `NeuralODE` have consistent `__call__` signature (t, x)"

    if issubclass(type(vector_field), nn.Module):
        if 't' not in getfullargspec(vector_field.forward).args:
            print("Your vector field callable (nn.Module) should have both time `t` and state `x` as arguments, "
                "we've wrapped it for you.")
            vector_field = DEFuncBase(vector_field, has_time_arg=False)
    else:
        # argspec for lambda functions needs to be done on the function itself
        if 't' not in getfullargspec(vector_field).args:
            print("Your vector field callable (lambda) should have both time `t` and state `x` as arguments, "
                "we've wrapped it for you.")
            vector_field = DEFuncBase(vector_field, has_time_arg=False)
        else: vector_field = DEFuncBase(vector_field, has_time_arg=True)
    if defunc_wrap: return DEFunc(vector_field, order)
    else: return vector_field


class InterpolationBase(torch.nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def grid_points(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def interval(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, t):
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, t):
        raise NotImplementedError


def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)


def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)


def validate_input_path(x, t):
    if not x.is_floating_point():
        raise ValueError("X must both be floating point.")

    if x.ndimension() < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels. It instead has "
                         "shape {}.".format(tuple(x.shape)))

    if t is None:
        t = torch.linspace(0, x.size(-2) - 1, x.size(-2), dtype=x.dtype, device=x.device)

    if not t.is_floating_point():
        raise ValueError("t must both be floating point.")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional. It instead has shape {}.".format(tuple(t.shape)))
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")
        prev_t_i = t_i

    if x.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t. X has shape {} and t has shape {}, "
                         "corresponding to time dimensions of {} and {} respectively."
                         .format(tuple(x.shape), tuple(t.shape), x.size(-2), t.size(0)))

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2. It instead has shape {}, corresponding to a "
                         "time dimension of size {}.".format(tuple(t.shape), t.size(0)))

    return t


def forward_fill(x, fill_index=-2):
    """Forward fills data in a torch tensor of shape (..., length, input_channels) along the length dim.

    Arguments:
        x: tensor of values with first channel index being time, of shape (..., length, input_channels), where ... is
            some number of batch dimensions.
        fill_index: int that denotes the index to fill down. Default is -2 as we tend to use the convention (...,
            length, input_channels) filling down the length dimension.

    Returns:
        A tensor with forward filled data.
    """
    # Checks
    assert isinstance(x, torch.Tensor)
    assert x.dim() >= 2

    mask = torch.isnan(x)
    if mask.any():
        cumsum_mask = (~mask).cumsum(dim=fill_index)
        cumsum_mask[mask] = 0
        _, index = cumsum_mask.cummax(dim=fill_index)
        x = x.gather(dim=fill_index, index=index)

    return x


class TupleControl(torch.nn.Module):
    def __init__(self, *controls):
        super(TupleControl, self).__init__()

        if len(controls) == 0:
            raise ValueError("Expected one or more controls to batch together.")

        self._interval = controls[0].interval
        grid_points = controls[0].grid_points
        same_grid_points = True
        for control in controls[1:]:
            if (control.interval != self._interval).any():
                raise ValueError("Can only batch togehter controls over the same interval.")
            if same_grid_points and (control.grid_points != grid_points).any():
                same_grid_points = False

        if same_grid_points:
            self._grid_points = grid_points
        else:
            self._grid_points = None

        self.controls = torch.nn.ModuleList(controls)

    @property
    def interval(self):
        return self._interval

    @property
    def grid_points(self):
        if self._grid_points is None:
            raise RuntimeError("Batch of controls have different grid points.")
        return self._grid_points

    def evaluate(self, t):
        return tuple(control.evaluate(t) for control in self.controls)

    def derivative(self, t):
        return tuple(control.derivative(t) for control in self.controls)


def _natural_cubic_spline_coeffs_without_missing_values(t, x):
    # x should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = x.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = x[..., :1]
        b = (x[..., 1:] - x[..., :1]) / (t[..., 1:] - t[..., :1])
        two_c = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
        three_d = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)
    else:
        # Set up some intermediate values
        time_diffs = t[1:] - t[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (x[..., 1:] - x[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=x.dtype, device=x.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal,
                                                  time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        a = x[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d


def _natural_cubic_spline_coeffs_with_missing_values(t, x, _version):
    if x.ndimension() == 1:
        # We have to break everything down to individual scalar paths because of the possibility of missing values
        # being different in different channels
        return _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x, _version)
    else:
        a_pieces = []
        b_pieces = []
        two_c_pieces = []
        three_d_pieces = []
        for p in x.unbind(dim=0):  # TODO: parallelise over this
            a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, p, _version)
            a_pieces.append(a)
            b_pieces.append(b)
            two_c_pieces.append(two_c)
            three_d_pieces.append(three_d)
        return (cheap_stack(a_pieces, dim=0),
                cheap_stack(b_pieces, dim=0),
                cheap_stack(two_c_pieces, dim=0),
                cheap_stack(three_d_pieces, dim=0))


def _natural_cubic_spline_coeffs_with_missing_values_scalar(t, x, _version):
    # t and x both have shape (length,)

    nan = torch.isnan(x)
    not_nan = ~nan
    path_no_nan = x.masked_select(not_nan)

    if path_no_nan.size(0) == 0:
        # Every entry is a NaN, so we take a constant path with derivative zero, so return zero coefficients.
        # Note that we may assume that X.size(0) >= 2 by the checks in __init__ so "X.size(0) - 1" is a valid
        # thing to do.
        return (torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device),
                torch.zeros(x.size(0) - 1, dtype=x.dtype, device=x.device))
    # else we have at least one non-NaN entry, in which case we're going to impute at least one more entry (as
    # the path is of length at least 2 so the start and the end aren't the same), so we will then have at least two
    # non-Nan entries. In particular we can call _compute_coeffs safely later.

    # How to deal with missing values at the start or end of the time series? We're creating some splines, so one
    # option is just to extend the first piece backwards, and the final piece forwards. But polynomials tend to
    # behave badly when extended beyond the interval they were constructed on, so the results can easily end up
    # being awful.
    if _version == 0:
        # Instead we impute an observation at the very start equal to the first actual observation made, and impute an
        # observation at the very end equal to the last actual observation made, and then proceed with splines as
        # normal.
        need_new_not_nan = False
        if torch.isnan(x[0]):
            if not need_new_not_nan:
                x = x.clone()
                need_new_not_nan = True
            x[0] = path_no_nan[0]
        if torch.isnan(x[-1]):
            if not need_new_not_nan:
                x = x.clone()
                need_new_not_nan = True
            x[-1] = path_no_nan[-1]
        if need_new_not_nan:
            not_nan = ~torch.isnan(x)
            path_no_nan = x.masked_select(not_nan)
    else:
        # Instead we fill forward and backward from the first/last observation made. This is better than the previous
        # approach as the splines instead rapidly stabilise to the first/last value.
        cumsum_mask = not_nan.cumsum(dim=0)
        cumsum_mask[nan] = -1
        last_non_nan_index = cumsum_mask.argmax(dim=0)
        cumsum_mask[nan] = 1 + last_non_nan_index
        first_non_nan_index = cumsum_mask.argmin(dim=0)
        x = x.clone()
        x[:first_non_nan_index] = x[first_non_nan_index]
        x[last_non_nan_index + 1:] = x[last_non_nan_index]
        not_nan = ~torch.isnan(x)
        path_no_nan = x.masked_select(not_nan)
    times_no_nan = t.masked_select(not_nan)

    # Find the coefficients on the pieces we do understand
    # These all have shape (len - 1,)
    (a_pieces_no_nan,
     b_pieces_no_nan,
     two_c_pieces_no_nan,
     three_d_pieces_no_nan) = _natural_cubic_spline_coeffs_without_missing_values(times_no_nan, path_no_nan)

    # Now we're going to normalise them to give coefficients on every interval
    a_pieces = []
    b_pieces = []
    two_c_pieces = []
    three_d_pieces = []

    iter_times_no_nan = iter(times_no_nan)
    iter_coeffs_no_nan = iter(zip(a_pieces_no_nan, b_pieces_no_nan, two_c_pieces_no_nan, three_d_pieces_no_nan))
    next_time_no_nan = next(iter_times_no_nan)
    for time in t[:-1]:
        # will always trigger on the first iteration because of how we've imputed missing values at the start and
        # end of the time series.
        if time >= next_time_no_nan:
            prev_time_no_nan = next_time_no_nan
            next_time_no_nan = next(iter_times_no_nan)
            next_a_no_nan, next_b_no_nan, next_two_c_no_nan, next_three_d_no_nan = next(iter_coeffs_no_nan)
        offset = prev_time_no_nan - time
        a_inner = (0.5 * next_two_c_no_nan - next_three_d_no_nan * offset / 3) * offset
        a_pieces.append(next_a_no_nan + (a_inner - next_b_no_nan) * offset)
        b_pieces.append(next_b_no_nan + (next_three_d_no_nan * offset - next_two_c_no_nan) * offset)
        two_c_pieces.append(next_two_c_no_nan - 2 * next_three_d_no_nan * offset)
        three_d_pieces.append(next_three_d_no_nan)

    return (cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0))


# The mathematics of this are adapted from  http://mathworld.wolfram.com/CubicSpline.html, although they only treat the
# case of each piece being parameterised by [0, 1]. (We instead take the length of each piece to be the difference in
# time stamps.)
def _natural_cubic_spline_coeffs(x, t, _version):
    t = validate_input_path(x, t)

    if torch.isnan(x).any():
        # Transpose because channels are a batch dimension for the purpose of finding interpolating polynomials.
        # b, two_c, three_d have shape (..., channels, length - 1)
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, x.transpose(-1, -2), _version)
    else:
        # Can do things more quickly in this case.
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, x.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    two_c = two_c.transpose(-1, -2)
    three_d = three_d.transpose(-1, -2)
    coeffs = torch.cat([a, b, two_c, three_d], dim=-1)  # for simplicity put them all together
    return coeffs


def natural_cubic_spline_coeffs(x, t=None):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    ********************
    DEPRECATED: this now exists for backward compatibility. For new projects please use `natural_cubic_coeffs` instead,
    which handles missing data at the start/end of a time series better.
    ********************

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]). If you are using neural CDEs then you **do not need to use this
            argument**. See the Further Documentation in README.md.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tensor, which should in turn be passed to `torchcde.CubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `torchcde.CubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your data.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your dataset as far as PyTorch's `torch.utils.data.Dataset` and
            `torch.utils.data.DataLoader` classes are concerned.
        (e) Call CubicSpline as the first part of your model.

        See also the accompanying example.py.
    """
    return _natural_cubic_spline_coeffs(x, t, _version=0)


def natural_cubic_coeffs(x, t=None):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    Arguments:
        x: tensor of values, of shape (..., length, input_channels), where ... is some number of batch dimensions. This
            is interpreted as a (batch of) paths taking values in an input_channels-dimensional real vector space, with
            length-many observations. Missing values are supported, and should be represented as NaNs.
        t: Optional one dimensional tensor of times. Must be monotonically increasing. If not passed will default to
            tensor([0., 1., ..., length - 1]). If you are using neural CDEs then you **do not need to use this
            argument**. See the Further Documentation in README.md.

    Warning:
        If there are missing values then calling this function can be pretty slow. Make sure to cache the result, and
        don't reinstantiate it on every forward pass, if at all possible.

    Returns:
        A tensor, which should in turn be passed to `torchcde.CubicSpline`.

        Why do we do it like this? Because typically you want to use PyTorch tensors at various interfaces, for example
        when loading a batch from a DataLoader. If we wrapped all of this up into just the
        `torchcde.CubicSpline` class then that sort of thing wouldn't be possible.

        As such the suggested use is to:
        (a) Load your data.
        (b) Preprocess it with this function.
        (c) Save the result.
        (d) Treat the result as your dataset as far as PyTorch's `torch.utils.data.Dataset` and
            `torch.utils.data.DataLoader` classes are concerned.
        (e) Call CubicSpline as the first part of your model.

        See also the accompanying example.py.
    """
    return _natural_cubic_spline_coeffs(x, t, _version=1)


class CubicSpline(InterpolationBase):
    """Calculates the cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        x = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_coeffs(x)
        # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
        spline = CubicSpline(coeffs)
        point = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, t=None, **kwargs):
        """
        Arguments:
            coeffs: As returned by `torchcde.natural_cubic_coeffs`.
            t: As passed to linear_interpolation_coeffs. (If it was passed. If you are using neural CDEs then you **do
                not need to use this argument**. See the Further Documentation in README.md.)
        """
        super(CubicSpline, self).__init__(**kwargs)

        if t is None:
            t = torch.linspace(0, coeffs.size(-2), coeffs.size(-2) + 1, dtype=coeffs.dtype, device=coeffs.device)

        channels = coeffs.size(-1) // 4
        if channels * 4 != coeffs.size(-1):  # check that it's a multiple of 4
            raise ValueError("Passed invalid coeffs.")
        a, b, two_c, three_d = (coeffs[..., :channels], coeffs[..., channels:2 * channels],
                                coeffs[..., 2 * channels:3 * channels], coeffs[..., 3 * channels:])

        self.register_buffer('_t', t)
        self.register_buffer('_a', a)
        self.register_buffer('_b', b)
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self.register_buffer('_two_c', two_c)
        self.register_buffer('_three_d', three_d)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._b.dtype, device=self._b.device)
        maxlen = self._b.size(-2) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv


class NaturalCubicSpline(CubicSpline):
    """Calculates the coefficients of the natural cubic spline approximation to the batch of controls given.

    ********************
    DEPRECATED: this now exists for backward compatibility. For new projects please use `CubicSpline` instead. This
    class is general for any cubic coeffs (currently natural cubic or Hermite with backwards differences).
    ********************
    """

