"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations
from numbers import Number

import random
from typing import TYPE_CHECKING, Union

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from .tensor import Tensor
    from typing import Any, List, Tuple
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


def _make_tensor_constant(
    c: Union[int, float], shape: UserShape, backend: TensorBackend
) -> Tensor:
    sz = 1
    for shapei in shape:
        sz *= shapei
    return minitorch.Tensor.make([c * 1.0] * sz, shape=shape, backend=backend)


def _make_sure_gradient_shape(gradient: Tensor, x_shape: UserShape) -> Tensor:
    """
    Force gradient that generated by broadcasted operation to recover its shape.
    We are doing this since minitorch cannot distinguish the shape of storage with bloated shape
    """
    if gradient.size == 1:
        return _make_tensor_constant(
            gradient._tensor._storage[0], x_shape, gradient.backend
        )
    assert gradient.size >= np.prod(x_shape)
    if gradient.shape == x_shape:
        return gradient
    else:  # gradient.shape: [1,1,3,5,5], x_shape: [3,1,5], y.shape[1,1,3,5,5]
        ext_dim = len(gradient.shape) - len(x_shape)
        assert ext_dim >= 0
        for i in range(ext_dim):
            # print("gradient", gradient.shape, gradient.strides, i)
            gradient = gradient.f.add_reduce(gradient, i)
        for i in range(ext_dim, len(gradient.shape)):
            if gradient.shape[i] != x_shape[i - ext_dim]:
                assert (
                    gradient.shape[i] > x_shape[i - ext_dim]
                    and x_shape[i - ext_dim] == 1
                )
                # a is broadcasted
                # print("gradient", gradient.shape, gradient.strides, i)
                gradient = gradient.f.add_reduce(gradient, i)
        if ext_dim > 0:
            return minitorch.Tensor.make(
                gradient._tensor._storage,
                gradient._tensor.shape[ext_dim:],
                gradient._tensor.strides[ext_dim:],
                backend=gradient.backend,
            ).contiguous()
    return gradient.contiguous()


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        # print("a.unique_id", a.unique_id, b.unique_id)
        return _make_sure_gradient_shape(
            a.f.mul_zip(grad_output, b), a.shape
        ), _make_sure_gradient_shape(a.f.mul_zip(grad_output, a), b.shape)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        s = t1.f.sigmoid_map(t1)
        _ans: Tensor = t1.f.mul_zip(
            t1.f.mul_zip(
                s,
                t1.f.add_zip(
                    _make_tensor_constant(1.0, t1.shape, t1.backend),
                    t1.f.neg_map(s),
                ),
            ),
            grad_output,
        )
        return _ans


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return t1.f.relu_back_zip(t1, grad_output)  # type: ignore


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return t1.f.log_back_zip(t1, grad_output)  # type: ignore


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return t1.f.mul_zip(t1.f.exp_map(t1), grad_output)  # type: ignore


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return _make_tensor_constant(
            0.0, a_shape, grad_output.backend
        ), _make_tensor_constant(0.0, b_shape, grad_output.backend)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return _make_tensor_constant(
            0.0, a_shape, grad_output.backend
        ), _make_tensor_constant(0.0, b_shape, grad_output.backend)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        order_list: List[int] = order.to_numpy().astype(np.int32).tolist()
        ctx.save_for_backward(order_list)
        need_permute = not all(i == o for i, o in enumerate(order_list))
        if need_permute:
            _ans = a._new(a._tensor.permute(*order_list))
            return _ans
        else:
            return a

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (order_list,) = ctx.saved_values
        after_ind_org_ind_prs = [(o, i) for i, o in enumerate(order_list)]
        after_ind_org_ind_prs.sort()
        back_order = [pr[1] for pr in after_ind_org_ind_prs]
        need_permute = not all(i == o for i, o in enumerate(back_order))
        if need_permute:
            if grad_output.size == 1:
                return grad_output, 0.0
            _ans = grad_output._new(grad_output._tensor.permute(*back_order))
            return _ans, 0.0
        else:
            return grad_output, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        # print("backward Matmul", t1, t2, grad_output)

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        # check_arr = [0.0] * x.size
        # for tmp_index in x._tensor.indices():
        #     from minitorch.tensor_data import index_to_position

        #     pos = index_to_position(tmp_index, x.strides)
        #     check_arr[pos] = grad_central_difference(f, *vals, arg=i, ind=tmp_index)
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        # print("x.grad: ", x.grad, "check:", check_arr)
        # print(
        #     "x.grad: ", x.grad, "ind:", ind, "x.grad[ind]", x.grad[ind], "check:", check
        # )
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
