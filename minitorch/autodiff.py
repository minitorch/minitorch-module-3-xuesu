from dataclasses import dataclass
from typing import Any, Iterable, Tuple, runtime_checkable

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    reduced_vals = list(vals)
    added_vals = list(vals)
    reduced_vals[arg] -= epsilon
    added_vals[arg] += epsilon
    return (f(*added_vals) - f(*reduced_vals)) / (2.0 * epsilon)


variable_count = 1


@runtime_checkable
class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    vis = set()
    que = [variable]
    id2cnt: dict[int, int] = dict()
    id2v = dict()
    while len(que) > 0:
        top_v = que[0]
        que = que[1:]
        id2v[top_v.unique_id] = top_v
        if not top_v.is_constant():
            if not top_v.is_constant() and not top_v.is_leaf():
                for parent_v in top_v.parents:
                    if not isinstance(parent_v, Variable):
                        continue
                    id2cnt[parent_v.unique_id] = id2cnt.get(parent_v.unique_id, 0) + 1
                    if parent_v.unique_id not in vis:
                        vis.add(parent_v.unique_id)
                        que.append(parent_v)
    _ans = []
    que = [v for i, v in id2v.items() if id2cnt.get(i, 0) == 0]
    assert variable in que
    vis = set([v.unique_id for v in que])
    while len(que) > 0:
        top_v = que[0]
        que = que[1:]
        if not top_v.is_constant():
            _ans.append(top_v)
            if not top_v.is_constant() and not top_v.is_leaf():
                for parent_v in top_v.parents:
                    if not isinstance(parent_v, Variable):
                        continue
                    org_cnt = id2cnt.get(parent_v.unique_id, 0)
                    assert org_cnt > 0
                    id2cnt[parent_v.unique_id] = org_cnt - 1
                    if org_cnt == 1 and parent_v.unique_id not in vis:
                        vis.add(parent_v.unique_id)
                        que.append(parent_v)
    assert len(_ans) == len([v for v in id2v.values() if not v.is_constant()])
    return _ans


def print_out_graph(varaible: Variable) -> None:
    que = topological_sort(varaible)
    for v in que:
        res = []
        res.append(str(v.unique_id))
        if not v.is_constant() and not v.is_leaf():
            res += ["<-", v.history.last_fn.__name__, "<-"]  # type: ignore
            for p_v in v.parents:
                if isinstance(p_v, Variable):
                    res += [
                        str(p_v.unique_id),
                        "is_constant: " + repr(p_v) if p_v.is_constant() else "",
                    ]
                else:
                    res.append(p_v)
        print(*res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    que = topological_sort(variable)
    # print_out_graph(variable)

    id2d = {variable.unique_id: deriv}
    for v in que:
        d = id2d[v.unique_id]
        # print("topv", v, d, v.history)
        from minitorch.tensor import Tensor

        assert (
            isinstance(d, Tensor)
            and isinstance(v, Tensor)
            and (d.size == 1 or v.shape == d.shape)
        )
        if v.is_leaf():
            v.accumulate_derivative(d)
        else:
            for p_v, p_d in v.chain_rule(d):
                if not isinstance(p_v, Tensor) or p_v.is_constant():
                    continue
                # print("p_v->v", p_v.unique_id, "->", v.unique_id)
                assert p_v.shape == p_d.shape or p_d.size == 1
                if p_v.unique_id not in id2d:
                    id2d[p_v.unique_id] = p_d
                else:
                    id2d[p_v.unique_id] += p_d
    assert len(list(que)) == len(id2d)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

    def __repr__(self) -> str:
        v_repr = [
            repr(v) if not isinstance(v, Variable) else str(v.unique_id)
            for v in self.saved_values
        ]
        return f"Context(no_grad={self.no_grad}, saved_values=(Var:{','.join(v_repr)}))"
