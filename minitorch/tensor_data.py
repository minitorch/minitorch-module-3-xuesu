from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage
    """
    _ans = 0
    for i in range(len(strides)):
        _ans += index[i] * strides[i]
    return _ans


def to_index(ordinal: int, shape: Union[Shape, UserShape], out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    bs = 1.0
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = int((ordinal % (bs * shape[i])) // bs)
        bs *= shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None
    """
    ext_dim = len(big_shape) - len(shape)
    assert ext_dim >= 0
    for i in range(ext_dim, len(big_shape)):
        if big_shape[i] != shape[i - ext_dim]:
            # assert big_shape[i] > shape[i - ext_dim] and shape[i - ext_dim] == 1
            out_index[i - ext_dim] = 0
        else:
            out_index[i - ext_dim] = big_index[i]
    return


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    if len(shape1) > len(shape2):
        shape1, shape2 = shape2, shape1
    _ans = list(shape1)
    if len(shape1) < len(shape2):
        ext_dim = len(shape2) - len(shape1)
        _ans = [1] * ext_dim + _ans
    else:
        ext_dim = 0

    for i in range(len(shape2)):
        if _ans[i] != shape2[i]:
            if _ans[i] == 1:
                _ans[i] = shape2[i]
            elif shape2[i] > 1:
                raise IndexingError(f"Cannot broadcase {shape1} into {shape2}")
    return tuple(_ans)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple) or isinstance(index, list):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = [self.shape[o] for o in order]
        new_strides = [self.strides[o] for o in order]
        _ans = TensorData(self._storage, tuple(new_shape), tuple(new_strides))
        return _ans

    def to_string(self) -> str:
        _ans = ["shape: ", str(self.shape), ",", "strides", str(self.strides), "\n"]
        tmp_index = [0] * self.dims
        dim_no_need_tab = self.dims - 1
        for dim_i in range(self.dims - 1, -1, -1):
            if self.shape[dim_i] > 1:
                dim_no_need_tab = dim_i
                break

        def _get_prefix(_dim_j: int) -> str:
            if _dim_j == dim_no_need_tab:
                return "\t" * _dim_j + "["
            elif _dim_j < dim_no_need_tab:
                return "\t" * _dim_j + "[" + "\n"
            return "["

        def _get_suffix(_dim_j: int) -> str:
            if _dim_j == dim_no_need_tab:
                return "]" + "\n"
            elif _dim_j < dim_no_need_tab:
                return "\t" * _dim_j + "]" + "\n"
            return "]"

        for dim_i in range(self.dims):
            _ans.append(_get_prefix(dim_i))
        for _ in range(self.size):
            v = self.get(tmp_index)
            _ans.append(f"{v:3.4f},")
            tmp_index[-1] += 1
            ok_dim_i = self.dims
            for dim_i in range(self.dims - 1, -1, -1):
                if tmp_index[dim_i] >= self.shape[dim_i]:
                    ok_dim_i = dim_i
                    tmp_index[dim_i] = 0
                    tmp_index[dim_i - 1] += 1
                else:
                    break
            if ok_dim_i > 0:
                for dim_i in range(self.dims - 1, ok_dim_i - 1, -1):
                    _ans.append(_get_suffix(dim_i))
                for dim_i in range(ok_dim_i, self.dims):
                    _ans.append(_get_prefix(dim_i))
        for dim_i in range(self.dims - 1, -1, -1):
            _ans.append(_get_suffix(dim_i))
        return "".join(_ans)
