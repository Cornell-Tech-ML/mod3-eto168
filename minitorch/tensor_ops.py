from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]: ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns:
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

        #     # out index is a 0 array of size out_shape
        #     out_index = np.zeros(len(out_shape), dtype=int)
        #     in_index = np.zeros(len(in_shape), dtype=int)

        #     # here, we test that out is equalt to the size of
        #     # the out tensor.
        #     assert len(out) == int(np.prod(out_shape)), (
        #         "len(out_shape) is "
        #         + str(len(out_shape))
        #         + "and np.prod(out_shape) is "
        #         + str(np.prod(out_shape))
        #     )

        #     # broadcasted case: if we assume out_shape and in_shape broadcast,
        #     # then we can get every element in out_shape from the in_shape.
        #     # to do this, we start with the in_shape. Then for each
        #     # element in `out`, which is the 1D array of out_shape,
        #     # we get the corresponding index in `in_storage`, the 1D array
        #     # of in_shape. Then we apply the function to the element at the
        #     # # index in the in_storage and save it to the out.
        #     # for each element in the 1D array of out_shape (i.e. = size)
        #     for i in range(len(out)):
        #         # get the index in the out_shape
        #         to_index(i, out_shape, out_index)
        #         # get the index in the in_shape (which is SMALLER)
        #         broadcast_index(out_index, out_shape, in_shape, in_index)
        #         # get the position in the in_storage
        #         in_position = index_to_position(in_index, in_strides)
        #         # get the position in the out storage. i.e. where to store
        #         out_position = index_to_position(out_index, out_strides)
        #         # apply the function to the in_storage and save it to the out
        #         out[out_position] = fn(in_storage[in_position])

        # return _map
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        in_index: Index = np.zeros(MAX_DIMS, np.int16)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply

    Returns:
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

        #     # out index is a 0 array of size out_shape
        #     out_index = np.zeros(len(out_shape), dtype=int)
        #     a_index = np.zeros(len(a_shape), dtype=int)
        #     b_index = np.zeros(len(b_shape), dtype=int)

        #     # for each element in out (1D array of out_shape)
        #     for i in range(len(out)):
        #         # get the index in the out_shape
        #         to_index(i, out_shape, out_index)
        #         # get the index in the a_shape (which is SMALLER)
        #         broadcast_index(out_index, out_shape, a_shape, a_index)
        #         # get the index in the b_shape (which is SMALLER)
        #         broadcast_index(out_index, out_shape, b_shape, b_index)
        #         # get the position in the a_storage
        #         a_position = index_to_position(a_index, a_strides)
        #         # get the position in the b_storage
        #         b_position = index_to_position(b_index, b_strides)

        #         out_position = index_to_position(out_index, out_strides)
        #         # apply the function to the a_storage and b_storage and save it to
        #         out[out_position] = fn(a_storage[a_position], b_storage[b_position])

        # return _zip
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        a_index: Index = np.zeros(MAX_DIMS, np.int32)
        b_index: Index = np.zeros(MAX_DIMS, np.int32)

        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # raise NotImplementedError("Need to implement for Task 2.3")

        #     out_index = np.zeros(len(out_shape), dtype=int)
        #     a_index = np.zeros(len(a_shape), dtype=int)

        #     # reduce is a bit trickier. out_shape will have the same number of
        #     # dimensions, but the size of the dimension == reduce_dim will be 1.
        #     for i in range(len(out)):
        #         # get the index in the out_shape
        #         to_index(i, out_shape, out_index)
        #         # translate the out index to the index in a_shape
        #         broadcast_index(out_index, out_shape, a_shape, a_index)
        #         a_position = index_to_position(a_index, a_strides)
        #         out_position = index_to_position(out_index, out_strides)

        #         # for each element in the reduce_dim, apply the function,
        #         # and accumulate the value.
        #         for j in range(a_shape[reduce_dim]):
        #             # get the index in the a_shape
        #             a_index[reduce_dim] = j
        #             a_position = index_to_position(a_index, a_strides)
        #             out[out_position] = fn(out[out_position], a_storage[a_position])

        # TODO: NEED TO FIX OPTIMIZATIONS
        # return _reduce
        out_index: Index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for j in range(reduce_size):
                out_index[reduce_dim] = j
                a = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[a])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
