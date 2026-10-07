"""Lp Normalization Operator"""

import math

import tvm
from tvm import te, tir, topi

from tvm.te import Tensor
from tvm.tir import PrimExpr


# helper functions to avoid repeated computations to be mentioned
def _lp_norm(data, ord, axis, keepdims, compute_dtype):
    if ord==0:
        nonzero = topi.cast(
            topi.not_equal(data, tir.const(0, compute_dtype)),
            compute_dtype
        )
        result = topi.sum(nonzero, axis, keepdims)
    elif math.isinf(ord):
        if ord>0:
            result = topi.max(data, axis, keepdims)
        else:
            result = topi.min(data, axis, keepdims)
    else:
        result = _lp_norm_p(data, ord, axis, keepdims, compute_dtype)

    return result


def _lp_norm_p(data, ord, axis, keepdims, compute_dtype):
    if isinstance(ord, (int, float)):
        ord_const = tir.const(ord, compute_dtype)
        inv_ord_const = tir.const(1.0 / ord, compute_dtype)
        powered = topi.power(data, ord_const)
        summed = topi.sum(powered, axis, keepdims=keepdims)
        result = topi.power(summed, inv_ord_const)
    else:
        inv_ord = tir.const(1.0, compute_dtype) / ord
        powered = topi.power(data, ord)
        summed = topi.sum(powered, axis, keepdims=keepdims)
        result = topi.power(summed, inv_ord)

    return result


def lp_norm(data, ord, axis, keepdims, dtype):
    """
    Computes the vector Lp norm of a tensor.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input tensor.

    ord : int or float, optional
        Order of the norm.

    dim : int or tuple/list of ints, optional
        Dimension(s) over which to compute the norm.
        If None, the input is treated as a flattened vector.

    keepdims : bool
        Whether to retain reduced dimensions.

    dtype : str, optional
        Accumulation/output dtype.

    Returns
    -------
    tvm.te.Tensor
        Vector norm.

    math
    -------
    out = sum(abs(x)^(ord))^(1 / ord)
    """

    # here, axis and keepdims must be static-defined (compile-time) parameters
    ndim = len(data.shape)

    # added check for dynamic ord values also
    if not isinstance(ord, (int, float, Tensor, PrimExpr)):
        raise TypeError("ord must be an int or float.")

    # TVM doesn't support complex numbers at low level
    
    base_dtype = data.dtype
    target_dtype = dtype if dtype not in (None, "") else base_dtype

    # float32 computation if output data type is bf16, since bf16 is not supported by some ops in TIR level
    compute_dtype = "float32" if target_dtype == "bfloat16" else target_dtype
    data_cast = topi.cast(data, compute_dtype) if data.dtype != compute_dtype else data

    # abs(x) only supports int, uint and float
    
    data_cast = topi.abs(data_cast)

    # scalar input handling in vector_norm
    if ndim == 0:
        # Static ord
        if isinstance(ord, (int, float, tvm.tir.IntImm, tvm.tir.FloatImm)):
            ord_val = ord.value if hasattr(ord, "value") else ord

            if ord_val == 0:
                nonzero = topi.cast(
                    topi.not_equal(data_cast, tir.const(0, compute_dtype)),
                    compute_dtype
                )
                return nonzero
            return data_cast

        # Dynamic ord
        nonzero = topi.cast(
            topi.not_equal(data_cast, tir.const(0, compute_dtype)),
            compute_dtype
        )

        return topi.where(ord == 0, nonzero, data_cast)

    # assigning all dim values to axes if dim is None or empty tuple/list
    if axis is None or (isinstance(axis, (list, tuple, tvm.ir.Array)) and len(axis) == 0):
        axes = list(range(ndim))

    # else, wrapping given axes in a list
    else:
        if isinstance(axis, int):
            axes = [axis]
        elif isinstance(axis, (tuple, list, tvm.ir.Array)):
            axes = list(axis)
        else:
            raise TypeError(
                "axis must be None, an int, or a tuple/list of ints."
            )

        # normalizing axis
        normalized_axes = []

        for ax in axes:
            if isinstance(ax, tvm.tir.IntImm):
                ax = ax.value
            elif not isinstance(ax, int):
                try:
                    ax = int(ax)
                except ValueError:
                    raise TypeError("Each dimension must be an integer.")

            if ax < 0:
                ax += ndim

            if ax < 0 or ax >= ndim:
                raise ValueError(
                    f"Dimension {ax} out of range for tensor "
                    f"with {ndim} dimensions."
                )

            normalized_axes.append(ax)

        # check for duplicates
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("Duplicate dimensions are not allowed.")

        axes = normalized_axes

    #if ord is static
    if isinstance(ord, (int, float, tvm.tir.IntImm, tvm.tir.FloatImm)):
        ord_val = ord.value if hasattr(ord, "value") else ord
        # ord check
        result = _lp_norm(data_cast, ord_val, axes, keepdims, compute_dtype)

    # if ord value is dynamic (not zero, inf or -inf), this condition is reached
    else:
        result = _lp_norm_p(data_cast, ord, axes, keepdims, compute_dtype)

    # typecasting back to original data type 
    if target_dtype == "bfloat16":
        result = topi.cast(result, "bfloat16")
       
    return result
