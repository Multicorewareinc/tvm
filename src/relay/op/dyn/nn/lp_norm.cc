/*!
 * \file lp_norm.cc
 * \brief lp_norm operator
 */

#include "lp_norm.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <utility>
#include <vector>

#include "../../op_common.h"

namespace tvm {
namespace relay {
namespace dyn {

// TVM_REGISTER_NODE_TYPE(LpNormAttrs);

bool LpNormRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  // input dtype check
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  if (!(data->dtype.is_float() || data->dtype.is_bfloat16())) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "Input dtype must be float or bfloat16");
    return false;
  }

  const LpNormAttrs* param = attrs.as<LpNormAttrs>();
  ICHECK(param != nullptr);

  // narrowing restrict check
  if (!param->dtype.is_void()) {
    DataType in_dtype = data->dtype;
    DataType out_dtype = param->dtype;
    if ((in_dtype.is_float() || in_dtype.is_bfloat16()) &&
        (out_dtype.is_float() || out_dtype.is_bfloat16())) {
      int in_bits = in_dtype.is_bfloat16() ? 16 : in_dtype.bits();
      int out_bits = out_dtype.is_bfloat16() ? 16 : out_dtype.bits();
      if (in_bits > out_bits) {
        reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                         << "Narrowing from " << in_dtype << " to " << out_dtype
                                         << " not supported.");
        return false;
      }
    }
  }

  // dim shape check
  std::vector<Integer> resolved_dims;
  if (!param->axis.empty()) {
    for (const Integer& val : param->axis) {
      int value = val->value;
      if (value < 0) {
        value = value + static_cast<int>(data->shape.size());
        if (value < 0 || value >= static_cast<int>(data->shape.size())) {
          reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                           << "Dimension out of range.");
          return false;
        }
        resolved_dims.push_back(Integer(value));
      } else {
        if (value >= static_cast<int>(data->shape.size())) {
          reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                           << "Dimension out of range.");
          return false;
        }
        resolved_dims.push_back(Integer(value));
      }
    }
  }

  // dim unique check
  std::unordered_set<int> unique_dims;
  for (const Integer& dim_idx : resolved_dims) {
    unique_dims.insert(dim_idx->value);
  }
  if (unique_dims.size() != resolved_dims.size()) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "Duplicate dimensions are not allowed.");
    return false;
  }

  // shape broadcast
  std::vector<IndexExpr> out_shape;
  if (resolved_dims.empty()) {
    if (param->keepdims) {
      for (size_t i = 0; i < data->shape.size(); ++i) {
        out_shape.push_back(1);
      }
    }
  } else {
    for (size_t i = 0; i < data->shape.size(); ++i) {
      bool is_reduce_dim = false;
      for (const Integer& d : resolved_dims) {
        if (static_cast<int>(i) == d->value) {
          is_reduce_dim = true;
          break;
        }
      }
      if (is_reduce_dim) {
        if (param->keepdims) {
          out_shape.push_back(1);
        }
      } else {
        out_shape.push_back(data->shape[i]);
      }
    }
  }
  DataType out_dtype = param->dtype.is_void() ? data->dtype : param->dtype;
  reporter->Assign(types[2], TensorType(out_shape, out_dtype));
  return true;
}

// changed ord to type Expr to take in dynamic values
Expr MakeDynLpNorm(Expr data, Expr ord, Array<Integer> axis, bool keepdims, DataType dtype) {
  auto attrs = make_object<LpNormAttrs>();
  attrs->axis = axis;
  attrs->keepdims = keepdims;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("dyn.nn.lp_norm");
  // added ord in inputs during return
  return Call(op, {data, ord}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.dyn.nn._make.lp_norm").set_body_typed(MakeDynLpNorm);

RELAY_REGISTER_OP("dyn.nn.lp_norm")
    .describe(R"code(Computes the vector L-p norm of the input tensor.

The function computes the norm of the input tensor. If `dim` is not specified, 
the input tensor is flattened before the norm is computed. If `dim` is an integer
or a tuple of integers, the norm is computed over those dimensions.

The `ord` attribute defines the type of vector norm computed.
Supported values for `ord`:

- `ord = 2` (default): 2-norm (Euclidean norm)
- `ord = inf`: Computes `max(abs(x))`
- `ord = -inf`: Computes `min(abs(x))`
- `ord = 0`: Computes `sum(x != 0)`
- `ord = other (int or float)`: Computes the p-norm according to the formula:

.. math::

    out = sum(abs(x)^{ord})^{(1 / ord)}

Example:

.. code-block:: python

    data = [[1.0, 2.0], [3.0, 4.0]]
    lp_norm(data, ord=2, dim=0) = [[3.162, 4.472]]
    lp_norm(data, ord=1, dim=1) = [[3.0], [7.0]]

)code" TVM_ADD_FILELINE)
    // changed number of inputs from 1 to 2 since ord is not an attr here
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input to which lp_norm will be applied.")
    .add_argument("ord", "Tensor", "The order of the norm.")
    .set_support_level(1)
    .set_attrs_type<LpNormAttrs>()
    .add_type_rel("DynLpNorm", LpNormRel)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", LpNormInferCorrectLayout);

}  // namespace dyn
}  // namespace relay
}  // namespace tvm