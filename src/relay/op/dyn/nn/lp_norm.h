/*!
 *
 * \file src/relay/op/dyn/nn/lp_norm.h
 * \brief implementation of the InferCorrectLayout pass for dynamic lp_norm
 */

#ifndef TVM_RELAY_OP_DYN_NN_LP_NORM_H_
#define TVM_RELAY_OP_DYN_NN_LP_NORM_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/tir/data_layout.h>

#include <unordered_set>

#include "../../op_common.h"

// namespaces used for dynamic op support
namespace tvm {
namespace relay {
namespace dyn {

InferCorrectLayoutOutput LpNormInferCorrectLayout(const Attrs& attrs,
                                                  const Array<Layout>& new_in_layouts,
                                                  const Array<Layout>& old_in_layouts,
                                                  const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<LpNormAttrs>();
  ICHECK(attrs_ptr);
  ObjectPtr<LpNormAttrs> param = make_object<LpNormAttrs>(*attrs_ptr);

  Array<IndexExpr> data_shape = old_in_types[0].as<TensorTypeNode>()->shape;

  // layout initialization
  Layout ret = Layout::Undef();

  // dim shape check
  std::vector<Integer> resolved_dims;
  if (!param->axis.empty()) {
    for (const Integer& val : param->axis) {
      int value = val->value;
      if (value < 0) {
        value = value + static_cast<int>(data_shape.size());
        resolved_dims.push_back(Integer(value));
      } else {
        resolved_dims.push_back(Integer(value));
      }
    }
  }

  param->axis = Array<Integer>(resolved_dims);

  // if new_in_layouts are defined, tries to modify the layout
  // considering split layout (such as NCHW16c)
  if (new_in_layouts.defined() && old_in_layouts.defined()) {
    std::unordered_set<std::string> reduced_dims;
    for (const Integer& axis : param->axis) {
      int axis_val = axis->value;
      // confirming if each dim is positive and in the range
      ICHECK_GE(axis_val, 0);
      ICHECK_LT(axis_val, static_cast<int>(data_shape.size()));

      reduced_dims.emplace(old_in_layouts[0][axis_val].name());
    }

    std::vector<Integer> new_dims;
    for (size_t axis_index = 0; axis_index < new_in_layouts[0].axes.size(); ++axis_index) {
      const auto& layout_axis = LayoutAxis::Get(new_in_layouts[0].axes[axis_index]);
      const std::string& layout_dim = layout_axis.name();

      if (layout_axis.IsPrimal()) {
        if (reduced_dims.count(layout_dim)) {
          new_dims.push_back(Integer(axis_index));
        }
      } else {
        auto primal_dim = layout_axis.ToPrimal().name();

        if (reduced_dims.count(primal_dim)) {
          new_dims.push_back(Integer(axis_index));
        }
      }
    }
    // reassigning dim with new dimension(s)
    param->axis = Array<Integer>(new_dims);
    ret = new_in_layouts[0];
  } else if (old_in_layouts.defined()) {
    ret = old_in_layouts[0];
  }
  // output layout: same as input if keepdims, else Undef
  Layout out_layout = param->keepdims ? ret : Layout::Undef();

  // 2 input layouts in return now since there are 2 inputs and all others as Attrs
  // which are mentioned in Attrs argument
  return InferCorrectLayoutOutput({ret, Layout::Undef()}, {out_layout}, Attrs(param));
}

}  // namespace dyn
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_DYN_NN_LP_NORM_H_
