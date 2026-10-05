#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

class CastFoldingMutator : public MixedModeMutator {
 private:
  const Op& cast_op_;

  // Helper to extract the target dtype and the inner argument of a Cast node
  bool IsCast(const Expr& expr, DataType* out_dtype, Expr* out_arg) const {
    const auto* call = expr.as<CallNode>();
    if (call == nullptr || !call->op.same_as(cast_op_) || call->args.size() != 1) {
      return false;
    }

    const auto* attrs = call->attrs.as<CastAttrs>();
    if (attrs == nullptr) {
      return false;
    }

    *out_dtype = attrs->dtype;
    *out_arg = call->args[0];
    return true;
  }

  bool IsFloatFamily(const DataType& t) const { return t.is_float() || t.is_bfloat16(); }

  bool IsIntFamily(const DataType& t) const { return t.is_int(); }

  bool IsUintFamily(const DataType& t) const { return t.is_uint(); }

  // Safely extract DataType from a TensorType, ignoring Tuples/Void
  DataType GetTensorDType(const Type& type) const {
    if (const auto* tensor_type = type.as<TensorTypeNode>()) {
      return tensor_type->dtype;
    }
    return DataType::Void();
  }

  // Strictly determines if `original -> intermediate -> target` is mathematically exact.
  bool IsLosslessRoundTrip(DataType original, DataType intermediate, DataType target) const {
    if (original != target) {
      return false;
    }

    if (original == intermediate) {
      return true;
    }

    bool same_family = (IsIntFamily(original) && IsIntFamily(intermediate)) ||
                       (IsFloatFamily(original) && IsFloatFamily(intermediate)) ||
                       (IsUintFamily(original) && IsUintFamily(intermediate));

    if (same_family && intermediate.bits() >= original.bits()) {
      return true;
    }

    if (original.is_uint() && intermediate.is_int() && intermediate.bits() > original.bits()) {
      return true;
    }

    return false;
  }

  // Determines if `original -> intermediate -> target` can be compressed to `original -> target`
  bool CanCompressTwoCasts(DataType original, DataType intermediate, DataType target) const {
    // Float family: e.g. fp16 -> fp64 -> fp32 becomes fp16 -> fp32
    if (IsFloatFamily(original) && IsFloatFamily(intermediate) && IsFloatFamily(target)) {
      if (intermediate.bits() >= original.bits() && intermediate.bits() >= target.bits()) {
        return true;
      }
    }

    // Signed Int family: e.g. int8 -> int32 -> int16 becomes int8 -> int16
    if (IsIntFamily(original) && IsIntFamily(intermediate) && IsIntFamily(target)) {
      if (intermediate.bits() >= original.bits() && intermediate.bits() >= target.bits()) {
        return true;
      }
    }

    // Unsigned Int family: e.g. uint8 -> uint32 -> uint16 becomes uint8 -> uint16
    if (IsUintFamily(original) && IsUintFamily(intermediate) && IsUintFamily(target)) {
      if (intermediate.bits() >= original.bits() && intermediate.bits() >= target.bits()) {
        return true;
      }
    }

    return false;
  }

 public:
  CastFoldingMutator() : cast_op_(Op::Get("cast")) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    const auto* post_call = post.as<CallNode>();
    // We only mutate Cast operators.
    if (post_call == nullptr || !post_call->op.same_as(cast_op_)) {
      return post;
    }

    const auto* attrs = post_call->attrs.as<CastAttrs>();
    if (attrs == nullptr) {
      return post;
    }

    DataType target_dtype = attrs->dtype;
    Expr arg = post_call->args[0];

    Type arg_type = arg->checked_type_;
    if (!arg_type.defined()) {
      arg_type = transform::InferTypeLocal(arg);
    }
    DataType source_dtype = GetTensorDType(arg_type);

    // Identity Cast Removal : cast(x, dtype(x)) -> x
    if (source_dtype == target_dtype) {
      return arg;
    }

    // Lossless Cast-Chain Folding
    // E.g., FP16 -> FP32 -> FP16 (folds)
    // E.g., FP32 -> FP16 -> FP32 (DOES NOT fold)
    DataType intermediate_dtype;
    Expr inner_arg;

    if (IsCast(arg, &intermediate_dtype, &inner_arg)) {
      Type inner_arg_type = inner_arg->checked_type_;
      if (!inner_arg_type.defined()) {
        inner_arg_type = transform::InferTypeLocal(inner_arg);
      }
      DataType original_dtype = GetTensorDType(inner_arg_type);

      // 2-Cast Round-Trip Elimination (2 casts -> 0 casts)
      // e.g., fp16 -> fp32 -> fp16 ==> x
      // e.g., int8 -> int32 -> int8 ==> x
      if (IsLosslessRoundTrip(original_dtype, intermediate_dtype, target_dtype)) {
        return inner_arg;
      }

      // 2-Cast Transitive Compression (2 casts -> 1 cast)
      // e.g., int8 -> int32 -> int16 ==> cast(x, "int16")
      // e.g., fp16 -> fp64 -> fp32 ==> cast(x, "fp32")
      if (CanCompressTwoCasts(original_dtype, intermediate_dtype, target_dtype)) {
        return Call(cast_op_, {inner_arg}, post_call->attrs, {});
      }
    }

    return post;
  }
};

Expr FoldMixedPrecisionCasts(const Expr& expr) { return CastFoldingMutator().Mutate(expr); }

namespace transform {
Pass FoldMixedPrecisionCastsPass() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldMixedPrecisionCasts(f));
      };

  return CreateFunctionPass(pass_func, 0, "FoldMixedPrecisionCasts", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldMixedPrecisionCasts")
    .set_body_typed(FoldMixedPrecisionCastsPass);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
