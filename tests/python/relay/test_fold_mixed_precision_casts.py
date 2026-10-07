# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Unit tests for testing FoldMixedPrecisionCasts pass"""
import tvm
from tvm import relay
from tvm.relay import transform
import pytest

FOLD = transform.FoldMixedPrecisionCasts()

def run_opt_pass(expr, opt_pass):
    """Helper to wrap an expression in a module, run type inference, apply the pass, and extract the body."""
    mod = tvm.IRModule.from_expr(expr)
    mod = transform.InferType()(mod)
    mod = opt_pass(mod)
    mod = transform.InferType()(mod)
    return mod["main"].body


def test_identity_cast_all_dtypes():
    """TC1: Identity cast removal across dtype families."""
    xf16 = relay.var("x", shape=(4,), dtype="float16")
    xi32 = relay.var("y", shape=(4,), dtype="int32")

    before_f = relay.cast(xf16, "float16")
    before_i = relay.cast(xi32, "int32")

    tvm.ir.assert_structural_equal(run_opt_pass(before_f, FOLD), xf16)
    tvm.ir.assert_structural_equal(run_opt_pass(before_i, FOLD), xi32)


def test_lossless_roundtrip_fp16_and_bf16():
    """TC2: Lossless float round-trip (FP16 and BF16 via FP32)."""
    x16 = relay.var("x", shape=(4,), dtype="float16")
    xbf = relay.var("x", shape=(4,), dtype="bfloat16")

    chain16 = relay.cast(relay.cast(x16, "float32"), "float16")
    chainbf = relay.cast(relay.cast(xbf, "float32"), "bfloat16")

    tvm.ir.assert_structural_equal(run_opt_pass(chain16, FOLD), x16)
    tvm.ir.assert_structural_equal(run_opt_pass(chainbf, FOLD), xbf)


def test_lossy_roundtrip_preserved():
    """TC3: Lossy round-trip preserved (the quantization safeguard)."""
    w = relay.var("w", shape=(4,), dtype="float32")
    chain = relay.cast(relay.cast(w, "float16"), "float32")

    out = run_opt_pass(chain, FOLD)
    tvm.ir.assert_structural_equal(out, chain)


def test_deep_chain_collapses_to_minimum():
    """TC4: Multi-hop chain collapses fully (recursion correctness)."""
    x = relay.var("x", shape=(4,), dtype="float16")
    # fp16 -> fp32 -> fp16 -> fp32 -> fp16 : all lossless hops, should collapse to just x
    chain = relay.cast(relay.cast(relay.cast(relay.cast(x, "float32"), "float16"), "float32"), "float16")
    
    out = run_opt_pass(chain, FOLD)
    tvm.ir.assert_structural_equal(out, x)


def test_int_roundtrip_lossless_and_lossy():
    """TC5: Integer round-trips (lossless vs lossy analog)."""
    small = relay.var("s", shape=(4,), dtype="int8")
    big = relay.var("b", shape=(4,), dtype="int32")

    # int8 -> int32 -> int8 : exact round trip, should fold
    lossless = relay.cast(relay.cast(small, "int32"), "int8")
    tvm.ir.assert_structural_equal(run_opt_pass(lossless, FOLD), small)

    # int32 -> int8 -> int32 : truncates, must be preserved
    lossy = relay.cast(relay.cast(big, "int8"), "int32")
    out = run_opt_pass(lossy, FOLD)
    tvm.ir.assert_structural_equal(out, lossy)


def test_cross_domain_int_float_not_folded():
    """TC6: Cross-domain (int <-> float) round-trip must NOT fold."""
    xi = relay.var("x", shape=(4,), dtype="int32")
    # int32 -> float32 -> int32
    chain = relay.cast(relay.cast(xi, "float32"), "int32")
    
    out = run_opt_pass(chain, FOLD)
    tvm.ir.assert_structural_equal(out, chain)


def test_no_cast_commuting_through_add():
    """TC7: No commuting through arithmetic """
    x = relay.var("x", shape=(4,), dtype="float16")
    y = relay.var("y", shape=(4,), dtype="float16")

    add_fp32 = relay.add(relay.cast(x, "float32"), relay.cast(y, "float32"))
    result = relay.cast(add_fp32, "float16")

    out = run_opt_pass(result, FOLD)
    tvm.ir.assert_structural_equal(out, result)


def test_tuple_bool_and_fanout_safety():
    """TC8: Structural safety: Tuple/Bool + shared (fan-out) cast."""
    x16 = relay.var("x", shape=(4,), dtype="float16")
    xb = relay.var("b", shape=(4,), dtype="bool")

    # (a) Tuple safety
    t = relay.Tuple([relay.cast(x16, "float32"), relay.cast(xb, "int32")])
    item0 = relay.TupleGetItem(t, 0)
    out_tuple = run_opt_pass(item0, FOLD)

    # (b) one cast feeds two consumers
    shared = relay.cast(x16, "float32")           
    branch_a = relay.cast(shared, "float16")     
    branch_b = relay.add(shared, shared)          
    
    out = relay.Tuple([branch_a, branch_b])
    result = run_opt_pass(out, FOLD)
    
    expected_branch_b = relay.add(shared, shared)
    expected_out = relay.Tuple([x16, expected_branch_b])
    
    tvm.ir.assert_structural_equal(result, expected_out)


def test_transitive_two_cast_compression():
    """TC9: Transitive 2-cast compression (2 casts -> 1 cast).
    Verifies that unnecessarily wide intermediate hops are bypassed:
      - int8  -> int32  -> int16   ==>  cast(x, "int16")
      - fp16  -> fp64   -> fp32    ==>  cast(x, "float32")
      - uint8 -> uint32 -> uint16  ==>  cast(x, "uint16")
    """
    x_i8  = relay.var("x_i8", shape=(4,), dtype="int8")
    x_f16 = relay.var("x_f16", shape=(4,), dtype="float16")
    x_u8  = relay.var("x_u8", shape=(4,), dtype="uint8")

    # (a) Signed Integer: int8 -> int32 -> int16 compresses to int8 -> int16
    chain_int = relay.cast(relay.cast(x_i8, "int32"), "int16")
    expected_int = relay.cast(x_i8, "int16")
    tvm.ir.assert_structural_equal(run_opt_pass(chain_int, FOLD), expected_int)

    # (b) Float: fp16 -> fp64 -> fp32 compresses to fp16 -> fp32
    chain_float = relay.cast(relay.cast(x_f16, "float64"), "float32")
    expected_float = relay.cast(x_f16, "float32")
    tvm.ir.assert_structural_equal(run_opt_pass(chain_float, FOLD), expected_float)

    # (c) Unsigned Integer: uint8 -> uint32 -> uint16 compresses to uint8 -> uint16
    chain_uint = relay.cast(relay.cast(x_u8, "uint32"), "uint16")
    expected_uint = relay.cast(x_u8, "uint16")
    tvm.ir.assert_structural_equal(run_opt_pass(chain_uint, FOLD), expected_uint)


def test_cross_sign_uint_to_int_lossless_and_lossy():
    """TC10: Cross-sign uint <-> int round-trips.
      - uint8 -> int32 -> uint8 : lossless (0..255 fits in int32), should fold to x.
      - uint8 -> int8  -> uint8 : lossy (values > 127 overflow), must NOT fold.
    """
    xu = relay.var("u", shape=(4,), dtype="uint8")

    # (a) Lossless: uint8 -> int32 -> uint8 (32 > 8 bits, exact round-trip)
    lossless_cross = relay.cast(relay.cast(xu, "int32"), "uint8")
    tvm.ir.assert_structural_equal(run_opt_pass(lossless_cross, FOLD), xu)

    # (b) Lossy: uint8 -> int8 -> uint8 (8 == 8 bits, sign overflow risk, preserved)
    lossy_cross = relay.cast(relay.cast(xu, "int8"), "uint8")
    out_lossy = run_opt_pass(lossy_cross, FOLD)
    tvm.ir.assert_structural_equal(out_lossy, lossy_cross)


if __name__ == "__main__":
    tvm.testing.main()
