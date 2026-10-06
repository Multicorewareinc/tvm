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
# pylint: disable=invalid-name
"""scheduler functions for cuda backend"""
from __future__ import absolute_import as _abs

import tvm
from tvm import te
from ..utils import traverse_inline


def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    def _callback(op):
        if "sqr_sum" in op.tag:
            pad = op.input_tensors[0]
            s[pad].compute_inline()
            fused_axis = s[outs[0]].fuse(*s[outs[0]].op.axis)
            bx, tx = s[outs[0]].split(fused_axis, factor=max_threads)
            s[outs[0]].bind(bx, te.thread_axis("blockIdx.x"))
            s[outs[0]].bind(tx, te.thread_axis("threadIdx.x"))
            s[op].compute_at(s[outs[0]], tx)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_lp_norm(outs):
    out = outs.op if isinstance(outs, te.tensor.Tensor) else outs[0].op
    s = te.create_schedule(out)
    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    visited = set()
    reduce_op = None

    def _traverse(op):
        nonlocal reduce_op
        if op in visited:
            return
        visited.add(op)

        for tensor in op.input_tensors:
            if isinstance(tensor.op, te.ComputeOp):
                _traverse(tensor.op)

        if isinstance(op, te.ComputeOp) and op != out:
            # Inline all injective ops (abs, pow, root_power)
            if len(op.reduce_axis) == 0:
                s[op].compute_inline()
            else:
                # Capture the single reduction op (sum, max, or min)
                reduce_op = op

    _traverse(out)

    if reduce_op is not None and len(reduce_op.reduce_axis) > 0:
        # Fusing reduction axes
        fused_reduce = s[reduce_op].fuse(*reduce_op.reduce_axis)
        ko, ki = s[reduce_op].split(fused_reduce, factor=32)
        # Parallelization via rfactor
        data_out_rf = s.rfactor(reduce_op.output(0), ki)
        tx = s[reduce_op].op.reduce_axis[0]
        thread_x = te.thread_axis((0, 32), "threadIdx.x")
        s[reduce_op].bind(tx, thread_x)
        s[data_out_rf].compute_at(s[reduce_op], tx)
        # Scheduling Output Dimensions
        if len(out.axis) > 0:
            fused_outer = s[out].fuse(*out.axis)
            bx, outer_in = s[out].split(fused_outer, factor=16)
            thread_y = te.thread_axis((0, 16), "threadIdx.y")
            block_x = te.thread_axis("blockIdx.x")
            s[out].bind(outer_in, thread_y)
            s[out].bind(bx, block_x)
            s[reduce_op].compute_at(s[out], outer_in)
        else:
            # Global reduction (scalar output)
            s[out].bind(fused_reduce, te.thread_axis("blockIdx.x"))
        
    return s
