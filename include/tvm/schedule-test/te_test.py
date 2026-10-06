import numpy as np
import torch
import tvm
from tvm import te, topi

# Define Shapes and Parameters
p = 2.0
axis = -1
shape = (1, 3, 256, 1024)
dtype = "float32"

target = tvm.target.Target("cuda")
dev = tvm.cuda(0)

# Setup TE Compute
A = te.placeholder(shape, name="A", dtype=dtype)
out = topi.nn.lp_norm(A, ord=p, axis=axis, keepdims=True, dtype=dtype)

# Naive Schedule (Basic 1D Thread/Block Binding for CUDA)
s_naive = te.create_schedule(out.op)
te.schedule.AutoInlineInjective(s_naive)
T_red = out.op.input_tensors[0]
fused_axis = s_naive[out].fuse(*out.op.axis)
bx, tx = s_naive[out].split(fused_axis, factor=64)
s_naive[out].bind(bx, te.thread_axis("blockIdx.x"))
s_naive[out].bind(tx, te.thread_axis("threadIdx.x"))
s_naive[T_red].compute_at(s_naive[out], tx)
f_naive = tvm.build(s_naive, [A, out], target=target, name="lp_norm_naive")

# schedule_reduce
with target:
    s_reduce = topi.cuda.schedule_reduce(out)
f_reduce = tvm.build(s_reduce, [A, out], target=target, name="lp_norm_reduce")

# schedule_lp_norm
with target:
    s_lp_norm = topi.cuda.schedule_lp_norm(out)
f_lp_norm = tvm.build(s_lp_norm, [A, out], target=target, name="lp_norm_lp_norm")

# Correctness Check against PyTorch Reference
np_a = np.random.uniform(-1.0, 1.0, size=shape).astype(dtype)
torch_a = torch.from_numpy(np_a)

# Reference from PyTorch (CPU)
ref_out = torch.linalg.vector_norm(torch_a, ord=p, dim=axis, keepdim=True).numpy()

a_nd = tvm.nd.array(np_a, dev)
out_naive_nd = tvm.nd.empty(ref_out.shape, dtype, dev)
out_reduce_nd = tvm.nd.empty(ref_out.shape, dtype, dev)
out_lp_norm_nd = tvm.nd.empty(ref_out.shape, dtype, dev)

f_naive(a_nd, out_naive_nd)
f_reduce(a_nd, out_reduce_nd)
f_lp_norm(a_nd, out_lp_norm_nd)

# Assert numerical correctness
np.testing.assert_allclose(out_naive_nd.numpy(), ref_out, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(out_reduce_nd.numpy(), ref_out, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(out_lp_norm_nd.numpy(), ref_out, rtol=1e-4, atol=1e-4)
print("Correctness Verified: All three schedules match PyTorch reference!")

# Performance Benchmarking
def run_benchmark(f_mod, out_buf):
    evaluator = f_mod.time_evaluator(f_mod.entry_name, dev, number=200, repeat=10)
    times = np.array(evaluator(a_nd, out_buf).results) * 1000.0
    return np.mean(times), np.median(times)

naive_mean, naive_med = run_benchmark(f_naive, out_naive_nd)
reduce_mean, reduce_med = run_benchmark(f_reduce, out_reduce_nd)
lp_norm_mean, lp_norm_med = run_benchmark(f_lp_norm, out_lp_norm_nd)


print(f"{'Schedule':<20} | {'Median (ms)':<12} | {'Mean (ms)':<10}")
print("-" * 48)

print(f"{'Naive (Baseline)':<20} | {naive_med:>10.4f} ms | {naive_mean:>8.4f} ms")
print(f"{'schedule_reduce':<20} | {reduce_med:>10.4f} ms | {reduce_mean:>8.4f} ms")
print(f"{'schedule_lp_norm':<20} | {lp_norm_med:>10.4f} ms | {lp_norm_mean:>8.4f} ms")
print("-" * 48)

print(f"Speedup vs Naive (schedule_reduce):  {naive_med / reduce_med:.2f}x")
print(f"Speedup vs Naive (schedule_lp_norm): {naive_med / lp_norm_med:.2f}x")
print(f"Speedup vs schedule_reduce:          {reduce_med / lp_norm_med:.2f}x")
