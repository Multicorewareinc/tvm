import numpy as np
import torch
import tvm
from tvm import te, topi

# 1. Define Shapes and Parameters
p = 2.0
axis = -1
shape = (16, 256, 1024)
dtype = "float32"

target = tvm.target.Target("cuda")
dev = tvm.cuda(0)

# 2. Setup TE Compute
A = te.placeholder(shape, name="A", dtype=dtype)
out = topi.nn.lp_norm(A, ord=p, axis=axis, keepdims=True, dtype=dtype)

# -------------------------------------------------------------
# Schedule 1: Default / Generic CUDA Schedule (schedule_reduce)
# -------------------------------------------------------------
with target:
    s_default = topi.cuda.schedule_reduce(out)
f_default = tvm.build(s_default, [A, out], target=target, name="lp_norm_default")

# -------------------------------------------------------------
# Schedule 2: Your Custom Optimized CUDA Schedule (schedule_lp_norm)
# -------------------------------------------------------------
with target:
    s_custom = topi.cuda.schedule_lp_norm(out)
f_custom = tvm.build(s_custom, [A, out], target=target, name="lp_norm_custom")

# -------------------------------------------------------------
# 3. Correctness Check against PyTorch Reference
# -------------------------------------------------------------
np_a = np.random.uniform(-1.0, 1.0, size=shape).astype(dtype)
torch_a = torch.from_numpy(np_a)

# Reference from PyTorch (CPU)
ref_out = torch.linalg.vector_norm(torch_a, ord=p, dim=axis, keepdim=True).numpy()

# TVM Buffers on CUDA
a_nd = tvm.nd.array(np_a, dev)
out_default_nd = tvm.nd.empty(ref_out.shape, dtype, dev)
out_custom_nd = tvm.nd.empty(ref_out.shape, dtype, dev)

# Run both kernels
f_default(a_nd, out_default_nd)
f_custom(a_nd, out_custom_nd)

# Assert numerical correctness
np.testing.assert_allclose(out_default_nd.numpy(), ref_out, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(out_custom_nd.numpy(), ref_out, rtol=1e-4, atol=1e-4)
print("✅ Correctness Verified: Both Default and Custom schedules match PyTorch reference!")

# -------------------------------------------------------------
# 4. Performance Benchmarking
# -------------------------------------------------------------
# Benchmark Default / Generic Schedule
eval_default = f_default.time_evaluator(f_default.entry_name, dev, number=200, repeat=10)
prof_default = eval_default(a_nd, out_default_nd)
time_default = np.median(prof_default.results) * 1000.0  # ms
mean_default = np.mean(prof_default.results) * 1000.0
std_default = np.std(prof_default.results) * 1000.0

# Benchmark Custom Schedule
eval_custom = f_custom.time_evaluator(f_custom.entry_name, dev, number=200, repeat=10)
prof_custom = eval_custom(a_nd, out_custom_nd)
time_custom = np.median(prof_custom.results) * 1000.0  # ms
mean_custom = np.mean(prof_custom.results) * 1000.0
std_custom = np.std(prof_custom.results) * 1000.0

# -------------------------------------------------------------
# 5. Report Results & Speedup Comparison
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(f"  Target Device:                     {dev.device_name}")
print(f"  Input Shape:                       {shape}")
print(f"  Order (p):                         {p}, Axis: {axis}")
print("-" * 65)
print(f"  Default/Generic Schedule (Median): {time_default:.4f} ms  (Mean: {mean_default:.4f} ms)")
print(f"  Custom Schedule (Median):          {time_custom:.4f} ms  (Mean: {mean_custom:.4f} ms)")
print("-" * 65)
speedup = time_default / time_custom
print(f"  Speedup vs Default Schedule:       {speedup:.2f}x ({'faster' if speedup >= 1 else 'slower'})")
print("=" * 65)