#!/usr/bin/env python3
import time
import platform
import importlib
import sys
import torch
import numpy as np

# Try to import MLX
try:
    mlx = importlib.import_module('mlx')
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

# Helper for pretty printing
HEADER = """
# Environment TFLOPS Benchmark Report

| Backend | Precision | TFLOPS | Notes |
|---------|-----------|--------|-------|
"""

PRECISIONS = [
    (torch.float64, 64, 'float64'),
    (torch.float32, 32, 'float32'),
    (torch.float16, 16, 'float16'),
    (torch.bfloat16, 16, 'bfloat16'),
]
# 8 and 4 bit: PyTorch does not natively support matmul for these, so we skip or note.

# Matrix size for benchmark (large enough for GPU, not too big for CPU)
N = 4096
REPEATS = 10

results = []

def benchmark_torch(device, dtype, bits, label):
    # 8/4 bit not supported for matmul, skip
    if bits < 16:
        results.append((device, f'int{bits}', 'N/A', 'Not supported for matmul'))
        return
    a = torch.randn((N, N), device=device, dtype=dtype)
    b = torch.randn((N, N), device=device, dtype=dtype)
    # Warmup
    for _ in range(2):
        c = torch.matmul(a, b)
    torch.cuda.synchronize() if device == 'cuda' else None
    # Benchmark
    start = time.time()
    for _ in range(REPEATS):
        c = torch.matmul(a, b)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    elapsed = end - start
    ops = 2 * N * N * N * REPEATS
    tflops = ops / elapsed / 1e12
    results.append((device, label, f'{tflops:.2f}', ''))

def benchmark_cpu():
    for dtype, bits, label in PRECISIONS:
        benchmark_torch('cpu', dtype, bits, label)
    # 8/4 bit note
    results.append(('cpu', 'int8', 'N/A', 'Not supported for matmul'))
    results.append(('cpu', 'int4', 'N/A', 'Not supported for matmul'))

def benchmark_cuda():
    if not torch.cuda.is_available():
        return
    for dtype, bits, label in PRECISIONS:
        try:
            benchmark_torch('cuda', dtype, bits, label)
        except RuntimeError as e:
            results.append(('cuda', label, 'N/A', f'Error: {e}'))
    results.append(('cuda', 'int8', 'N/A', 'Not supported for matmul'))
    results.append(('cuda', 'int4', 'N/A', 'Not supported for matmul'))

def benchmark_mlx():
    if not HAS_MLX:
        return
    # MLX supports float64, float32, float16, bfloat16
    mlx_precisions = [
        (mx.float64, 64, 'float64'),
        (mx.float32, 32, 'float32'),
        (mx.float16, 16, 'float16'),
        (mx.bfloat16, 16, 'bfloat16'),
    ]
    for dtype, bits, label in mlx_precisions:
        a = mx.array(np.random.randn(N, N), dtype=dtype)
        b = mx.array(np.random.randn(N, N), dtype=dtype)
        # Warmup
        for _ in range(2):
            c = mx.matmul(a, b)
        mx.eval(c)
        start = time.time()
        for _ in range(REPEATS):
            c = mx.matmul(a, b)
        mx.eval(c)
        end = time.time()
        elapsed = end - start
        ops = 2 * N * N * N * REPEATS
        tflops = ops / elapsed / 1e12
        results.append(('mlx', label, f'{tflops:.2f}', ''))
    results.append(('mlx', 'int8', 'N/A', 'Not supported for matmul'))
    results.append(('mlx', 'int4', 'N/A', 'Not supported for matmul'))

def main():
    print("\nRunning environment TFLOPS benchmark...\n")
    print(f"Python: {platform.python_version()} | Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if HAS_MLX:
        print(f"MLX: {mlx.__version__ if hasattr(mlx, '__version__') else 'present'}")
    else:
        print("MLX: not installed")
    print("\nThis may take a minute...\n")
    benchmark_cpu()
    benchmark_cuda()
    benchmark_mlx()
    print(HEADER, end='')
    for backend, precision, tflops, notes in results:
        print(f"| {backend} | {precision} | {tflops} | {notes} |")
    print("\n**Legend:** TFLOPS = Trillions of FLoating-point Operations Per Second (higher is better).\n")
    print("**Note:** int8/int4 matmul is not natively supported in PyTorch/MLX, so only float/bfloat types are benchmarked.\n")

if __name__ == "__main__":
    main() 