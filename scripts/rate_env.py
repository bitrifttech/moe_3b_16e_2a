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
N = 1024
REPEATS = 10

results = []

def benchmark_torch(device, dtype, bits, label):
    # 8/4 bit not supported for matmul, skip
    if bits < 16:
        print(f"[SKIP] {device.upper()} {label}: int{bits} not supported for matmul.")
        results.append((device, f'int{bits}', 'N/A', 'Not supported for matmul'))
        return
    print(f"[RUN] {device.upper()} {label}: benchmarking {N}x{N} matmul ...", end='', flush=True)
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
    print(f" done. {tflops:.2f} TFLOPS")
    results.append((device, label, f'{tflops:.2f}', ''))

def benchmark_cpu():
    print("\n[CPU] Benchmarking on CPU ...")
    for dtype, bits, label in PRECISIONS:
        benchmark_torch('cpu', dtype, bits, label)
    # 8/4 bit note
    print("[SKIP] CPU int8: not supported for matmul.")
    print("[SKIP] CPU int4: not supported for matmul.")
    results.append(('cpu', 'int8', 'N/A', 'Not supported for matmul'))
    results.append(('cpu', 'int4', 'N/A', 'Not supported for matmul'))

def benchmark_cuda():
    if not torch.cuda.is_available():
        print("\n[CUDA] CUDA not available, skipping.")
        return
    print("\n[CUDA] Benchmarking on CUDA GPU ...")
    for dtype, bits, label in PRECISIONS:
        try:
            benchmark_torch('cuda', dtype, bits, label)
        except RuntimeError as e:
            print(f"[ERROR] CUDA {label}: {e}")
            results.append(('cuda', label, 'N/A', f'Error: {e}'))
    print("[SKIP] CUDA int8: not supported for matmul.")
    print("[SKIP] CUDA int4: not supported for matmul.")
    results.append(('cuda', 'int8', 'N/A', 'Not supported for matmul'))
    results.append(('cuda', 'int4', 'N/A', 'Not supported for matmul'))

def benchmark_mlx():
    if not HAS_MLX:
        print("\n[MLX] MLX not installed, skipping.")
        return
    print("\n[MLX] Benchmarking on MLX (Apple Silicon) ...")
    # MLX supports float32, float16, bfloat16 (not float64)
    mlx_precisions = [
        (mx.float32, 32, 'float32'),
        (mx.float16, 16, 'float16'),
        (mx.bfloat16, 16, 'bfloat16'),
    ]
    # Skip float64
    print("[SKIP] MLX float64: not supported on MLX GPU.")
    results.append(('mlx', 'float64', 'N/A', 'Not supported on MLX GPU'))
    for dtype, bits, label in mlx_precisions:
        print(f"[RUN] MLX {label}: benchmarking {N}x{N} matmul ...", end='', flush=True)
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
        print(f" done. {tflops:.2f} TFLOPS")
        results.append(('mlx', label, f'{tflops:.2f}', ''))
    print("[SKIP] MLX int8: not supported for matmul.")
    print("[SKIP] MLX int4: not supported for matmul.")
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
    print("\n[REPORT] Benchmarking complete. Results below:\n")
    print(HEADER, end='')
    for backend, precision, tflops, notes in results:
        print(f"| {backend} | {precision} | {tflops} | {notes} |")
    print("\n**Legend:** TFLOPS = Trillions of FLoating-point Operations Per Second (higher is better).\n")
    print("**Note:** int8/int4 matmul is not natively supported in PyTorch/MLX, so only float/bfloat types are benchmarked.\n")

if __name__ == "__main__":
    main() 