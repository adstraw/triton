"""
Benchmark script for 09-persistent-matmul.py using do_bench
Outputs CSV files for fp8 and fp16 comparisons between Triton and cuBLAS
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import triton
import triton.testing
from triton.tools.tensor_descriptor import TensorDescriptor

# Import cuBLAS if available
if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


HAS_TENSOR_DESC = supports_tma() and hasattr(triton.language, "make_tensor_descriptor")

# Load the matmul kernels from tutorial 09
sys.path.insert(0, str(Path(__file__).parent.parent / "tutorials"))
import importlib.util
spec = importlib.util.spec_from_file_location("persistent_matmul", 
                                               Path(__file__).parent.parent / "tutorials" / "09-persistent-matmul.py")
persistent_matmul_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(persistent_matmul_module)

matmul_tma_persistent = persistent_matmul_module.matmul_tma_persistent
matmul_descriptor_persistent = persistent_matmul_module.matmul_descriptor_persistent


def benchmark_matmul(M, N, K, dtype, warmup_reps=10000, bench_reps=10000):
    """
    Benchmark both Triton and cuBLAS for a given configuration
    
    Returns:
        dict with keys: K, triton_tflops, cublas_tflops
    """
    print(f"Benchmarking M={M}, N={N}, K={K}, dtype={dtype}")
    
    # Create input tensors
    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(dtype)
    b_T = b.T.contiguous()  # cuBLAS and TMA kernels expect transposed B
    
    # Choose best available Triton kernel
    if HAS_TENSOR_DESC:
        # Use descriptor persistent (best for Blackwell)
        triton_fn = lambda: matmul_descriptor_persistent(a, b_T, warp_specialize=True)
        kernel_name = "descriptor_persistent"
    else:
        # Fallback to regular matmul
        triton_fn = lambda: persistent_matmul_module.matmul_persistent(a, b.T)
        kernel_name = "persistent"
    
    # Warmup
    print(f"  Warming up ({warmup_reps} iterations using {kernel_name})...")
    for _ in range(warmup_reps):
        _ = triton_fn()
        if cublas is not None:
            c = torch.empty((M, N), device=a.device, dtype=dtype)
            cublas.matmul(a, b_T, c)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    print(f"  Benchmarking Triton ({kernel_name})...")
    triton_ms = triton.testing.do_bench(triton_fn, warmup=100, rep=bench_reps)
    triton_tflops = 2.0 * M * N * K * 1e-12 / (triton_ms * 1e-3)
    
    # Benchmark cuBLAS
    print(f"  Benchmarking cuBLAS...")
    if cublas is not None:
        c = torch.empty((M, N), device=a.device, dtype=dtype)
        cublas_fn = lambda: cublas.matmul(a, b_T, c)
        cublas_ms = triton.testing.do_bench(cublas_fn, warmup=100, rep=bench_reps)
        cublas_tflops = 2.0 * M * N * K * 1e-12 / (cublas_ms * 1e-3)
    else:
        cublas_tflops = 0.0
    
    print(f"  Results: Triton={triton_tflops:.2f} TFLOPS, cuBLAS={cublas_tflops:.2f} TFLOPS")
    
    return {
        "K": K,
        "triton_tflops": triton_tflops,
        "cublas_tflops": cublas_tflops
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark 09-persistent-matmul")
    parser.add_argument("--prec", type=str, choices=["fp8", "fp16"], required=True,
                        help="Precision to benchmark (fp8 or fp16)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV filename")
    parser.add_argument("--warmup", type=int, default=10000,
                        help="Number of warmup iterations")
    parser.add_argument("--reps", type=int, default=10000,
                        help="Number of benchmark iterations")
    args = parser.parse_args()
    
    # Fixed problem size
    M = 8192
    N = 8192
    
    # K values to sweep (matching email data: 128 increments from 128 to 1024)
    K_values = [128, 256, 384, 512, 640, 768, 896, 1024]
    
    # Determine dtype
    if args.prec == 'fp8':
        if not hasattr(torch, "float8_e4m3fn"):
            print("ERROR: FP8 not available in this PyTorch version")
            sys.exit(1)
        dtype = torch.float8_e4m3fn
    else:
        dtype = torch.float16
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Persistent Matmul - {args.prec.upper()}")
    print(f"Problem size: M={M}, N={N}")
    print(f"K values: {K_values}")
    print(f"Warmup: {args.warmup}, Reps: {args.reps}")
    print(f"{'='*60}\n")
    
    # Run benchmarks
    results = []
    for K in K_values:
        result = benchmark_matmul(M, N, K, dtype, args.warmup, args.reps)
        results.append(result)
        print()
    
    # Write CSV
    output_path = Path(args.output)
    print(f"Writing results to {output_path}")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["K", "triton_tflops", "cublas_tflops"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDone! Results saved to {output_path}")
    print("\nSummary:")
    print(f"{'K':<8} {'Triton TFLOPS':<20} {'cuBLAS TFLOPS':<20}")
    print("-" * 50)
    for r in results:
        print(f"{r['K']:<8} {r['triton_tflops']:<20.2f} {r['cublas_tflops']:<20.2f}")


if __name__ == "__main__":
    main()

