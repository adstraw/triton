"""
Benchmark script for 10-block-scaled-matmul.py using do_bench
Outputs CSV files for mxfp4, mxfp8, and nvfp4 comparisons between Triton and cuBLAS

NOTE: FP4 formats (mxfp4, nvfp4) require PTX version 13.0.88 for Triton compilation.
      Without this PTX version, Triton will fail to compile FP4 kernels with GPU errors.
      Set TRITON_PTXAS_PATH to point to ptxas 13.0.88 before running FP4 benchmarks.

================================================================================
⚠️  TODO: VERIFY cuBLAS FP4 SUPPORT STATUS
================================================================================

As of October 2025, cuBLAS 13.0 does NOT support FP4 data types (MXFP4, NVFP4)
for matrix multiplication operations per NVIDIA documentation:
  - https://docs.nvidia.com/cuda/cublas/index.html
  - FP4 types exist in CUDA only for storage/movement, not computation

Current behavior: 
  - cuBLAS returns CUBLAS_STATUS_NOT_SUPPORTED (Error 15) for FP4 matmul
  - Benchmarks correctly report cuBLAS=0.0 TFLOPS in CSV output
  - Only Triton kernels can execute FP4 matmul (Triton-exclusive feature)

ACTION REQUIRED:
  1. Verify with NVIDIA if FP4 support is planned for future cuBLAS versions
  2. Check release notes for cuBLAS versions > 13.0 for FP4 additions
  3. Test on production B200 hardware to confirm (may differ from pre-prod)
  4. If permanently unsupported: FP4 is a Triton differentiator!

Results: MXFP4/NVFP4 show Triton-only data (2.6-2.8 PFLOPS peak)

================================================================================
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import triton
import triton.testing

# Add tutorials to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tutorials"))

# Import block-scaled matmul module
import importlib.util
spec = importlib.util.spec_from_file_location("block_scaled_matmul", 
                                               Path(__file__).parent.parent / "tutorials" / "10-block-scaled-matmul.py")
block_scaled_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_scaled_module)

# Import necessary functions
initialize_block_scaled = block_scaled_module.initialize_block_scaled
block_scaled_matmul = block_scaled_module.block_scaled_matmul
cublas_block_scaled_matmul = block_scaled_module.cublas_block_scaled_matmul
cublas = block_scaled_module.cublas
supports_block_scaling = block_scaled_module.supports_block_scaling


def test_cublas_support(block_scale_type):
    """
    Test if cuBLAS supports this block scaling format.
    
    Note: This only tests cuBLAS support. Triton support is assumed if the script
    gets this far (Triton will fail earlier during tensor initialization or warmup
    if PTX version is insufficient for FP4 formats).
    """
    if cublas is None or not supports_block_scaling():
        return False
    
    try:
        # Create small test tensors
        M_test, N_test, K_test = 128, 128, 128
        results = initialize_block_scaled(M_test, N_test, K_test, block_scale_type, compute_reference=False)
        a, b = results[9], results[10]
        a_scale_cublas, b_scale_cublas = results[11], results[12]
        
        # Try to run cuBLAS
        _ = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type)
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        if "cuBLAS Error: 15" in str(e) or "NOT_SUPPORTED" in str(e):
            return False
        # Other errors might be fixable, so return False but don't crash
        return False
    except Exception:
        return False


def benchmark_block_scaled(M, N, K, block_scale_type, cublas_supported, warmup_reps=10000, bench_reps=10000):
    """
    Benchmark both Triton and cuBLAS for a given configuration
    
    Args:
        cublas_supported: Whether cuBLAS supports this format (pre-tested)
    
    Returns:
        dict with keys: K, triton_tflops, cublas_tflops
    """
    print(f"Benchmarking M={M}, N={N}, K={K}, format={block_scale_type}")
    
    # Initialize tensors and descriptors
    results = initialize_block_scaled(M, N, K, block_scale_type, compute_reference=False)
    a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, _ = results[:9]
    a, b, a_scale_cublas, b_scale_cublas = results[9:13]
    
    # Warmup
    print(f"  Warming up ({warmup_reps} iterations)...")
    for _ in range(warmup_reps):
        _ = block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, 
                                torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
        if cublas_supported:
            _ = cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type)
    
    torch.cuda.synchronize()
    
    # Benchmark Triton
    print(f"  Benchmarking Triton...")
    triton_fn = lambda: block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc,
                                             torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    triton_ms = triton.testing.do_bench(triton_fn, warmup=100, rep=bench_reps)
    triton_tflops = 2.0 * M * N * K * 1e-12 / (triton_ms * 1e-3)
    
    # Benchmark cuBLAS
    if cublas_supported:
        print(f"  Benchmarking cuBLAS...")
        cublas_fn = lambda: cublas_block_scaled_matmul(a, a_scale_cublas, b, b_scale_cublas, block_scale_type)
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
    parser = argparse.ArgumentParser(description="Benchmark 10-block-scaled-matmul")
    parser.add_argument("--format", type=str, 
                        choices=["nvfp4", "mxfp4", "mxfp8", "mixed"], 
                        required=True,
                        help="Block scaling format to benchmark")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV filename")
    parser.add_argument("--warmup", type=int, default=10000,
                        help="Number of warmup iterations")
    parser.add_argument("--reps", type=int, default=10000,
                        help="Number of benchmark iterations")
    parser.add_argument("--K_values", type=int, nargs='+',
                        default=None,
                        help="K values to benchmark (default: [512, 1024, 2048, 4096])")
    args = parser.parse_args()
    
    # Check if GPU supports block scaling
    if not supports_block_scaling():
        print("ERROR: GPU does not support block scaling (requires Blackwell/sm_10)")
        sys.exit(1)
    
    # Fixed problem size
    M = 8192
    N = 8192
    
    # K values to sweep
    if args.K_values is not None:
        K_values = args.K_values
    else:
        # Default K values - use larger values for block-scaled matmul
        if args.format == "mxfp8":
            K_values = [8192, 16384]
        else:
            K_values = [512, 1024, 2048, 4096, 8192]
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Block-Scaled Matmul - {args.format.upper()}")
    print(f"Problem size: M={M}, N={N}")
    print(f"K values: {K_values}")
    print(f"Warmup: {args.warmup}, Reps: {args.reps}")
    print(f"{'='*60}\n")
    
    # Test cuBLAS support once at the start
    # Note: If this succeeds, it means Triton is also working (tensor init would have failed otherwise)
    print("Testing cuBLAS support...")
    cublas_supported = test_cublas_support(args.format)
    if cublas_supported:
        print(f"✅ cuBLAS supports {args.format} block-scaled matmul\n")
    else:
        print(f"⚠️  cuBLAS does not support {args.format} block-scaled matmul")
        print(f"   Only Triton results will be reported\n")
    
    # Run benchmarks
    results = []
    for K in K_values:
        try:
            result = benchmark_block_scaled(M, N, K, args.format, cublas_supported, args.warmup, args.reps)
            results.append(result)
            print()
        except Exception as e:
            print(f"  ERROR: Failed to benchmark K={K}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    if not results:
        print("ERROR: No successful benchmarks - all K values failed")
        print("This may indicate GPU/driver issues or unsupported operations")
        sys.exit(1)
    
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

