"""
Master script to run all Triton vs cuBLAS benchmarks and generate CSVs

This script runs:
1. Tutorial 09 (persistent matmul): fp8, fp16
2. Tutorial 10 (block-scaled matmul): mxfp4, mxfp8, nvfp4

Outputs 5 CSV files ready for charting.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import time

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"


def run_benchmark(script_name, args, description, env_vars=None):
    """
    Run a benchmark script and return success status
    
    Args:
        script_name: Name of the script to run
        args: List of command-line arguments
        description: Description for logging
        env_vars: Optional dict of environment variables to set
    """
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: python {script_name} {' '.join(args)}")
    if env_vars:
        print(f"Environment: {env_vars}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        cmd = [sys.executable, str(SCRIPT_DIR / script_name)] + args
        
        # Prepare environment (inherit current env and add custom vars)
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            env=env
        )
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f}s (exit code: {e.returncode})")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f}s: {e}")
        return False


def setup_ptx_13_0_88():
    """
    Setup PTX 13.0.88 required for FP4 support.
    Returns the path to ptxas binary or None if setup fails.
    """
    ptx_dir = SCRIPT_DIR / "ptx_13_0_88"
    ptxas_path = ptx_dir / "cuda_nvcc-linux-x86_64-13.0.88-archive" / "bin" / "ptxas"
    
    # Check if already set up
    if ptxas_path.exists():
        print(f"✓ PTX 13.0.88 already installed at: {ptxas_path}")
        return str(ptxas_path.absolute())
    
    # Run setup script
    print(f"\n{'='*70}")
    print("Setting up PTX 13.0.88 for FP4 support...")
    print(f"{'='*70}\n")
    
    setup_script = SCRIPT_DIR / "setup_ptx_13_0_88.sh"
    if not setup_script.exists():
        print(f"❌ Setup script not found: {setup_script}")
        return None
    
    try:
        # Make script executable
        os.chmod(setup_script, 0o755)
        
        # Run setup script
        result = subprocess.run(
            [str(setup_script), str(ptx_dir)],
            check=True,
            text=True
        )
        
        # Verify installation
        if ptxas_path.exists():
            print(f"\n✓ PTX 13.0.88 successfully installed")
            return str(ptxas_path.absolute())
        else:
            print(f"\n❌ PTX setup completed but binary not found at: {ptxas_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ PTX setup failed (exit code: {e.returncode})")
        return None
    except Exception as e:
        print(f"\n❌ PTX setup failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run all Triton vs cuBLAS benchmarks")
    parser.add_argument("--warmup", type=int, default=10000,
                        help="Number of warmup iterations (default: 10000)")
    parser.add_argument("--reps", type=int, default=10000,
                        help="Number of benchmark iterations (default: 10000)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for CSV files (default: ./benchmark_results)")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Skip the Triton rebuild warning")
    parser.add_argument("--ptx-path", type=str,
                        help="Path to PTX 13.0.88 binary (if not specified, will auto-download)")
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup PTX 13.0.88 for tutorial 10 (required for FP4 support)
    ptx_path = None
    if args.ptx_path:
        ptx_path = args.ptx_path
        print(f"\n✓ Using provided PTX path: {ptx_path}")
    else:
        # Check if PTX is already in environment
        ptx_path = os.environ.get("TRITON_PTXAS_PATH")
        if ptx_path:
            print(f"\n✓ Using PTX from environment: {ptx_path}")
        else:
            # Auto-download PTX 13.0.88
            ptx_path = setup_ptx_13_0_88()
            if not ptx_path:
                print("\n⚠️  WARNING: PTX 13.0.88 setup failed!")
                print("    Tutorial 10 (FP4 benchmarks) may not work correctly.")
                print("    You can manually set TRITON_PTXAS_PATH or use --ptx-path")
    
    print(f"\n{'='*70}")
    print(f"TRITON vs cuBLAS BENCHMARK SUITE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.reps}")
    if ptx_path:
        print(f"PTX 13.0.88: {ptx_path}")
    print(f"{'='*70}")
    
    if not args.skip_rebuild:
        print("\n⚠️  IMPORTANT: Make sure you have rebuilt Triton after applying C++ changes!")
        print("   Run: pip install -e . --no-build-isolation")
        print("   (Use --skip-rebuild to skip this warning)")
        response = input("\nHave you rebuilt Triton? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Please rebuild Triton first.")
            sys.exit(1)
    
    # Define all benchmarks to run
    benchmarks = [
        {
            "script": "bench_09_persistent_matmul.py",
            "args": [
                "--prec", "fp16",
                "--output", str(output_dir / "fp16_results.csv"),
                "--warmup", str(args.warmup),
                "--reps", str(args.reps)
            ],
            "description": "Tutorial 09 - FP16 Persistent Matmul"
        },
        {
            "script": "bench_09_persistent_matmul.py",
            "args": [
                "--prec", "fp8",
                "--output", str(output_dir / "fp8_results.csv"),
                "--warmup", str(args.warmup),
                "--reps", str(args.reps)
            ],
            "description": "Tutorial 09 - FP8 Persistent Matmul"
        },
        {
            "script": "bench_10_block_scaled_matmul.py",
            "args": [
                "--format", "mxfp4",
                "--output", str(output_dir / "mxfp4_results.csv"),
                "--warmup", str(args.warmup),
                "--reps", str(args.reps)
            ],
            "description": "Tutorial 10 - MXFP4 Block-Scaled Matmul",
            "requires_ptx": True
        },
        {
            "script": "bench_10_block_scaled_matmul.py",
            "args": [
                "--format", "nvfp4",
                "--output", str(output_dir / "nvfp4_results.csv"),
                "--warmup", str(args.warmup),
                "--reps", str(args.reps)
            ],
            "description": "Tutorial 10 - NVFP4 Block-Scaled Matmul",
            "requires_ptx": True
        },
        {
            "script": "bench_10_block_scaled_matmul.py",
            "args": [
                "--format", "mxfp8",
                "--output", str(output_dir / "mxfp8_results.csv"),
                "--warmup", str(args.warmup),
                "--reps", str(args.reps),
                "--K_values", "8192", "16384"
            ],
            "description": "Tutorial 10 - MXFP8 Block-Scaled Matmul",
            "requires_ptx": True
        },
    ]
    
    # Run all benchmarks
    results = []
    total_start = time.time()
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Benchmark {i}/{len(benchmarks)}")
        print(f"{'#'*70}")
        
        # Check if this benchmark requires PTX 13.0.88
        env_vars = None
        if benchmark.get("requires_ptx", False):
            if ptx_path:
                env_vars = {"TRITON_PTXAS_PATH": ptx_path}
                print(f"Using PTX 13.0.88 for {benchmark['description']}")
            else:
                print(f"\n⚠️  WARNING: {benchmark['description']} requires PTX 13.0.88 but it's not available!")
                print("    This benchmark may fail. Proceeding anyway...")
        
        success = run_benchmark(
            benchmark["script"],
            benchmark["args"],
            benchmark["description"],
            env_vars=env_vars
        )
        results.append({
            "description": benchmark["description"],
            "success": success
        })
        
        # Brief pause between benchmarks
        if i < len(benchmarks):
            time.sleep(2)
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print(f"\n\n{'='*70}")
    print(f"BENCHMARK SUITE COMPLETE")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    print("Results Summary:")
    print(f"{'Benchmark':<50} {'Status':<10}")
    print("-" * 70)
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{result['description']:<50} {status:<10}")
    
    # Check if any failed
    failed_count = sum(1 for r in results if not r["success"])
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} benchmark(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All benchmarks completed successfully!")
        print(f"\nOutput files in: {output_dir}")
        print("CSV files:")
        for csv_file in sorted(output_dir.glob("*.csv")):
            print(f"  - {csv_file.name}")


if __name__ == "__main__":
    main()

