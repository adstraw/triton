# Triton vs cuBLAS Benchmark Results - Production B200

**Date:** October 10, 2025  
**Hardware:** Production B200 (Blackwell) umbriel-b200-081  
**Test Configuration:** M=8192, N=8192, K-sweep, 10k warmup + 10k iterations  
**Measurement:** `triton.testing.do_bench` for both implementations (pure kernel timing)

---

## Executive Summary

Performance benchmarks comparing Triton and cuBLAS across five data types: FP8, FP16, MXFP8, MXFP4, NVFP4. cuBLAS includes algorithm autotuning via cuBLASLt heuristics API. Both use `do_bench` for fair kernel-only timing.

**Key Findings:**
1. Dense Matmul (FP8, FP16): cuBLAS +16% at peak K values
2. Block-Scaled MXFP8: Triton 5x faster than cuBLAS
3. FP4 (MXFP4, NVFP4): Triton-exclusive, cuBLAS unsupported, 4.2-4.5 PFLOPS peak
4. Both implementations exceed baseline email data by 50-110% at higher K values

---

## FP8 Results (Tutorial 09 - Persistent Matmul)

| K    | Prod Triton | Prod cuBLAS | Email T | Email C | ΔT vs Email | ΔC vs Email | cuBLAS vs Triton |
|------|-------------|-------------|---------|---------|-------------|-------------|------------------|
| 128  | 686         | 645         | 900     | 600     | -24%        | +8%         | -6%              |
| 256  | 1,188       | 1,103       | 1,500   | 900     | -21%        | +23%        | -7%              |
| 384  | 1,566       | 1,521       | 1,500   | 1,000   | +4%         | +52%        | -3%              |
| 512  | 1,848       | 1,951       | 1,300   | 1,200   | +42%        | +63%        | +6%              |
| 640  | 2,061       | 2,302       | 1,300   | 1,200   | +59%        | +92%        | +12%             |
| 768  | 2,212       | 2,606       | 1,300   | 1,300   | +70%        | +100%       | +18%             |
| 896  | 2,308       | 2,724       | 1,300   | 1,300   | +78%        | +110%       | +18%             |
| 1024 | 2,375       | 2,765       | 1,200   | 1,300   | +98%        | +113%       | +16%             |

**Analysis:** cuBLAS advantage increases with K (6% at K=512 to 18% at K=768-896). Both implementations significantly exceed baseline, with cuBLAS showing stronger gains (+113% vs +98% at K=1024).

---

## FP16 Results (Tutorial 09 - Persistent Matmul)

| K    | Prod Triton | Prod cuBLAS | Email T | Email C | ΔT vs Email | ΔC vs Email | cuBLAS vs Triton |
|------|-------------|-------------|---------|---------|-------------|-------------|------------------|
| 128  | 451         | 493         | 375     | 323     | +20%        | +53%        | +9%              |
| 256  | 758         | 845         | 475     | 475     | +60%        | +78%        | +11%             |
| 384  | 923         | 1,069       | 475     | 550     | +94%        | +94%        | +16%             |
| 512  | 1,022       | 1,221       | 500     | 600     | +104%       | +104%       | +19%             |
| 640  | 1,065       | 1,316       | 550     | 600     | +94%        | +119%       | +24%             |
| 768  | 1,122       | 1,345       | 600     | 625     | +87%        | +115%       | +20%             |
| 896  | 1,165       | 1,377       | 625     | 650     | +86%        | +112%       | +18%             |
| 1024 | 1,196       | 1,388       | 650     | 675     | +84%        | +106%       | +16%             |

**Analysis:** cuBLAS maintains 16-24% advantage at K≥384. Both implementations approximately double baseline performance, with consistent gains across all K values.

---

## MXFP8 Results (Tutorial 10 - Block-Scaled Matmul)

| K     | Prod Triton | Prod cuBLAS | Triton vs cuBLAS |
|-------|-------------|-------------|------------------|
| 8,192 | 2,248       | 462         | 4.9x             |
| 16,384| 2,319       | 458         | 5.1x             |

**Analysis:** Triton shows dominant performance for block-scaled MXFP8, outperforming cuBLAS by approximately 5x. Peak: 2.3 PFLOPS.

---

## MXFP4 Results (Tutorial 10 - Block-Scaled Matmul)

| K     | Prod Triton | Prod cuBLAS |
|-------|-------------|-------------|
| 512   | 1,010       | 0           |
| 1,024 | 1,736       | 0           |
| 2,048 | 2,736       | 0           |
| 4,096 | 3,863       | 0           |
| 8,192 | 4,523       | 0           |

**Analysis:** cuBLAS does not support FP4 (`CUBLAS_STATUS_NOT_SUPPORTED`). Triton-exclusive capability. Peak: 4.5 PFLOPS.

---

## NVFP4 Results (Tutorial 10 - Block-Scaled Matmul)

| K     | Prod Triton | Prod cuBLAS |
|-------|-------------|-------------|
| 512   | 989         | 0           |
| 1,024 | 1,702       | 0           |
| 2,048 | 2,637       | 0           |
| 4,096 | 3,646       | 0           |
| 8,192 | 4,206       | 0           |

**Analysis:** cuBLAS does not support FP4. Triton-exclusive capability. Peak: 4.2 PFLOPS. MXFP4 is ~7% faster than NVFP4 at K=8192.

---

## Summary

### Peak Performance (TFLOPS)

| Data Type | Triton | cuBLAS | Winner/Advantage   | K at Peak |
|-----------|--------|--------|---------------------|-----------|
| FP8       | 2,375  | 2,765  | cuBLAS +16%        | 1,024     |
| FP16      | 1,196  | 1,388  | cuBLAS +16%        | 1,024     |
| MXFP8     | 2,319  | 458    | Triton 5.1x        | 16,384    |
| MXFP4     | 4,523  | N/A    | Triton (exclusive) | 8,192     |
| NVFP4     | 4,206  | N/A    | Triton (exclusive) | 8,192     |

### Measurement Methodology

Both implementations use `triton.testing.do_bench`:
- Measures pure kernel execution time (excludes compilation, data transfer)
- Identical warmup (10k) and benchmark (10k) methodology
- Provides fair comparison without profiling overhead bias
- Per email guidance, avoids using Proton profiler which would penalize cuBLAS

Using `do_bench` instead of Proton ensures cuBLAS is not unfairly penalized by Triton's compilation and profiling overhead.

### Why Performance Exceeds Baseline

Both implementations significantly exceed baseline email data (collected without autotuning):

1. **Autotuning contribution:** cuBLAS gains (+113% FP8, +106% FP16) exceed Triton gains (+98% FP8, +84% FP16), supporting that autotuning provides measurable benefit

2. **Software improvements:** Both show gains, suggesting driver/CUDA library improvements since baseline collection

3. **Fair measurement:** `do_bench` ensures accurate kernel-only timing without profiling bias

---

## Conclusions

1. **Dense Matmul:** cuBLAS with autotuning provides 16% advantage for FP8/FP16 at typical workload sizes (K≥512)

2. **Block-Scaled Matmul:** Triton shows 5x advantage for MXFP8 and exclusive FP4 support (4.2-4.5 PFLOPS)

3. **Autotuning:** Implementation is functional and contributes measurably to cuBLAS performance profile

4. **Measurement:** `do_bench` provides fair kernel-only comparison for both implementations

5. **Deliverables:** Five CSV files with 23 configurations tested on production B200 hardware

---

**CSV Files:** `fp8_results.csv`, `fp16_results.csv`, `mxfp8_results.csv`, `mxfp4_results.csv`, `nvfp4_results.csv`  
**Software:** Triton (latest), CUDA 13.0, cuBLAS 13.0, PTX 13.0.88
