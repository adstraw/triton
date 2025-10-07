import os

import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import (
    tcgen05_mma,
    tcgen05_scaled_mma,
    allocate_tensor_memory,
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
)

from triton.profiler import proton

LOAD_X = gl.constexpr(False)
LOAD_W = gl.constexpr(True)
DO_MMA = gl.constexpr(os.environ.get("DO_MMA", "0") == "1")

USE_PROTON = os.environ.get("USE_PROTON", "0") == "1"
"""
USE_PROTON=1 python gluon_load_mma_test.py && proton-viewer test.hatchet -m time/ms,tbyte/s

DO_MMA = 0
├─ 32.275 6.654 0.839 test_fp8_32x128x128 [E_32(M) = 32]
├─ 34.616 6.204 0.782 test_fp8_32x256x64 [E_32(M) = 32]
├─ 33.429 6.424 0.810 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
├─ 33.493 6.412 0.809 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
├─ 33.119 6.484 0.818 test_fp8_32x256x64_swiz_trans_flat_128 [E_32(M) = 32]
├─ 17.797 6.033 0.761 test_mxfp4_32x256x128 [E_32(M) = 32]
├─ 17.114 6.274 0.791 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
├─ 17.098 6.280 0.792 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]
└─ 16.987 6.321 0.797 test_mxfp4_32x256x128_swiz_trans_flat_128 [E_32(M) = 32]

DO_MMA = 1
(3.30% slower) ├─ 33.340 6.441 test_fp8_32x128x128 [E_32(M) = 32]
(4.88% slower) ├─ 36.305 5.915 test_fp8_32x256x64 [E_32(M) = 32]
(3.32% slower) ├─ 34.540 6.217 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
(3.28% slower) ├─ 34.591 6.208 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
(7.89% slower) ├─ 19.201 5.592 test_mxfp4_32x256x128 [E_32(M) = 32]
(6.50% slower) ├─ 18.227 5.891 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
(6.59% slower) └─ 18.225 5.892 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]

--------------------------------

instead, double K for the mxfp4 tests so the inner loop iterations are the same

DO_MMA = 0
├─ 32.409 6.626 test_fp8_32x128x128 [E_32(M) = 32]
├─ 34.684 6.192 test_fp8_32x256x64 [E_32(M) = 32]
├─ 33.617 6.388 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
├─ 33.585 6.394 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
├─ 33.236 6.461 test_fp8_32x256x64_swiz_trans_flat_128 [E_32(M) = 32]
├─ 34.777 6.175 test_mxfp4_32x256x128 [E_32(M) = 32]
├─ 33.583 6.395 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
├─ 33.638 6.384 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]
└─ 33.382 6.433 test_mxfp4_32x256x128_swiz_trans_flat_128 [E_32(M) = 32]

DO_MMA = 1
(3.18%  slower) ├─ 33.440 6.422 test_fp8_32x128x128 [E_32(M) = 32]
(4.57%  slower) ├─ 36.269 5.921 test_fp8_32x256x64 [E_32(M) = 32]
(3.16%  slower) ├─ 34.679 6.192 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
(3.22%  slower) ├─ 34.668 6.194 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
(10.10% slower) ├─ 38.288 5.609 test_mxfp4_32x256x128 [E_32(M) = 32]
(6.85%  slower) ├─ 35.884 5.984 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
(6.56%  slower) └─ 35.846 5.991 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]

--------------------------------

using CU_TENSOR_MAP_L2_PROMOTION_L2_256B

DO_MMA = 0
├─ 32.145 6.681 test_fp8_32x128x128 [E_32(M) = 32]
├─ 32.597 6.588 test_fp8_32x256x64 [E_32(M) = 32]
├─ 33.058 6.496 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
├─ 33.071 6.493 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
├─ 32.960 6.515 test_fp8_32x256x64_swiz_trans_flat_128 [E_32(M) = 32]
├─ 17.007 6.313 test_mxfp4_32x256x128 [E_32(M) = 32]
├─ 16.922 6.345 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
├─ 16.941 6.338 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]
└─ 16.809 6.388 test_mxfp4_32x256x128_swiz_trans_flat_128 [E_32(M) = 32]

DO_MMA = 1
(2.29% slower) ├─ 32.880 6.531 test_fp8_32x128x128 [E_32(M) = 32]
(2.68% slower) ├─ 33.472 6.416 test_fp8_32x256x64 [E_32(M) = 32]
(2.57% slower) ├─ 33.908 6.333 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
(2.70% slower) ├─ 33.963 6.323 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
(4.29% slower) ├─ 17.737 6.054 test_mxfp4_32x256x128 [E_32(M) = 32]
(4.27% slower) ├─ 17.645 6.085 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
(3.68% slower) └─ 17.564 6.113 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]


--------------------------------

using CU_TENSOR_MAP_L2_PROMOTION_L2_256B and doubling K for mxfp4 tests

DO_MMA = 0
├─ 32.152 6.679 test_fp8_32x128x128 [E_32(M) = 32]
├─ 32.607 6.586 test_fp8_32x256x64 [E_32(M) = 32]
├─ 33.068 6.494 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
├─ 33.059 6.496 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
├─ 32.991 6.509 test_fp8_32x256x64_swiz_trans_flat_128 [E_32(M) = 32]
├─ 32.675 6.572 test_mxfp4_32x256x128 [E_32(M) = 32]
├─ 33.055 6.497 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
├─ 33.077 6.492 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]
└─ 32.861 6.535 test_mxfp4_32x256x128_swiz_trans_flat_128 [E_32(M) = 32]

DO_MMA = 1
(2.28% slower) ├─ 32.886 6.530 test_fp8_32x128x128 [E_32(M) = 32]
(2.66% slower) ├─ 33.473 6.416 test_fp8_32x256x64 [E_32(M) = 32]
(2.51% slower) ├─ 33.898 6.335 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
(2.66% slower) ├─ 33.937 6.328 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
(4.74% slower) ├─ 34.223 6.275 test_mxfp4_32x256x128 [E_32(M) = 32]
(4.42% slower) ├─ 34.515 6.222 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
(4.24% slower) └─ 34.480 6.228 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]
"""


"""
DO_MMA=1 ncu --section MemoryWorkloadAnalysis -k "regex:.*test_.*" python gluon_load_mma_test.py
  test_fp8_32x128x128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.32
    Mem Busy                               %        46.30
    Max Bandwidth                          %        77.27
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3241
    L2 Hit Rate                            %        24.84
    Mem Pipes Busy                         %        19.75
    ---------------------------- ----------- ------------

  test_fp8_32x256x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.88
    Mem Busy                               %        50.35
    Max Bandwidth                          %        71.91
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3444
    L2 Hit Rate                            %        36.60
    Mem Pipes Busy                         %        18.49
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.03
    Mem Busy                               %        51.68
    Max Bandwidth                          %        73.75
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3187
    L2 Hit Rate                            %        36.61
    Mem Pipes Busy                         %        18.84
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_trans_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.96
    Mem Busy                               %        50.94
    Max Bandwidth                          %        72.84
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3133
    L2 Hit Rate                            %        36.65
    Mem Pipes Busy                         %        18.77
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.13
    Mem Busy                               %        43.65
    Max Bandwidth                          %        62.66
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2995
    L2 Hit Rate                            %        36.55
    Mem Pipes Busy                         %        35.40
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.65
    Mem Busy                               %        48.06
    Max Bandwidth                          %        69.05
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2396
    L2 Hit Rate                            %        36.57
    Mem Pipes Busy                         %        38.57
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_trans_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.66
    Mem Busy                               %        48.17
    Max Bandwidth                          %        69.15
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector        28708
    L2 Hit Rate                            %        36.60
    Mem Pipes Busy                         %        38.64
    ---------------------------- ----------- ------------
"""

"""
DO_MMA=0 ncu --section MemoryWorkloadAnalysis -k "regex:.*test_.*" python gluon_load_mma_test.py

  test_fp8_32x128x128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.60
    Mem Busy                               %        48.37
    Max Bandwidth                          %        80.67
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3096
    L2 Hit Rate                            %        24.85
    Mem Pipes Busy                         %         2.26
    ---------------------------- ----------- ------------

  test_fp8_32x256x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.40
    Mem Busy                               %        54.87
    Max Bandwidth                          %        78.25
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3041
    L2 Hit Rate                            %        36.59
    Mem Pipes Busy                         %         2.12
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.39
    Mem Busy                               %        54.73
    Max Bandwidth                          %        78.14
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2995
    L2 Hit Rate                            %        36.61
    Mem Pipes Busy                         %         2.12
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_trans_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.33
    Mem Busy                               %        54.13
    Max Bandwidth                          %        77.41
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2995
    L2 Hit Rate                            %        36.64
    Mem Pipes Busy                         %         2.11
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_trans_flat_128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.44
    Mem Busy                               %        47.13
    Max Bandwidth                          %        78.68
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3413
    L2 Hit Rate                            %        24.84
    Mem Pipes Busy                         %         2.12
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.14
    Mem Busy                               %        52.30
    Max Bandwidth                          %        75.11
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2391
    L2 Hit Rate                            %        36.56
    Mem Pipes Busy                         %         2.21
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.35
    Mem Busy                               %        54.08
    Max Bandwidth                          %        77.58
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         1655
    L2 Hit Rate                            %        36.57
    Mem Pipes Busy                         %         2.28
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_trans_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.33
    Mem Busy                               %        53.86
    Max Bandwidth                          %        77.40
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         1771
    L2 Hit Rate                            %        36.57
    Mem Pipes Busy                         %         2.30
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_trans_flat_128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.46
    Mem Busy                               %        54.92
    Max Bandwidth                          %        78.94
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         1583
    L2 Hit Rate                            %        36.58
    Mem Pipes Busy                         %         2.32
    ---------------------------- ----------- ------------
"""

@triton.constexpr_function
def get_mma_instr_shape(shape, element_ty):
    m = 128 if shape[0] >= 128 else 64
    n = 256 if shape[1] >= 256 else shape[1]
    k = 256 // element_ty.primitive_bitwidth
    return (m, n, k)


@triton.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: tl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    tl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n

@triton.jit
def load_tile_attrs(pid, grid_m, grid_n, GROUP_M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TOKENS_PER_EXPT: gl.constexpr):
    BLOCKS_PER_EXPT = (TOKENS_PER_EXPT + BLOCK_M - 1) // BLOCK_M
    pid_m, pid_n = swizzle2d(pid, grid_m, grid_n, GROUP_M)
    expt_id = pid_m // BLOCKS_PER_EXPT
    block_id = pid_m % BLOCKS_PER_EXPT
    off_m = expt_id * TOKENS_PER_EXPT + block_id * BLOCK_M
    off_n = pid_n * BLOCK_N

    return expt_id, off_m, off_n


@gluon.jit
def _load_partition(X, W, x_smem, w_smem, acc_mem, k_ready_bars, k_empty_bars, t_ready_bars, t_empty_bars, grid_m, grid_n, K, NUM_STAGES: gl.constexpr, NUM_ACC: gl.constexpr, GROUP_M: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, W_PACK: gl.constexpr, W_SWIZZLE: gl.constexpr, TOKENS_PER_EXPT: gl.constexpr, SWAP_XW: gl.constexpr, NUM_SMS: gl.constexpr):
    k_s = 0
    k_p = 0

    t_s = 0
    t_p = 0

    num_tiles = grid_m * grid_n
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    PACKED_BLOCK_K: gl.constexpr = BLOCK_K // W_PACK
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS):
        expt_id, off_m, off_n = load_tile_attrs(tile_id, grid_m, grid_n, GROUP_M, BLOCK_M, BLOCK_N, TOKENS_PER_EXPT)

        for ki in tl.range(k_tiles):
            mbarrier.wait(k_empty_bars.index(k_s), k_p)
            mbarrier.expect(
                k_ready_bars.index(k_s),
                0 + (BLOCK_M * BLOCK_K if LOAD_X else 0) + (BLOCK_N * PACKED_BLOCK_K if LOAD_W else 0))
            if LOAD_X:
                tma.async_copy_global_to_shared(X, [off_m, ki * BLOCK_K], k_ready_bars.index(k_s), x_smem.index(k_s))
            if LOAD_W:
                if W_SWIZZLE == "2x64":
                    tma.async_copy_global_to_shared(W, [expt_id, off_n // 2, ki * PACKED_BLOCK_K // 64, 0, 0], k_ready_bars.index(k_s), w_smem.index(k_s))
                elif W_SWIZZLE == "trans_2x64":
                    tma.async_copy_global_to_shared(W, [expt_id, ki * PACKED_BLOCK_K // 64, off_n // 2, 0, 0], k_ready_bars.index(k_s), w_smem.index(k_s))
                elif W_SWIZZLE == "trans_flat_128":
                    tma.async_copy_global_to_shared(W, [expt_id, ki * PACKED_BLOCK_K // 64, off_n // 2, 0], k_ready_bars.index(k_s), w_smem.index(k_s))
                else:
                    tma.async_copy_global_to_shared(W, [expt_id, off_n, ki * BLOCK_K], k_ready_bars.index(k_s), w_smem.index(k_s))

            k_s = k_s + 1
            k_p = tl.where(k_s == NUM_STAGES, k_p ^ 1, k_p)
            k_s = tl.where(k_s == NUM_STAGES, 0, k_s)

        t_s = t_s + 1
        t_p = tl.where(t_s == NUM_ACC, t_p ^ 1, t_p)
        t_s = tl.where(t_s == NUM_ACC, 0, t_s)


@gluon.jit
def _mma_partition(X, W, x_smem, w_smem, acc_mem, k_ready_bars, k_empty_bars, t_ready_bars, t_empty_bars, grid_m, grid_n, K, NUM_STAGES: gl.constexpr, NUM_ACC: gl.constexpr, GROUP_M: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, W_PACK: gl.constexpr, W_SWIZZLE: gl.constexpr, TOKENS_PER_EXPT: gl.constexpr, SWAP_XW: gl.constexpr, NUM_SMS: gl.constexpr):
    k_s = 0
    k_p = 0

    t_s = 0
    t_p = 0

    x_scales = allocate_tensor_memory(gl.uint8, [BLOCK_M, BLOCK_K // 32], TensorMemoryScalesLayout())
    w_scales = allocate_tensor_memory(gl.uint8, [BLOCK_N, BLOCK_K // 32], TensorMemoryScalesLayout())

    num_tiles = grid_m * grid_n
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS):
        mbarrier.wait(t_empty_bars.index(t_s), t_p)
        for ki in tl.range(k_tiles):
            mbarrier.wait(k_ready_bars.index(k_s), k_p)

            if DO_MMA:
                w_view = w_smem.index(k_s).reshape((BLOCK_N, BLOCK_K // W_PACK))

                if SWAP_XW:
                    if W_PACK == 2:
                        tcgen05_scaled_mma(
                            w_view, w_scales, "e2m1",
                            x_smem.index(k_s).permute((1, 0)), x_scales, "e4m3",
                            acc_mem.index(t_s), use_acc=ki != 0,
                            mbarriers=[k_empty_bars.index(k_s), t_ready_bars.index(t_s)],
                            mbarrier_preds=[True, ki == k_tiles - 1],
                        )
                    else:
                        tcgen05_mma(
                            w_view, x_smem.index(k_s).permute((1, 0)), acc_mem.index(t_s),
                            use_acc=ki != 0,
                            mbarriers=[k_empty_bars.index(k_s), t_ready_bars.index(t_s)],
                            mbarrier_preds=[True, ki == k_tiles - 1],
                        )
                else:
                    if W_PACK == 2:
                        tcgen05_scaled_mma(
                            x_smem.index(k_s), x_scales, "e4m3",
                            w_view.permute((1, 0)), w_scales, "e2m1",
                            acc_mem.index(t_s), use_acc=ki != 0,
                            mbarriers=[k_empty_bars.index(k_s), t_ready_bars.index(t_s)],
                            mbarrier_preds=[True, ki == k_tiles - 1],
                        )
                    else:
                        tcgen05_mma(
                            x_smem.index(k_s), w_view.permute((1, 0)), acc_mem.index(t_s),
                            use_acc=ki != 0,
                            mbarriers=[k_empty_bars.index(k_s), t_ready_bars.index(t_s)],
                            mbarrier_preds=[True, ki == k_tiles - 1],
                        )
            else:
                mbarrier.arrive(k_empty_bars.index(k_s), count=1)
                if ki == k_tiles - 1:
                    mbarrier.arrive(t_ready_bars.index(t_s), count=1)

            k_s = k_s + 1
            k_p = tl.where(k_s == NUM_STAGES, k_p ^ 1, k_p)
            k_s = tl.where(k_s == NUM_STAGES, 0, k_s)

        t_s = t_s + 1
        t_p = tl.where(t_s == NUM_ACC, t_p ^ 1, t_p)
        t_s = tl.where(t_s == NUM_ACC, 0, t_s)


@gluon.jit
def _epilogue_partition(W, O, w_smem, acc_mem, k_ready_bars, k_empty_bars, t_ready_bars, t_empty_bars, grid_m, grid_n, K, NUM_STAGES: gl.constexpr, NUM_ACC: gl.constexpr, GROUP_M: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, W_PACK: gl.constexpr, W_SWIZZLE: gl.constexpr, TOKENS_PER_EXPT: gl.constexpr, SWAP_XW: gl.constexpr, NUM_SMS: gl.constexpr, num_warps: gl.constexpr):
    t_s = 0
    t_p = 0

    num_tiles = grid_m * grid_n
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS):
        expt_id, off_m, off_n = load_tile_attrs(tile_id, grid_m, grid_n, GROUP_M, BLOCK_M, BLOCK_N, TOKENS_PER_EXPT)
        mbarrier.wait(t_ready_bars.index(t_s), t_p)
        mbarrier.arrive(t_empty_bars.index(t_s), count=1)

        t_s = t_s + 1
        t_p = tl.where(t_s == NUM_ACC, t_p ^ 1, t_p)
        t_s = tl.where(t_s == NUM_ACC, 0, t_s)


def _repr(specialization):
    if "u8" in specialization.signature["W"]:
        dtype = "mxfp4"
    elif "fp8" in specialization.signature["W"]:
        dtype = "fp8"
    name = f"test_{dtype}_{specialization.constants['BLOCK_M']}x{specialization.constants['BLOCK_N']}x{specialization.constants['BLOCK_K']}"
    if specialization.constants["W_SWIZZLE"]:
        name += "_swiz_" + specialization.constants["W_SWIZZLE"]
    return name


def _launch_metadata(grid, kernel, args):
    name = kernel.name + f" [E_{args['E']}(M) = {args['TOKENS_PER_EXPT']}]"
    bytes = 0
    if LOAD_X:
        bytes += args["M"] * args["K"] * args["X"].base.dtype.itemsize
    if LOAD_W:
        bytes += args["E"] * args["N"] * args["K"] // args["W_PACK"] * args["W"].base.dtype.itemsize
    return {"name": name, "bytes": bytes}


@gluon.jit(repr=_repr, launch_metadata=_launch_metadata)
def _test(X, W, O, E, M, N, K, grid_m, grid_n, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, W_PACK: gl.constexpr, W_SWIZZLE: gl.constexpr, TOKENS_PER_EXPT: gl.constexpr, SWAP_XW: gl.constexpr, NUM_SMS: gl.constexpr, num_warps: gl.constexpr):
    GROUP_M: gl.constexpr = 8
    NUM_STAGES: gl.constexpr = 4
    NUM_ACC: gl.constexpr = 1

    if W_SWIZZLE == "trans_flat_128":
        w_smem_shape: gl.constexpr = [NUM_STAGES, BLOCK_N // 2, 128]
    elif W_SWIZZLE == "trans_2x64":
        w_smem_shape: gl.constexpr = [NUM_STAGES, BLOCK_K // W_PACK // 64, BLOCK_N // 2, 2, 64]
    elif W_SWIZZLE == "2x64":
        w_smem_shape: gl.constexpr = [NUM_STAGES, BLOCK_N // 2, BLOCK_K // W_PACK // 64, 2, 64]
    else:
        w_smem_shape: gl.constexpr = [NUM_STAGES, BLOCK_N, BLOCK_K // W_PACK]

    w_smem_layout: gl.constexpr = gl.NVMMASharedLayout(
        swizzle_byte_width=W.type.layout.swizzle_byte_width,
        element_bitwidth=W.type.layout.element_bitwidth,
        rank=len(w_smem_shape) - 1,
        transposed=W.type.layout.transposed,
        fp4_padded=W.type.layout.fp4_padded,
    )
    w_smem = gl.allocate_shared_memory(W.dtype, w_smem_shape, w_smem_layout)
    x_smem = gl.allocate_shared_memory(X.dtype, [NUM_STAGES, BLOCK_M, BLOCK_K], X.type.layout)

    acc_shape: gl.constexpr = [BLOCK_N, BLOCK_M] if SWAP_XW else [BLOCK_M, BLOCK_N]
    acc_mma_shape: gl.constexpr = get_mma_instr_shape(acc_shape, gl.float32)
    acc_mem = allocate_tensor_memory(gl.float32, [NUM_ACC] + acc_shape, TensorMemoryLayout((acc_mma_shape[0], acc_mma_shape[1]), col_stride=1))

    # Barriers for tracking when data for the inner K loop is ready and when the MMA has consumed it.
    k_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    k_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for i in tl.static_range(NUM_STAGES):
        mbarrier.init(k_ready_bars.index(i), count=1)
        mbarrier.init(k_empty_bars.index(i), count=1)
        mbarrier.arrive(k_empty_bars.index(i), count=1)

    # Barriers for tracking when the MMA has produced an accumulator and when the epilogue has consumed it.
    t_ready_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC, 1], mbarrier.MBarrierLayout())
    t_empty_bars = gl.allocate_shared_memory(gl.int64, [NUM_ACC, 1], mbarrier.MBarrierLayout())
    for i in tl.static_range(NUM_ACC):
        mbarrier.init(t_ready_bars.index(i), count=1)
        mbarrier.init(t_empty_bars.index(i), count=1)
        mbarrier.arrive(t_empty_bars.index(i), count=1)

    gl.warp_specialize(
        (W, O, w_smem, acc_mem, k_ready_bars, k_empty_bars, t_ready_bars, t_empty_bars, grid_m, grid_n, K, NUM_STAGES, NUM_ACC, GROUP_M, BLOCK_M, BLOCK_N, BLOCK_K, W_PACK, W_SWIZZLE, TOKENS_PER_EXPT, SWAP_XW, NUM_SMS, num_warps),
        _epilogue_partition,
        (X, W, x_smem, w_smem, acc_mem, k_ready_bars, k_empty_bars, t_ready_bars, t_empty_bars, grid_m, grid_n, K, NUM_STAGES, NUM_ACC, GROUP_M, BLOCK_M, BLOCK_N, BLOCK_K, W_PACK, W_SWIZZLE, TOKENS_PER_EXPT, SWAP_XW, NUM_SMS),
        [_load_partition, _mma_partition], [1, 1], [24, 24]
    )


def apply_swizzle(w, N, PACKED_K, w_swizzle):
    align_to = lambda x, alignment: (x + alignment - 1) // alignment * alignment

    if w_swizzle:
        n_pad = align_to(N, 2) - N
        k_pad = align_to(PACKED_K, 128) - PACKED_K

        w = torch.nn.functional.pad(w, (0, k_pad, 0, n_pad))

        N = align_to(N, 2)
        PACKED_K = align_to(PACKED_K, 128)

        w = w.reshape(E, N // 2, 2, PACKED_K // 64, 64)
        if w_swizzle == "2x64":
            w = w.transpose(2, 3).flatten(-2, -1).reshape(E, N // 2, PACKED_K // 64, 2, 64)
        elif w_swizzle == "trans_2x64":
            w = w.transpose(2, 3).flatten(-2, -1).reshape(E, N // 2, PACKED_K // 64, 2, 64).transpose(1, 2)
        elif w_swizzle == "trans_flat_128":
            w = w.transpose(2, 3).flatten(-2, -1).transpose(1, 2)
        else:
            raise ValueError(f"Invalid w_swizzle: {w_swizzle}")

    return w

def test(E, TOKENS_PER_EXPT, N, K, BLOCK_M, BLOCK_N, BLOCK_K, w_dtype, w_swizzle=None):
    torch.manual_seed(42)
    # make the number of rows exactly num experts times block_m
    # in the kernel, we'll assume that expt_id == pid_m, so there are exactly block_m tokens per expert
    M = E * TOKENS_PER_EXPT

    grid_m = E * triton.cdiv(TOKENS_PER_EXPT, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    x = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    W_PACK = 2 if w_dtype == torch.uint8 else 1
    PACKED_K = K // W_PACK
    w = torch.zeros(E, N, PACKED_K, dtype=w_dtype, device="cuda")
    w = apply_swizzle(w, N, PACKED_K, w_swizzle)
    w_copy = torch.zeros_like(w)
    o = torch.empty(M, N // 2, dtype=torch.float8_e4m3fn, device="cuda")

    SWAP_XW = BLOCK_M <= 64

    x_desc = TensorDescriptor(
        x, shape=x.shape, strides=x.stride(),
        block_shape=[BLOCK_M, BLOCK_K],
        layout=gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float8e4nv),
    )
    if w_swizzle:
        if w_swizzle == "2x64":
            assert BLOCK_K // W_PACK // 64 == 1
            w_block_shape = [1, BLOCK_N // 2, BLOCK_K // W_PACK // 64, 2, 64]
        elif w_swizzle == "trans_2x64":
            assert BLOCK_K // W_PACK // 64 == 1
            w_block_shape = [1, BLOCK_K // W_PACK // 64, BLOCK_N // 2, 2, 64]
        elif w_swizzle == "trans_flat_128":
            assert BLOCK_K // W_PACK // 64 == 1
            w_block_shape = [1, BLOCK_K // W_PACK // 64, BLOCK_N // 2, 128]
        else:
            raise ValueError(f"Invalid w_swizzle: {w_swizzle}")
    else:
        w_block_shape = [1, BLOCK_N, BLOCK_K // W_PACK]

    w_desc = TensorDescriptor(
        w, shape=w.shape, strides=w.stride(),
        block_shape=w_block_shape,
        layout=gl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=8,
            rank=len(w_block_shape),
            transposed=False,
            fp4_padded=True,
         ) if w_dtype == torch.uint8 else gl.NVMMASharedLayout.get_default_for(w_block_shape, gl.float8e4nv)
    )

    o_desc = TensorDescriptor(
        o, shape=o.shape, strides=o.stride(),
        block_shape=[BLOCK_M, BLOCK_N // 2],
        layout=gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N // 2], gl.float8e4nv)
    )

    k = None
    for i in range(100 if USE_PROTON else 1):
        w.copy_(w_copy) # evict any l2 cache by filling W
        k = _test[(num_sms,)](
            x_desc, w_desc, o_desc,
            E=E, M=M, N=N, K=K,
            grid_m=grid_m, grid_n=grid_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            W_PACK=W_PACK,
            W_SWIZZLE=w_swizzle,
            TOKENS_PER_EXPT=TOKENS_PER_EXPT,
            SWAP_XW=SWAP_XW,
            NUM_SMS=num_sms,
            num_warps=8 if BLOCK_M == 128 else 4)
    with open(f"/tmp/{k.name}.ttgir", "w") as f:
        f.write(k.asm["ttgir"])
    with open(f"/tmp/{k.name}.ptx", "w") as f:
        f.write(k.asm["ptx"])


if __name__ == "__main__":
    if USE_PROTON:
        proton.start("test", hook="triton")

    E = 32
    TOKENS_PER_EXPT = 32
    N = 8192
    K = 8192

    # Baseline. Load 16kb blocks of W where we hit a full cacheline.
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, w_dtype=torch.float8_e4m3fn)

    # fp8. Load 16kb blocks of fp8 W. Test with both swizzled and not. Swizzled is a 2x64 byte region which is a full cacheline.
    # - trans version swaps the K and N dimensions which doesn't really matter because BLOCK_K // 64 == 1
    # - but trans is needed for flat_2x64 version which flattens the 2x64 region to a 1D 128-byte region
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, w_dtype=torch.float8_e4m3fn)
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, w_dtype=torch.float8_e4m3fn, w_swizzle="2x64")
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, w_dtype=torch.float8_e4m3fn, w_swizzle="trans_2x64")
    if not DO_MMA:
        test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, w_dtype=torch.float8_e4m3fn, w_swizzle="trans_flat_128")

    # mxfp4
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8)
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8, w_swizzle="2x64")
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8, w_swizzle="trans_2x64")
    if not DO_MMA:
        test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8, w_swizzle="trans_flat_128")

    if USE_PROTON:
        proton.finalize()