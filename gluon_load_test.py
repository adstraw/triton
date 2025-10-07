import os

import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory,
    TensorMemoryLayout,
)

from triton.profiler import proton

LOAD_X = gl.constexpr(False)
LOAD_W = gl.constexpr(True)

USE_PROTON = os.environ.get("USE_PROTON", "0") == "1"

"""
ncu --section MemoryWorkloadAnalysis -k "regex:.*test_.*" python gluon_load_test.py

  test_fp8_32x128x128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.52
    Mem Busy                               %        47.78
    Max Bandwidth                          %        79.67
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2771
    L2 Hit Rate                            %         0.01
    Mem Pipes Busy                         %         2.21
    ---------------------------- ----------- ------------

  test_fp8_32x256x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.64
    Mem Busy                               %        41.40
    Max Bandwidth                          %        68.98
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2673
    L2 Hit Rate                            %        24.81
    Mem Pipes Busy                         %         1.86
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.08
    Mem Busy                               %        44.56
    Max Bandwidth                          %        74.33
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2663
    L2 Hit Rate                            %        24.82
    Mem Pipes Busy                         %         2.01
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_trans_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.06
    Mem Busy                               %        44.42
    Max Bandwidth                          %        74.00
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2700
    L2 Hit Rate                            %        24.83
    Mem Pipes Busy                         %         1.99
    ---------------------------- ----------- ------------

  test_fp8_32x256x64_swiz_trans_flat_128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.30
    Mem Busy                               %        46.12
    Max Bandwidth                          %        77.03
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         4374
    L2 Hit Rate                            %         0.02
    Mem Pipes Busy                         %         2.07
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.69
    Mem Busy                               %        41.59
    Max Bandwidth                          %        69.51
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2899
    L2 Hit Rate                            %        24.75
    Mem Pipes Busy                         %         2.03
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.01
    Mem Busy                               %        44.01
    Max Bandwidth                          %        73.42
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         2346
    L2 Hit Rate                            %        24.80
    Mem Pipes Busy                         %         2.14
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_trans_2x64 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         5.96
    Mem Busy                               %        43.64
    Max Bandwidth                          %        72.79
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         3897
    L2 Hit Rate                            %        24.81
    Mem Pipes Busy                         %         2.13
    ---------------------------- ----------- ------------

  test_mxfp4_32x256x128_swiz_trans_flat_128 (152, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Tbyte/s         6.12
    Mem Busy                               %        44.77
    Max Bandwidth                          %        74.83
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                   %            0
    L2 Compression Input Sectors      sector         1848
    L2 Hit Rate                            %        24.80
    Mem Pipes Busy                         %         2.19
    ---------------------------- ----------- ------------
"""

"""
USE_PROTON=1 python gluon_load_test.py && proton-viewer test.hatchet -m time/ms,tbyte/s

├─ 32.275 6.654 0.839 test_fp8_32x128x128 [E_32(M) = 32]
├─ 34.616 6.204 0.782 test_fp8_32x256x64 [E_32(M) = 32]
├─ 33.429 6.424 0.810 test_fp8_32x256x64_swiz_2x64 [E_32(M) = 32]
├─ 33.493 6.412 0.809 test_fp8_32x256x64_swiz_trans_2x64 [E_32(M) = 32]
├─ 33.119 6.484 0.818 test_fp8_32x256x64_swiz_trans_flat_128 [E_32(M) = 32]
├─ 17.797 6.033 0.761 test_mxfp4_32x256x128 [E_32(M) = 32]
├─ 17.114 6.274 0.791 test_mxfp4_32x256x128_swiz_2x64 [E_32(M) = 32]
├─ 17.098 6.280 0.792 test_mxfp4_32x256x128_swiz_trans_2x64 [E_32(M) = 32]
└─ 16.987 6.321 0.797 test_mxfp4_32x256x128_swiz_trans_flat_128 [E_32(M) = 32]
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

    num_tiles = grid_m * grid_n
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS):
        mbarrier.wait(t_empty_bars.index(t_s), t_p)
        for ki in tl.range(k_tiles):
            mbarrier.wait(k_ready_bars.index(k_s), k_p)

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
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, w_dtype=torch.float8_e4m3fn, w_swizzle="trans_flat_128")

    # mxfp4
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8)
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8, w_swizzle="2x64")
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8, w_swizzle="trans_2x64")
    test(E=E, TOKENS_PER_EXPT=TOKENS_PER_EXPT, N=N, K=K, BLOCK_M=32, BLOCK_N=256, BLOCK_K=128, w_dtype=torch.uint8, w_swizzle="trans_flat_128")

    if USE_PROTON:
        proton.finalize()
