// RUN: triton-opt %s -split-input-file --triton-nvidia-check-matmul-two-cta -verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-ctas" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_invalid_num_ctas(%smem: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                        %tmem: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>) {
    // expected-error @+1 {{Only 1 or 2 CTAs supported for now}}
    ttng.tmem_copy %smem, %tmem : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_inconsistent_cta_modes(%arg0: !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>,
                                               %arg1: !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>,
                                               %arg2: !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>,
                                               %arg3: !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>,
                                               %arg4: !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>,
                                               %arg5: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {
    %true = arith.constant true
    // expected-note @+1 {{conflicts with previous tcgen05 op that uses 2 CTA mode}}
    ttng.tmem_copy %arg0, %arg1 : !ttg.memdesc<64x16xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<64x16xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // expected-error @+1 {{inconsistent CTA mode between tcgen05 operations; current tcgen05 op uses 1 CTA mode}}
    ttng.tc_gen5_mma %arg2, %arg3, %arg4, %true, %true, %arg5[%true] {is_async} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    ttng.tc_gen5_commit %arg5 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}
