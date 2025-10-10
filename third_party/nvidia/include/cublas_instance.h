#ifndef TRITON_CUBLAS_INSTANCE_H
#define TRITON_CUBLAS_INSTANCE_H

#include "cublas_types.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cstring>

class CublasLtInstance {
  // Typedefs for cublas functions
  typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *);
  typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
  typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t *,
                                                       cublasComputeType_t,
                                                       cudaDataType_t);
  typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
  typedef cublasStatus_t (*cublasLtMatmulDescSetAttribute_t)(
      cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *,
      size_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(
      cublasLtMatrixLayout_t *, cudaDataType_t, uint64_t, uint64_t, int64_t);
  typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(
      cublasLtMatrixLayout_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(
      cublasLtMatmulPreference_t *);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(
      cublasLtMatmulPreference_t);
  typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(
      cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
      const void *, size_t);
  typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
      cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
      cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t *,
      int *);
  typedef cublasStatus_t (*cublasLtMatmul_t)(
      cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *,
      const cublasLtMatrixLayout_t, const void *, const cublasLtMatrixLayout_t,
      const void *, const void *, const cublasLtMatrixLayout_t, void *,
      const cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *,
      size_t, cudaStream_t);

  // CUDA runtime typedefs for timing
  typedef int (*cudaStreamCreate_t)(cudaStream_t *);
  typedef int (*cudaStreamDestroy_t)(cudaStream_t);
  typedef int (*cudaEventCreate_t)(cudaEvent_t *);
  typedef int (*cudaEventDestroy_t)(cudaEvent_t);
  typedef int (*cudaEventRecord_t)(cudaEvent_t, cudaStream_t);
  typedef int (*cudaEventSynchronize_t)(cudaEvent_t);
  typedef int (*cudaEventElapsedTime_t)(float *, cudaEvent_t, cudaEvent_t);

  static constexpr const char *name = "libcublas.so";
  static constexpr const char *cuda_name = "libcudart.so";

  cublasLtCreate_t cublasLtCreate;
  cublasLtDestroy_t cublasLtDestroy;
  cublasLtMatmulDescCreate_t cublasLtMatmulDescCreate;
  cublasLtMatmulDescDestroy_t cublasLtMatmulDescDestroy;
  cublasLtMatmulDescSetAttribute_t cublasLtMatmulDescSetAttribute;
  cublasLtMatrixLayoutCreate_t cublasLtMatrixLayoutCreate;
  cublasLtMatrixLayoutDestroy_t cublasLtMatrixLayoutDestroy;
  cublasLtMatmulPreferenceCreate_t cublasLtMatmulPreferenceCreate;
  cublasLtMatmulPreferenceDestroy_t cublasLtMatmulPreferenceDestroy;
  cublasLtMatmulPreferenceSetAttribute_t cublasLtMatmulPreferenceSetAttribute;
  cublasLtMatmulAlgoGetHeuristic_t cublasLtMatmulAlgoGetHeuristic;
  cublasLtMatmul_t cublasLtMatmul;

  // CUDA runtime functions
  cudaStreamCreate_t cudaStreamCreate;
  cudaStreamDestroy_t cudaStreamDestroy;
  cudaEventCreate_t cudaEventCreate;
  cudaEventDestroy_t cudaEventDestroy;
  cudaEventRecord_t cudaEventRecord;
  cudaEventSynchronize_t cudaEventSynchronize;
  cudaEventElapsedTime_t cudaEventElapsedTime;

  void *dylibHandle = nullptr;
  void *cudaDylibHandle = nullptr;
  cublasLtHandle_t ltHandle;

  void *workspace = nullptr;
  size_t workspaceSize = 0;

  cublasLtMatmulPreference_t preference = NULL;

  // Cache for autotuned algorithms: (m, n, k, dtype) -> algo
  using AlgoKey = std::tuple<int, int, int, cudaDataType_t>;
  std::map<AlgoKey, cublasLtMatmulAlgo_t> algoCache;
  
  // Cache for block-scaled matmul: (m, n, k, dtype, use_1d_scaling) -> algo
  using BlockScaledAlgoKey = std::tuple<int, int, int, cudaDataType_t, bool>;
  std::map<BlockScaledAlgoKey, cublasLtMatmulAlgo_t> blockScaledAlgoCache;

  void loadCublasDylib() {
    if (dylibHandle == nullptr) {
      // First reuse the existing handle
      dylibHandle = dlopen(name, RTLD_NOLOAD);
    }
    if (dylibHandle == nullptr) {
      // If not found, try to load it
      dylibHandle = dlopen(name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (dylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(name) +
                               "`. Make sure it is in your "
                               "LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    cublasLtCreate = (cublasLtCreate_t)dlsym(dylibHandle, "cublasLtCreate");
    cublasLtDestroy = (cublasLtDestroy_t)dlsym(dylibHandle, "cublasLtDestroy");
    cublasLtMatmulDescCreate = (cublasLtMatmulDescCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulDescCreate");
    cublasLtMatmulDescDestroy = (cublasLtMatmulDescDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulDescDestroy");
    cublasLtMatmulDescSetAttribute = (cublasLtMatmulDescSetAttribute_t)dlsym(
        dylibHandle, "cublasLtMatmulDescSetAttribute");
    cublasLtMatrixLayoutCreate = (cublasLtMatrixLayoutCreate_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutCreate");
    cublasLtMatrixLayoutDestroy = (cublasLtMatrixLayoutDestroy_t)dlsym(
        dylibHandle, "cublasLtMatrixLayoutDestroy");
    cublasLtMatmulPreferenceCreate = (cublasLtMatmulPreferenceCreate_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceCreate");
    cublasLtMatmulPreferenceDestroy = (cublasLtMatmulPreferenceDestroy_t)dlsym(
        dylibHandle, "cublasLtMatmulPreferenceDestroy");
    cublasLtMatmulPreferenceSetAttribute =
        (cublasLtMatmulPreferenceSetAttribute_t)dlsym(
            dylibHandle, "cublasLtMatmulPreferenceSetAttribute");
    cublasLtMatmulAlgoGetHeuristic = (cublasLtMatmulAlgoGetHeuristic_t)dlsym(
        dylibHandle, "cublasLtMatmulAlgoGetHeuristic");
    cublasLtMatmul = (cublasLtMatmul_t)dlsym(dylibHandle, "cublasLtMatmul");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadCublasDylib() { 
    if (dylibHandle) dlclose(dylibHandle); 
  }

  void loadCudaDylib() {
    if (cudaDylibHandle == nullptr) {
      cudaDylibHandle = dlopen(cuda_name, RTLD_NOLOAD);
    }
    if (cudaDylibHandle == nullptr) {
      cudaDylibHandle = dlopen(cuda_name, RTLD_LOCAL | RTLD_LAZY);
    }
    if (cudaDylibHandle == nullptr) {
      throw std::runtime_error("Could not find `" + std::string(cuda_name) +
                               "`. Make sure it is in your LD_LIBRARY_PATH.");
    }
    dlerror(); // Clear any existing error

    cudaStreamCreate = (cudaStreamCreate_t)dlsym(cudaDylibHandle, "cudaStreamCreate");
    cudaStreamDestroy = (cudaStreamDestroy_t)dlsym(cudaDylibHandle, "cudaStreamDestroy");
    cudaEventCreate = (cudaEventCreate_t)dlsym(cudaDylibHandle, "cudaEventCreate");
    cudaEventDestroy = (cudaEventDestroy_t)dlsym(cudaDylibHandle, "cudaEventDestroy");
    cudaEventRecord = (cudaEventRecord_t)dlsym(cudaDylibHandle, "cudaEventRecord");
    cudaEventSynchronize = (cudaEventSynchronize_t)dlsym(cudaDylibHandle, "cudaEventSynchronize");
    cudaEventElapsedTime = (cudaEventElapsedTime_t)dlsym(cudaDylibHandle, "cudaEventElapsedTime");

    const char *dlsym_error = dlerror();
    if (dlsym_error) {
      throw std::runtime_error("Could not load symbol from `" +
                               std::string(cuda_name) +
                               "`: " + std::string(dlsym_error));
    }
  }

  void unloadCudaDylib() { 
    if (cudaDylibHandle) dlclose(cudaDylibHandle); 
  }

  float median(std::vector<float>& times) {
    const size_t size = times.size();
    if (size == 0) return 0;
    std::sort(times.begin(), times.end());
    const size_t mid = size / 2;
    if (size % 2 == 0) {
      return (times[mid] + times[mid - 1]) / 2.0f;
    } else {
      return times[mid];
    }
  }

  void successOrExit(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS Error: " + std::to_string(status) +
                               "\n");
    }
  }

  // Autotune and return the best algorithm for the given configuration
  cublasLtMatmulAlgo_t autotune_algorithm(
      int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
      cudaDataType_t dtype, cublasLtMatmulDesc_t matmulDesc,
      cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
      cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
      float alpha, float beta) {
    
    constexpr int requestedAlgoCount = 8;
    constexpr int repeatAlgoCheck = 5;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = {0};

    // Get heuristic algorithms
    successOrExit(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
        requestedAlgoCount, heuristicResult, &returnedResults));

    if (returnedResults == 0) {
      throw std::runtime_error(
          "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic");
    }

    // Benchmark each algorithm
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    cudaStreamCreate(&stream);
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    std::vector<float> algoTimes(repeatAlgoCheck);
    int bestAlgoIdx = 0;
    float bestAlgoTime = 0;

    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
      for (int checkIdx = 0; checkIdx < repeatAlgoCheck; checkIdx++) {
        cudaEventRecord(startEvent, stream);

        successOrExit(cublasLtMatmul(
            ltHandle, matmulDesc, &alpha, (void *)A, Adesc, (void *)B, Bdesc,
            &beta, (void *)C, Cdesc, (void *)D, Ddesc,
            &heuristicResult[algoIdx].algo, (void *)workspace, workspaceSize,
            stream));

        cudaEventRecord(stopEvent, stream);
        cudaEventSynchronize(stopEvent);
        float time;
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        algoTimes[checkIdx] = time;
      }

      float time = median(algoTimes);
      if (algoIdx == 0 || time < bestAlgoTime) {
        bestAlgoTime = time;
        bestAlgoIdx = algoIdx;
      }
    }

    cublasLtMatmulAlgo_t bestAlgo;
    memcpy(&bestAlgo, &heuristicResult[bestAlgoIdx].algo, sizeof(bestAlgo));

    cudaStreamDestroy(stream);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return bestAlgo;
  }

  // Simple wrapper around the cublasLtMatmul function with autotuning
  void gemm_impl(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                 uint64_t D, cudaDataType_t dtype, float alpha, float beta) {
    cublasLtMatmulDesc_t matmulDesc = NULL;

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL,
                           Ddesc = NULL;

    // Select compute type. Use TF32 when inputs are FP32, otherwise default
    // FP32 accumulation.
    cublasComputeType_t computeType = (dtype == CUDA_R_32F)
                                          ? CUBLAS_COMPUTE_32F_FAST_TF32
                                          : CUBLAS_COMPUTE_32F;
    successOrExit(
        cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    if (dtype == CUDA_R_8F_E4M3) {
      successOrExit(cublasLtMatmulDescSetAttribute(
          matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum,
          sizeof(fastAccum)));
    }

    auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));
    successOrExit(cublasLtMatrixLayoutCreate(&Ddesc, dtype, m, n, m));

    // Check if we have a cached algorithm for this configuration
    AlgoKey key = std::make_tuple(m, n, k, dtype);
    cublasLtMatmulAlgo_t algo;
    
    if (algoCache.find(key) == algoCache.end()) {
      // Autotune and cache the best algorithm
      algo = autotune_algorithm(m, n, k, A, B, C, D, dtype, matmulDesc,
                                Adesc, Bdesc, Cdesc, Ddesc, alpha, beta);
      algoCache[key] = algo;
    } else {
      // Use cached algorithm
      algo = algoCache[key];
    }

    // Execute with the best algorithm
    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)A, Adesc,
                                 (void *)B, Bdesc, &beta, (void *)C, Cdesc,
                                 (void *)D, Ddesc, &algo,
                                 (void *)workspace, workspaceSize, 0));
    
    if (Ddesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }

public:
  CublasLtInstance(uint64_t workspace, size_t workspaceSize)
      : workspace((void *)workspace), workspaceSize(workspaceSize) {
    loadCublasDylib();
    loadCudaDylib();
    cublasLtCreate(&ltHandle);

    successOrExit(cublasLtMatmulPreferenceCreate(&preference));
    successOrExit(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize,
        sizeof(workspaceSize)));
  }
  ~CublasLtInstance() {
    if (preference)
      successOrExit(cublasLtMatmulPreferenceDestroy(preference));

    cublasLtDestroy(ltHandle);
    unloadCublasDylib();
    unloadCudaDylib();
  }

  // C = A * B
  // Matrix B needs to be transposed, while matrix A does not. The function
  // *will-not* transpose the matrices, so the caller is responsible for
  // ensuring that the matrices are in the correct format and have the correct
  // dimensions.
  void matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
              cudaDataType_t dtype) {
    // CUDA is column-major, while triton is row-major, therefore we need to
    // reverse the order of the matrices ( A * B = (B^T * A^T)^T ).
    gemm_impl(n, m, k, B, A, 0, C, dtype, 1.0f, 0.0f);
  }

  void gemm(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C, uint64_t D,
            cudaDataType_t dtype, float alpha, float beta) {
    gemm_impl(n, m, k, B, A, C, D, dtype, alpha, beta);
  }

  // ===========================================================================
  // ⚠️  TODO: VERIFY cuBLAS FP4 SUPPORT STATUS
  // ===========================================================================
  //
  // As of October 2025, cuBLAS 13.0 does NOT support FP4 data types 
  // (CUDA_R_4F_E2M1 for MXFP4/NVFP4) for matmul operations per NVIDIA docs:
  //   - https://docs.nvidia.com/cuda/cublas/index.html
  //   - FP4 types exist in CUDA only for storage/movement, not computation
  //
  // Current behavior: 
  //   - When dtype=CUDA_R_4F_E2M1, cuBLASLt returns CUBLAS_STATUS_NOT_SUPPORTED (15)
  //   - Benchmarks correctly report 0.0 TFLOPS for cuBLAS FP4 operations
  //   - Only Triton kernels can execute FP4 matmul (Triton-exclusive feature)
  //
  // ACTION REQUIRED:
  //   1. Verify with NVIDIA if FP4 support is planned for future cuBLAS
  //   2. Check release notes for cuBLAS versions > 13.0 for FP4 additions
  //   3. Test on production B200 hardware to confirm (may differ from pre-prod)
  //   4. If permanently unsupported: FP4 is a Triton differentiator
  //
  // ===========================================================================

  // Block-scaled matmul: C = (A * scale_A) @ (B * scale_B)
  // A and B are in packed FP4 (2 elements/byte) or FP8 format
  // scale_A and scale_B are FP32 scale factors in block layout
  // Matrix B needs to be transposed (col-major for FP4, either for FP8)
  void block_scaled_matmul(int m, int n, int k, uint64_t A, uint64_t B, uint64_t C,
                           uint64_t scale_A, uint64_t scale_B,
                           cudaDataType_t dtype, bool use_1d_scaling) {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    int8_t fastAccum = 1;

    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    
    // Use FP32 compute and accumulation
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    successOrExit(cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Enable fast accumulation for FP8/FP4
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum)));

    // Set scale pointers for block scaling
    void* scale_A_ptr = (void*)scale_A;
    void* scale_B_ptr = (void*)scale_B;
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &scale_A_ptr, sizeof(scale_A_ptr)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &scale_B_ptr, sizeof(scale_B_ptr)));

    // Set scale matrix types (1D or 2D block scaling)
    cublasLtMatmulMatrixScale_t scale_type = use_1d_scaling
        ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
        : CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MATRIX_TYPE,
        &scale_type, sizeof(scale_type)));
    successOrExit(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MATRIX_TYPE,
        &scale_type, sizeof(scale_type)));

    // Create matrix layouts - output is always FP16
    cudaDataType_t c_dtype = CUDA_R_16F;
    successOrExit(cublasLtMatrixLayoutCreate(&Adesc, dtype, k, m, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Bdesc, dtype, k, n, k));
    successOrExit(cublasLtMatrixLayoutCreate(&Cdesc, c_dtype, m, n, m));

    // Check if we have a cached algorithm for this configuration
    BlockScaledAlgoKey key = std::make_tuple(m, n, k, dtype, use_1d_scaling);
    cublasLtMatmulAlgo_t algo;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    if (blockScaledAlgoCache.find(key) == blockScaledAlgoCache.end()) {
      // Autotune and cache the best algorithm
      constexpr int requestedAlgoCount = 8;
      constexpr int repeatAlgoCheck = 5;
      int returnedResults = 0;
      cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = {0};

      // Get heuristic algorithms
      successOrExit(cublasLtMatmulAlgoGetHeuristic(
          ltHandle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, preference,
          requestedAlgoCount, heuristicResult, &returnedResults));
      if (returnedResults == 0) {
        throw std::runtime_error(
            "No valid algorithm found by cublasLtMatmulAlgoGetHeuristic for block-scaled matmul");
      }

      // Benchmark each algorithm and pick the best
      cudaStream_t stream;
      cudaEvent_t startEvent, stopEvent;
      cudaStreamCreate(&stream);
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);

      std::vector<float> algoTimes(repeatAlgoCheck);
      int bestAlgoIdx = 0;
      float bestAlgoTime = 0;

      for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
        for (int checkIdx = 0; checkIdx < repeatAlgoCheck; checkIdx++) {
          cudaEventRecord(startEvent, stream);

          successOrExit(cublasLtMatmul(
              ltHandle, matmulDesc, &alpha, (void *)B, Bdesc, (void *)A, Adesc,
              &beta, (void *)C, Cdesc, (void *)C, Cdesc,
              &heuristicResult[algoIdx].algo, (void *)workspace, workspaceSize,
              stream));

          cudaEventRecord(stopEvent, stream);
          cudaEventSynchronize(stopEvent);
          float time;
          cudaEventElapsedTime(&time, startEvent, stopEvent);
          algoTimes[checkIdx] = time;
        }

        float time = median(algoTimes);
        if (algoIdx == 0 || time < bestAlgoTime) {
          bestAlgoTime = time;
          bestAlgoIdx = algoIdx;
        }
      }

      memcpy(&algo, &heuristicResult[bestAlgoIdx].algo, sizeof(algo));
      blockScaledAlgoCache[key] = algo;

      cudaStreamDestroy(stream);
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
    } else {
      // Use cached algorithm
      algo = blockScaledAlgoCache[key];
    }

    // Execute with the best algorithm
    successOrExit(cublasLtMatmul(ltHandle, matmulDesc, &alpha, (void *)B, Bdesc,
                                 (void *)A, Adesc, &beta, (void *)C, Cdesc,
                                 (void *)C, Cdesc, &algo,
                                 (void *)workspace, workspaceSize, 0));

    if (Cdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
      successOrExit(cublasLtMatrixLayoutDestroy(Adesc));
    if (matmulDesc)
      successOrExit(cublasLtMatmulDescDestroy(matmulDesc));
  }
};

#endif // TRITON_CUBLAS_INSTANCE_H
