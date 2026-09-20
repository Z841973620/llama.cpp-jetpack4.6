#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#if CUDART_VERSION >= 11000 && !defined(GGML_CUDA_FORCE_BF16_SHIM)
#include <cuda_bf16.h>
#else
// CUDA < 11.0 has no cuda_bf16.h: fall back to a software bf16 implementation.
// GGML_CUDA_FORCE_BF16_SHIM can be defined on newer toolkits to exercise the fallback.
#include "cuda-bf16-shim.cuh"
#endif
#include <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
