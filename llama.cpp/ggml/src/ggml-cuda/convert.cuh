#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

// C++14 replacement for the former if-constexpr type dispatch in ggml_cuda_cast:
// each destination type gets a helper with overloaded one-step conversions.
__host__ __device__ inline float ggml_cuda_cast_to_float(const float x) {
    return x;
}

__host__ __device__ inline float ggml_cuda_cast_to_float(const half x) {
    return __half2float(x);
}

__host__ __device__ inline float ggml_cuda_cast_to_float(const nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename dst_t>
struct ggml_cuda_cast_helper {
    // generic fallback: convert through fp32
    template <typename src_t>
    static __host__ __device__ inline dst_t cast(const src_t & x) {
        return dst_t(ggml_cuda_cast_to_float(x));
    }
};

template <>
struct ggml_cuda_cast_helper<float> {
    static __host__ __device__ inline float cast(const float x) { return x; }
    static __host__ __device__ inline float cast(const half x) { return ggml_cuda_cast_to_float(x); }
    static __host__ __device__ inline float cast(const nv_bfloat16 x) { return ggml_cuda_cast_to_float(x); }
    // Only reached from branches whose runtime type condition is false; kept compilable for C++14.
    static __host__ __device__ inline float cast(const half2 x) { return __low2float(x); }
    static __host__ __device__ inline float cast(const nv_bfloat162 x) { return __low2float(x); }
    template <typename src_t>
    static __host__ __device__ inline float cast(const src_t & x) { return float(x); }
};

template <>
struct ggml_cuda_cast_helper<half> {
    static __host__ __device__ inline half cast(const half x) { return x; }
    template <typename src_t>
    static __host__ __device__ inline half cast(const src_t & x) { return __float2half(ggml_cuda_cast_to_float(x)); }
};

template <>
struct ggml_cuda_cast_helper<nv_bfloat16> {
    static __host__ __device__ inline nv_bfloat16 cast(const nv_bfloat16 x) { return x; }
    template <typename src_t>
    static __host__ __device__ inline nv_bfloat16 cast(const src_t & x) { return __float2bfloat16(ggml_cuda_cast_to_float(x)); }
};

template <>
struct ggml_cuda_cast_helper<half2> {
    static __host__ __device__ inline half2 cast(const half2 x) { return x; }
    static __host__ __device__ inline half2 cast(const float2 x) { return __float22half2_rn(x); }
    // scalar broadcast, also keeps branches that are dead at runtime compilable for C++14
    static __host__ __device__ inline half2 cast(const float x) { return __floats2half2_rn(x, x); }
};

template <>
struct ggml_cuda_cast_helper<nv_bfloat162> {
    static __host__ __device__ inline nv_bfloat162 cast(const nv_bfloat162 x) { return x; }
    static __host__ __device__ inline nv_bfloat162 cast(const float2 x) { return __floats2bfloat162_rn(x.x, x.y); }
    static __host__ __device__ inline nv_bfloat162 cast(const float x) { return __floats2bfloat162_rn(x, x); }
};

template <>
struct ggml_cuda_cast_helper<float2> {
    static __host__ __device__ inline float2 cast(const float2 x) { return x; }
    static __host__ __device__ inline float2 cast(const half2 x) { return __half22float2(x); }
    static __host__ __device__ inline float2 cast(const nv_bfloat162 x) {
#if defined(GGML_USE_HIP)
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(GGML_CUDA_FORCE_BF16_SHIM)
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif
    }
};

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    return ggml_cuda_cast_helper<dst_t>::cast(x);
}

