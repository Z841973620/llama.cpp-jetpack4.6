#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same<dst_t, src_t>::value) {
        return x;
    } else if constexpr(std::is_same<src_t, float2>::value && std::is_same<dst_t, half2>::value) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same<dst_t, int32_t>::value) {
        return int32_t(x);
    } else {
        return float(x);
    }
}