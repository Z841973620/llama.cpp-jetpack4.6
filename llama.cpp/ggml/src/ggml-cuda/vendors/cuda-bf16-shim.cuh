// Software bf16 fallback for CUDA toolkits that predate cuda_bf16.h (CUDA < 11.0).
// Provides nv_bfloat16/nv_bfloat162 with the API surface used by the ggml CUDA backend.
// All arithmetic is performed in fp32 and rounded back to bf16, so the numerical
// behavior matches hardware bf16 that stores fp32 values rounded to 8 mantissa bits.
// The type layout is bit-identical to cuda_bf16.h (uint16_t), so tensors allocated
// elsewhere can be reinterpreted freely.
#pragma once

#include <cstdint>
#include <cstring>

struct __nv_bfloat16_raw {
    uint16_t x;
};

struct __nv_bfloat162_raw {
    __nv_bfloat16_raw x;
    __nv_bfloat16_raw y;
};

// Round-to-nearest-even fp32 -> bf16, NaN-preserving.
static __host__ __device__ __forceinline__ uint16_t ggml_cuda_fp32_to_bf16_bits(const float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        // NaN: keep it a quiet NaN after truncation
        return (uint16_t)((bits >> 16) | 0x0040u);
    }

    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding_bias) >> 16);
}

static __host__ __device__ __forceinline__ float ggml_cuda_bf16_bits_to_fp32(const uint16_t h) {
    uint32_t bits = (uint32_t) h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

struct __nv_bfloat16 {
    __nv_bfloat16_raw __x;

    __host__ __device__ __nv_bfloat16() = default;

    __host__ __device__ __nv_bfloat16(const __nv_bfloat16_raw & h) : __x(h) {}

    __host__ __device__ __nv_bfloat16(const __nv_bfloat16 & h) : __x(h.__x) {}

    __host__ __device__ __nv_bfloat16 & operator=(const __nv_bfloat16_raw & h) {
        __x = h;
        return *this;
    }

    __host__ __device__ operator __nv_bfloat16_raw() const { return __x; }

    __host__ __device__ __nv_bfloat16(const float f) : __x(__nv_bfloat16_raw{ggml_cuda_fp32_to_bf16_bits(f)}) {}

    __host__ __device__ __nv_bfloat16 & operator=(const float f) {
        __x.x = ggml_cuda_fp32_to_bf16_bits(f);
        return *this;
    }

    __host__ __device__ operator float() const { return ggml_cuda_bf16_bits_to_fp32(__x.x); }
};

typedef __nv_bfloat16 nv_bfloat16;

struct __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;

    __host__ __device__ __nv_bfloat162() = default;

    __host__ __device__ __nv_bfloat162(const __nv_bfloat162_raw & h) : x(h.x), y(h.y) {}

    __host__ __device__ __nv_bfloat162 & operator=(const __nv_bfloat162_raw & h) {
        x = h.x;
        y = h.y;
        return *this;
    }

    __host__ __device__ operator __nv_bfloat162_raw() const {
        __nv_bfloat162_raw h;
        h.x = x;
        h.y = y;
        return h;
    }

    __host__ __device__ __nv_bfloat162(const __nv_bfloat16 & x_, const __nv_bfloat16 & y_) : x(x_), y(y_) {}

    __host__ __device__ __nv_bfloat162(const float x_, const float y_) : x(x_), y(y_) {}
};

typedef __nv_bfloat162 nv_bfloat162;

// ---------------------------------------------------------------------------
// conversion intrinsics (names mirror cuda_bf16.h)
// ---------------------------------------------------------------------------

static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16(const float f) {
    nv_bfloat16 h;
    h.__x.x = ggml_cuda_fp32_to_bf16_bits(f);
    return h;
}

static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16_rn(const float f) {
    return __float2bfloat16(f);
}

static __host__ __device__ __forceinline__ float __bfloat162float(const nv_bfloat16 h) {
    return ggml_cuda_bf16_bits_to_fp32(h.__x.x);
}

static __host__ __device__ __forceinline__ nv_bfloat16 __bfloat16_as_ushort(const uint16_t v) {
    nv_bfloat16 h;
    h.__x.x = v;
    return h;
}

static __host__ __device__ __forceinline__ uint16_t __ushort_as_bfloat16(const nv_bfloat16 h) {
    return h.__x.x;
}

static __host__ __device__ __forceinline__ nv_bfloat16 __low2bfloat16(const nv_bfloat162 h) {
    return h.x;
}

static __host__ __device__ __forceinline__ nv_bfloat16 __high2bfloat16(const nv_bfloat162 h) {
    return h.y;
}

static __host__ __device__ __forceinline__ float __low2float(const nv_bfloat162 h) {
    return ggml_cuda_bf16_bits_to_fp32(h.x.__x.x);
}

static __host__ __device__ __forceinline__ float __high2float(const nv_bfloat162 h) {
    return ggml_cuda_bf16_bits_to_fp32(h.y.__x.x);
}

static __host__ __device__ __forceinline__ float2 __bfloat1622float2(const nv_bfloat162 h) {
    return make_float2(ggml_cuda_bf16_bits_to_fp32(h.x.__x.x), ggml_cuda_bf16_bits_to_fp32(h.y.__x.x));
}

static __host__ __device__ __forceinline__ nv_bfloat162 __floats2bfloat162_rn(const float x, const float y) {
    return nv_bfloat162(x, y);
}

static __host__ __device__ __forceinline__ nv_bfloat162 __float22bfloat162_rn(const float2 f) {
    return nv_bfloat162(f.x, f.y);
}

static __host__ __device__ __forceinline__ nv_bfloat162 __halves2bfloat162(const nv_bfloat16 x, const nv_bfloat16 y) {
    return nv_bfloat162(x, y);
}

static __host__ __device__ __forceinline__ nv_bfloat162 make_bfloat162(const nv_bfloat16 x, const nv_bfloat16 y) {
    return nv_bfloat162(x, y);
}

static __host__ __device__ __forceinline__ nv_bfloat162 __lowhigh2highlow(const nv_bfloat162 h) {
    return nv_bfloat162(h.y, h.x);
}

// ---------------------------------------------------------------------------
// arithmetic (computed in fp32, rounded back to bf16)
// ---------------------------------------------------------------------------

#define GGML_CUDA_BF16_BINOP(op)                                                                           \
    static __host__ __device__ __forceinline__ nv_bfloat16 operator op (const nv_bfloat16 a,                \
                                                                        const nv_bfloat16 b) {              \
        return __float2bfloat16(__bfloat162float(a) op __bfloat162float(b));                                \
    }                                                                                                       \
    static __host__ __device__ __forceinline__ nv_bfloat16 operator op (const nv_bfloat16 a, const float b) { \
        return __float2bfloat16(__bfloat162float(a) op b);                                                  \
    }                                                                                                       \
    static __host__ __device__ __forceinline__ nv_bfloat16 operator op (const float a, const nv_bfloat16 b) { \
        return __float2bfloat16(a op __bfloat162float(b));                                                  \
    }                                                                                                       \
    static __host__ __device__ __forceinline__ nv_bfloat162 operator op (const nv_bfloat162 a,              \
                                                                        const nv_bfloat162 b) {              \
        return __floats2bfloat162_rn(__low2float(a) op __low2float(b), __high2float(a) op __high2float(b)); \
    }

GGML_CUDA_BF16_BINOP(+)
GGML_CUDA_BF16_BINOP(-)
GGML_CUDA_BF16_BINOP(*)
GGML_CUDA_BF16_BINOP(/)

#undef GGML_CUDA_BF16_BINOP

#define GGML_CUDA_BF16_CMPOUND(op)                                                            \
    static __host__ __device__ __forceinline__ nv_bfloat16 & operator op (nv_bfloat16 & a,      \
                                                                           const nv_bfloat16 b) { \
        a = a op b;                                                                            \
        return a;                                                                              \
    }                                                                                          \
    static __host__ __device__ __forceinline__ nv_bfloat16 & operator op (nv_bfloat16 & a,      \
                                                                           const float b) {       \
        a = a op nv_bfloat16(b);                                                               \
        return a;                                                                              \
    }                                                                                          \
    static __host__ __device__ __forceinline__ nv_bfloat162 & operator op (nv_bfloat162 & a,    \
                                                                             const nv_bfloat162 b) { \
        a = a op b;                                                                            \
        return a;                                                                              \
    }

GGML_CUDA_BF16_CMPOUND(+=)
GGML_CUDA_BF16_CMPOUND(-=)
GGML_CUDA_BF16_CMPOUND(*=)
GGML_CUDA_BF16_CMPOUND(/=)

#undef GGML_CUDA_BF16_CMPOUND

#define GGML_CUDA_BF16_CMP(op)                                                                       \
    static __host__ __device__ __forceinline__ bool operator op (const nv_bfloat16 a, const nv_bfloat16 b) { \
        return __bfloat162float(a) op __bfloat162float(b);                                            \
    }                                                                                                 \
    static __host__ __device__ __forceinline__ bool operator op (const nv_bfloat16 a, const float b) {  \
        return __bfloat162float(a) op b;                                                              \
    }                                                                                                 \
    static __host__ __device__ __forceinline__ bool operator op (const float a, const nv_bfloat16 b) {  \
        return a op __bfloat162float(b);                                                              \
    }

GGML_CUDA_BF16_CMP(==)
GGML_CUDA_BF16_CMP(!=)
GGML_CUDA_BF16_CMP(<)
GGML_CUDA_BF16_CMP(>)
GGML_CUDA_BF16_CMP(<=)
GGML_CUDA_BF16_CMP(>=)

#undef GGML_CUDA_BF16_CMP

static __host__ __device__ __forceinline__ nv_bfloat16 operator-(const nv_bfloat16 a) {
    return __float2bfloat16(-__bfloat162float(a));
}

static __host__ __device__ __forceinline__ nv_bfloat162 operator-(const nv_bfloat162 a) {
    return __floats2bfloat162_rn(-__low2float(a), -__high2float(a));
}

static __host__ __device__ __forceinline__ nv_bfloat16 __hneg(const nv_bfloat16 a) {
    return -a;
}

static __host__ __device__ __forceinline__ nv_bfloat162 __hneg2(const nv_bfloat162 a) {
    return -a;
}

static __host__ __device__ __forceinline__ nv_bfloat16 __hadd(const nv_bfloat16 a, const nv_bfloat16 b) { return a + b; }
static __host__ __device__ __forceinline__ nv_bfloat16 __hsub(const nv_bfloat16 a, const nv_bfloat16 b) { return a - b; }
static __host__ __device__ __forceinline__ nv_bfloat16 __hmul(const nv_bfloat16 a, const nv_bfloat16 b) { return a * b; }
static __host__ __device__ __forceinline__ nv_bfloat16 __hfma(const nv_bfloat16 a, const nv_bfloat16 b, const nv_bfloat16 c) {
    return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b) + __bfloat162float(c));
}

static __host__ __device__ __forceinline__ nv_bfloat162 __hadd2(const nv_bfloat162 a, const nv_bfloat162 b) { return a + b; }
static __host__ __device__ __forceinline__ nv_bfloat162 __hsub2(const nv_bfloat162 a, const nv_bfloat162 b) { return a - b; }
static __host__ __device__ __forceinline__ nv_bfloat162 __hmul2(const nv_bfloat162 a, const nv_bfloat162 b) { return a * b; }
static __host__ __device__ __forceinline__ nv_bfloat162 __hfma2(const nv_bfloat162 a, const nv_bfloat162 b, const nv_bfloat162 c) {
    return __floats2bfloat162_rn(__low2float(a) * __low2float(b) + __low2float(c),
                                 __high2float(a) * __high2float(b) + __high2float(c));
}

static __host__ __device__ __forceinline__ nv_bfloat16 __hmax(const nv_bfloat16 a, const nv_bfloat16 b) {
    return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
}

static __host__ __device__ __forceinline__ nv_bfloat16 __hmin(const nv_bfloat16 a, const nv_bfloat16 b) {
    return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
}

static __host__ __device__ __forceinline__ nv_bfloat162 __hmax2(const nv_bfloat162 a, const nv_bfloat162 b) {
    return __floats2bfloat162_rn(fmaxf(__low2float(a), __low2float(b)), fmaxf(__high2float(a), __high2float(b)));
}

static __host__ __device__ __forceinline__ nv_bfloat162 __hmin2(const nv_bfloat162 a, const nv_bfloat162 b) {
    return __floats2bfloat162_rn(fminf(__low2float(a), __low2float(b)), fminf(__high2float(a), __high2float(b)));
}
