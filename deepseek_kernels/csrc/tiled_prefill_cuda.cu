// Tiled prefill CUDA kernel for long-sequence attention.
//
// Processes source tokens in tiles so that the full attention matrix never
// materialises in HBM.  The contract matches the serving-layer tensor shapes:
//   q  : [B, T, H, D]       target tokens (T may be > 1 for prefill)
//   kv : [B, S, 2, D]       source KV entries (K then V in dim 2)
//   sink : [H] or scalar     per-head sink logit
//   out : [B, T, H, D]       softmax-weighted output

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define MAX_TILE 256

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) { return static_cast<float>(v); }
template <>
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) { return static_cast<scalar_t>(v); }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// One warp handles one (batch, head, query-token) triple and streams over all
// source tiles accumulating a running softmax.
template <typename scalar_t>
__global__ void tiled_prefill_kernel(
    const scalar_t* __restrict__ q,   // [B, T, H, D]
    const scalar_t* __restrict__ kv,  // [B, S, 2, D]
    const scalar_t* __restrict__ sink,// [H] or [1]
    scalar_t* __restrict__ out,       // [B, T, H, D]
    int B, int T, int S, int H, int D,
    float scale,
    int tile_size,
    bool scalar_sink)
{
    // Grid: (B * T * H)  blocks, each with 32 threads (one warp).
    int global_idx = blockIdx.x;
    int b = global_idx / (T * H);
    int rem = global_idx % (T * H);
    int t = rem / H;
    int h = rem % H;
    if (b >= B || t >= T || h >= H) return;

    int lane = threadIdx.x;

    // Running softmax state.
    float acc_m = -1e38f;   // running max
    float acc_l = 0.0f;     // normaliser exp(m_prev - m_new)
    float acc_v[256];        // output accumulator, max D=256
    for (int d = lane; d < D; d += WARP_SIZE) acc_v[d] = 0.0f;

    float sink_val = scalar_sink
                     ? to_float(sink[0])
                     : to_float(sink[h]);

    // q pointer for this (b, t, h)
    const scalar_t* q_ptr = q + ((b * T + t) * H + h) * D;

    // Stream over source tiles.
    for (int tile_start = 0; tile_start < S; tile_start += tile_size) {
        int tile_end = min(tile_start + tile_size, S);

        // Compute dot products for this tile.
        for (int s = tile_start; s < tile_end; s++) {
            // K pointer for this (b, s, channel=0)
            const scalar_t* k_ptr = kv + ((b * S + s) * 2 + 0) * D;
            float dot = 0.0f;
            for (int d = lane; d < D; d += WARP_SIZE) {
                dot += to_float(q_ptr[d]) * to_float(k_ptr[d]);
            }
            // Warp reduce.
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            dot = __shfl_sync(0xffffffff, dot, 0);
            dot *= scale;

            // Online softmax update.
            float new_m = max(acc_m, max(dot, sink_val));
            float exp_dot = expf(dot - new_m);
            float sink_exp = expf(sink_val - new_m);
            float scale_prev = expf(acc_m - new_m);
            float new_l = acc_l * scale_prev + exp_dot + sink_exp;

            // Update output accumulator.
            const scalar_t* v_ptr = kv + ((b * S + s) * 2 + 1) * D;
            for (int d = lane; d < D; d += WARP_SIZE) {
                acc_v[d] = acc_v[d] * (acc_l * scale_prev / fmaxf(new_l, 1e-12f))
                           + to_float(v_ptr[d]) * (exp_dot / fmaxf(new_l, 1e-12f));
            }
            acc_m = new_m;
            acc_l = new_l;
        }
    }

    // Write output.
    scalar_t* out_ptr = out + ((b * T + t) * H + h) * D;
    for (int d = lane; d < D; d += WARP_SIZE)
        out_ptr[d] = from_float<scalar_t>(acc_v[d]);
}

torch::Tensor tiled_prefill_attention_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& sink,
    double softmax_scale,
    int tile_size)
{
    auto B = q.size(0), T = q.size(1), H = q.size(2), D = q.size(3);
    auto S = kv.size(1);
    TORCH_CHECK(D <= 256, "tiled_prefill: head_dim must be <= 256, got ", D);
    auto out = torch::zeros_like(q);
    bool scalar_sink = (sink.numel() == 1);

    int blocks = B * T * H;
    int threads = WARP_SIZE;
    int ts = min(tile_size, MAX_TILE);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "tiled_prefill_attention_cuda", ([&] {
            tiled_prefill_kernel<scalar_t><<<blocks, threads>>>(
                q.data_ptr<scalar_t>(),
                kv.data_ptr<scalar_t>(),
                sink.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                (int)B, (int)T, (int)S, (int)H, (int)D,
                (float)softmax_scale, ts, scalar_sink);
        }));
    return out;
}
