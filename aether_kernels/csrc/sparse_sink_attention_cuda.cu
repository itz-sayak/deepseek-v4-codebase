#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <limits>

namespace {

template <typename scalar_t>
__device__ inline float to_float_device(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ inline scalar_t from_float_device(float value) {
  return static_cast<scalar_t>(value);
}

__device__ inline float block_reduce_max(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ inline float block_reduce_sum(float value, float* scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

template <typename scalar_t, typename index_t>
__global__ void sparse_sink_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ kv,
    const index_t* __restrict__ topk_indices,
    const scalar_t* __restrict__ sink,
    scalar_t* __restrict__ out,
    int batch_size,
    int target_tokens,
    int num_heads,
    int head_dim,
    int source_tokens,
    int top_k,
    float softmax_scale,
    bool sink_is_scalar) {
  const int linear = blockIdx.x;
  const int head = linear % num_heads;
  const int token = (linear / num_heads) % target_tokens;
  const int batch = linear / (num_heads * target_tokens);

  extern __shared__ float shared[];
  float* logits = shared;
  float* scratch = shared + top_k;

  const scalar_t* q_ptr = q + (((batch * target_tokens + token) * num_heads + head) * head_dim);
  const index_t* topk_ptr = topk_indices + ((batch * target_tokens + token) * top_k);
  const float sink_logit = sink_is_scalar ? to_float_device(sink[0]) : to_float_device(sink[head]);

  float local_max = sink_logit;
  for (int k_idx = threadIdx.x; k_idx < top_k; k_idx += blockDim.x) {
    const int src = static_cast<int>(topk_ptr[k_idx]);
    float score = -INFINITY;
    if (src >= 0 && src < source_tokens) {
      const scalar_t* k_ptr = kv + (((batch * source_tokens + src) * 2 + 0) * head_dim);
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += to_float_device(q_ptr[d]) * to_float_device(k_ptr[d]);
      }
      score = dot * softmax_scale;
    }
    logits[k_idx] = score;
    local_max = fmaxf(local_max, score);
  }
  __syncthreads();
  const float max_logit = block_reduce_max(local_max, scratch);
  __syncthreads();

  float local_sum = threadIdx.x == 0 ? expf(sink_logit - max_logit) : 0.0f;
  for (int k_idx = threadIdx.x; k_idx < top_k; k_idx += blockDim.x) {
    local_sum += expf(logits[k_idx] - max_logit);
  }
  const float denom = block_reduce_sum(local_sum, scratch);
  __syncthreads();

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int k_idx = 0; k_idx < top_k; ++k_idx) {
      const int src = static_cast<int>(topk_ptr[k_idx]);
      if (src < 0 || src >= source_tokens) {
        continue;
      }
      const float weight = expf(logits[k_idx] - max_logit) / denom;
      const scalar_t* v_ptr = kv + (((batch * source_tokens + src) * 2 + 1) * head_dim);
      acc += weight * to_float_device(v_ptr[d]);
    }
    out[(((batch * target_tokens + token) * num_heads + head) * head_dim) + d] = from_float_device<scalar_t>(acc);
  }
}

}  // namespace

torch::Tensor sparse_sink_attention_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& topk_indices,
    const torch::Tensor& sink,
    double softmax_scale) {
  const auto batch_size = static_cast<int>(q.size(0));
  const auto target_tokens = static_cast<int>(q.size(1));
  const auto num_heads = static_cast<int>(q.size(2));
  const auto head_dim = static_cast<int>(q.size(3));
  const auto source_tokens = static_cast<int>(kv.size(1));
  const auto top_k = static_cast<int>(topk_indices.size(2));
  auto out = torch::zeros({q.size(0), q.size(1), q.size(2), q.size(3)}, q.options());
  const int threads = head_dim >= 256 ? 256 : (head_dim >= 128 ? 128 : 64);
  const int blocks = batch_size * target_tokens * num_heads;
  const size_t shared_bytes = sizeof(float) * (top_k + threads);
  auto stream = c10::cuda::getDefaultCUDAStream();
  const bool sink_is_scalar = sink.numel() == 1;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      q.scalar_type(),
      "sparse_sink_attention_cuda",
      [&] {
        if (topk_indices.scalar_type() == torch::kInt64) {
          sparse_sink_attention_kernel<scalar_t, int64_t><<<blocks, threads, shared_bytes, stream>>>(
              q.data_ptr<scalar_t>(),
              kv.data_ptr<scalar_t>(),
              topk_indices.data_ptr<int64_t>(),
              sink.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),
              batch_size,
              target_tokens,
              num_heads,
              head_dim,
              source_tokens,
              top_k,
              static_cast<float>(softmax_scale),
              sink_is_scalar);
        } else {
          sparse_sink_attention_kernel<scalar_t, int32_t><<<blocks, threads, shared_bytes, stream>>>(
              q.data_ptr<scalar_t>(),
              kv.data_ptr<scalar_t>(),
              topk_indices.data_ptr<int32_t>(),
              sink.data_ptr<scalar_t>(),
              out.data_ptr<scalar_t>(),
              batch_size,
              target_tokens,
              num_heads,
              head_dim,
              source_tokens,
              top_k,
              static_cast<float>(softmax_scale),
              sink_is_scalar);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
