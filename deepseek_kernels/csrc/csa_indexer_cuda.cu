// CSA lightning indexer top-k CUDA kernel.
//
// Replaces the pure-PyTorch einsum + topk selection inside the CSA attention
// path with a single fused kernel that avoids materialising intermediate score
// matrices at long sequence lengths.
//
// Inputs:
//   index_q       : [B, IH, IC]   per-head compressed queries  (float32)
//   index_kv      : [B, S, IC]    compressed KV index embeddings
//   index_weight  : [B, IH]       per-head routing weights (after relu, before sum)
//   k             : int           number of entries to select
// Output:
//   topk_indices  : [B, k]        indices into the source sequence

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// One block handles one batch element.  Threads cooperate to compute
// per-source scores and then perform a parallel partial sort for top-k.
__global__ void csa_indexer_topk_kernel(
    const float* __restrict__ index_q,      // [B, IH, IC]
    const float* __restrict__ index_kv,     // [B, S, IC]
    const float* __restrict__ index_weight, // [B, IH]
    int64_t* __restrict__ topk_out,         // [B, k]
    int B, int S, int IH, int IC, int K)
{
    int b = blockIdx.x;
    if (b >= B) return;

    // Each thread scores a contiguous range of source entries.
    extern __shared__ float shared[];
    float* scores = shared;           // [S]
    // We also need a scratch top-k buffer; we keep K best (val, idx) pairs.
    // For K <= 256, a warp-level scan is enough.

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Zero scores.
    for (int s = tid; s < S; s += nthreads) scores[s] = 0.0f;
    __syncthreads();

    // Accumulate: score[s] += sum_h( relu(index_weight[b,h]) * dot(index_q[b,h,:], index_kv[b,s,:]) )
    for (int h = 0; h < IH; h++) {
        float w = index_weight[b * IH + h];
        if (w <= 0.0f) continue;
        const float* q_h = index_q + (b * IH + h) * IC;
        for (int s = tid; s < S; s += nthreads) {
            const float* kv_s = index_kv + (b * S + s) * IC;
            float dot = 0.0f;
            for (int c = 0; c < IC; c++) dot += q_h[c] * kv_s[c];
            scores[s] += w * dot;
        }
    }
    __syncthreads();

    // Thread 0 does a simple partial-sort top-k (K is small, usually 256).
    if (tid == 0) {
        int64_t* out = topk_out + b * K;
        // Initialise with first K entries.
        for (int i = 0; i < K && i < S; i++) out[i] = i;
        // Replace minimum if a better entry is found.
        float min_val = 1e38f;
        int   min_pos = 0;
        if (K <= S) {
            for (int i = 0; i < K; i++)
                if (scores[i] < min_val) { min_val = scores[i]; min_pos = i; }
        }
        for (int s = K; s < S; s++) {
            if (scores[s] > min_val) {
                out[min_pos] = s;
                min_val = 1e38f;
                for (int i = 0; i < K; i++)
                    if (scores[out[i]] < min_val) { min_val = scores[out[i]]; min_pos = i; }
            }
        }
    }
}

torch::Tensor csa_indexer_topk_cuda(
    const torch::Tensor& index_q,
    const torch::Tensor& index_kv,
    const torch::Tensor& index_weight,
    int k)
{
    auto B  = index_q.size(0);
    auto IH = index_q.size(1);
    auto IC = index_q.size(2);
    auto S  = index_kv.size(1);

    TORCH_CHECK(index_q.scalar_type()      == torch::kFloat, "index_q must be float32");
    TORCH_CHECK(index_kv.scalar_type()     == torch::kFloat, "index_kv must be float32");
    TORCH_CHECK(index_weight.scalar_type() == torch::kFloat, "index_weight must be float32");
    TORCH_CHECK(k > 0 && k <= S, "k must be in [1, S]");

    auto topk_out = torch::empty({(long)B, (long)k}, index_q.options().dtype(torch::kInt64));

    int threads = min((int)S, 256);
    size_t shared_bytes = S * sizeof(float);

    csa_indexer_topk_kernel<<<B, threads, shared_bytes>>>(
        index_q.data_ptr<float>(),
        index_kv.data_ptr<float>(),
        index_weight.data_ptr<float>(),
        topk_out.data_ptr<int64_t>(),
        (int)B, (int)S, (int)IH, (int)IC, k);

    return topk_out;
}
