#include <torch/extension.h>

#include <stdexcept>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// ---------------------------------------------------------------------------
// sparse_sink_attention (decode, T == 1 typical use case)
// ---------------------------------------------------------------------------
torch::Tensor sparse_sink_attention_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& topk_indices,
    const torch::Tensor& sink,
    double softmax_scale);

torch::Tensor sparse_sink_attention(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& topk_indices,
    const torch::Tensor& sink,
    double softmax_scale) {
  CHECK_INPUT(q);
  CHECK_INPUT(kv);
  CHECK_INPUT(topk_indices);
  CHECK_INPUT(sink);
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, H, D]");
  TORCH_CHECK(kv.dim() == 4, "kv must have shape [B, S, 2, D]");
  TORCH_CHECK(kv.size(2) == 2, "kv third dimension must be 2 for [K, V]");
  TORCH_CHECK(topk_indices.dim() == 3, "topk_indices must have shape [B, T, K]");
  TORCH_CHECK(q.size(0) == kv.size(0), "q and kv batch dimensions must match");
  TORCH_CHECK(q.size(0) == topk_indices.size(0), "q and topk_indices batch dimensions must match");
  TORCH_CHECK(q.size(1) == topk_indices.size(1), "q and topk_indices target-token dimensions must match");
  TORCH_CHECK(q.size(3) == kv.size(3), "q and kv head dimensions must match");
  TORCH_CHECK(
      topk_indices.scalar_type() == torch::kInt64 || topk_indices.scalar_type() == torch::kInt32,
      "topk_indices must be int32 or int64");
  TORCH_CHECK(
      sink.numel() == 1 || (sink.dim() == 1 && sink.size(0) == q.size(2)),
      "sink must be a scalar or a [num_heads] tensor");
  return sparse_sink_attention_cuda(q, kv, topk_indices, sink, softmax_scale);
}

// ---------------------------------------------------------------------------
// tiled_prefill_attention (prefill path, streams source tokens in tiles)
// ---------------------------------------------------------------------------
torch::Tensor tiled_prefill_attention_cuda(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& sink,
    double softmax_scale,
    int tile_size);

torch::Tensor tiled_prefill_attention(
    const torch::Tensor& q,
    const torch::Tensor& kv,
    const torch::Tensor& sink,
    double softmax_scale,
    int tile_size) {
  CHECK_INPUT(q);
  CHECK_INPUT(kv);
  CHECK_INPUT(sink);
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, H, D]");
  TORCH_CHECK(kv.dim() == 4, "kv must have shape [B, S, 2, D]");
  TORCH_CHECK(kv.size(2) == 2, "kv third dimension must be 2 for [K, V]");
  TORCH_CHECK(q.size(0) == kv.size(0), "q and kv batch dimensions must match");
  TORCH_CHECK(q.size(3) == kv.size(3), "q and kv head dimensions must match");
  TORCH_CHECK(tile_size > 0, "tile_size must be positive");
  TORCH_CHECK(
      sink.numel() == 1 || (sink.dim() == 1 && sink.size(0) == q.size(2)),
      "sink must be a scalar or a [num_heads] tensor");
  return tiled_prefill_attention_cuda(q, kv, sink, softmax_scale, tile_size);
}

// ---------------------------------------------------------------------------
// csa_indexer_topk (CSA lightning indexer fused score + top-k)
// ---------------------------------------------------------------------------
torch::Tensor csa_indexer_topk_cuda(
    const torch::Tensor& index_q,
    const torch::Tensor& index_kv,
    const torch::Tensor& index_weight,
    int k);

torch::Tensor csa_indexer_topk(
    const torch::Tensor& index_q,
    const torch::Tensor& index_kv,
    const torch::Tensor& index_weight,
    int k) {
  CHECK_INPUT(index_q);
  CHECK_INPUT(index_kv);
  CHECK_INPUT(index_weight);
  TORCH_CHECK(index_q.dim() == 3,      "index_q must have shape [B, IH, IC]");
  TORCH_CHECK(index_kv.dim() == 3,     "index_kv must have shape [B, S, IC]");
  TORCH_CHECK(index_weight.dim() == 2, "index_weight must have shape [B, IH]");
  TORCH_CHECK(k > 0, "k must be positive");
  return csa_indexer_topk_cuda(index_q, index_kv, index_weight, k);
}

// ---------------------------------------------------------------------------
// hca_compress (fused softmax-weighted compression for one HCA window)
// ---------------------------------------------------------------------------
torch::Tensor hca_compress_cuda(
    const torch::Tensor& buffer_c,
    const torch::Tensor& buffer_z);

torch::Tensor hca_compress(
    const torch::Tensor& buffer_c,
    const torch::Tensor& buffer_z) {
  CHECK_INPUT(buffer_c);
  CHECK_INPUT(buffer_z);
  TORCH_CHECK(buffer_c.dim() == 3, "buffer_c must have shape [B, M, D]");
  TORCH_CHECK(buffer_z.dim() == 3, "buffer_z must have shape [B, M, D]");
  TORCH_CHECK(buffer_c.sizes() == buffer_z.sizes(), "buffer_c and buffer_z must have the same shape");
  return hca_compress_cuda(buffer_c, buffer_z);
}

// ---------------------------------------------------------------------------
// Module export
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "sparse_sink_attention",
      &sparse_sink_attention,
      "Fused sparse sink attention over compressed KV entries (CUDA)");
  m.def(
      "tiled_prefill_attention",
      &tiled_prefill_attention,
      "Tiled prefill attention for long sequences; streams source tokens in tiles (CUDA)");
  m.def(
      "csa_indexer_topk",
      &csa_indexer_topk,
      "CSA lightning indexer: fused dot-product score and top-k selection (CUDA)");
  m.def(
      "hca_compress",
      &hca_compress,
      "Fused softmax-weighted HCA KV compression for one compression window (CUDA)");
}
