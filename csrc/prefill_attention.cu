#include "common.cuh"
#include <cublas_v2.h>

// cuBLAS handle from gemv.cu
extern cublasHandle_t g_cublas_handle;

#define HEAD_DIM 128
#define SOFTMAX_BLOCK 256
#define SOFTMAX_NUM_WARPS (SOFTMAX_BLOCK / WARP_SIZE)

// ============================================================================
// Kernel 1: Per-head QK RMSNorm + RoPE (in-place on Q and K batches)
//
// Grid: (num_q_heads + num_kv_heads, seq_len)
// Block: head_dim (128) threads
// Each block normalizes one head of one token, then applies RoPE.
// ============================================================================
__global__ void prefill_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,        // [q_dim, seq_len] modified in-place
    __nv_bfloat16* __restrict__ k,        // [kv_dim, seq_len] modified in-place
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_cache,      // [max_pos * head_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int q_dim, int kv_dim, int start_pos, float eps
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = (head_global < num_q_heads);
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    __nv_bfloat16* data = is_q ? q : k;
    int dim_stride = is_q ? q_dim : kv_dim;
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    int offset = head_local * head_dim + d + token * dim_stride;
    float val = __bfloat162float(data[offset]);

    // RMSNorm: sum of squares via warp reduction
    float sq = val * val;
    sq = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[4];  // head_dim/32 = 4 warps
    if (lane_id == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    __shared__ float s_inv_rms;
    if (warp_id == 0) {
        float v = (lane_id < 4) ? warp_sums[lane_id] : 0.0f;
        float total = warp_reduce_sum(v);
        if (lane_id == 0) s_inv_rms = rsqrtf(total / head_dim + eps);
    }
    __syncthreads();

    // Match HF precision: round to bf16 after norm, then multiply weight
    __nv_bfloat16 normed = __float2bfloat16(val * s_inv_rms);
    float normed_f = __bfloat162float(normed) * __bfloat162float(norm_w[d]);

    // RoPE via shared memory exchange
    __shared__ __nv_bfloat16 smem[HEAD_DIM];
    smem[d] = __float2bfloat16(normed_f);
    __syncthreads();

    int half = head_dim / 2;
    int pos = start_pos + token;

    __nv_bfloat16 result;
    if (d < half) {
        float lo = __bfloat162float(smem[d]);
        float hi = __bfloat162float(smem[d + half]);
        float c = __bfloat162float(cos_cache[pos * head_dim + d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + d]);
        result = __float2bfloat16(lo * c - hi * s);
    } else {
        int pair_d = d - half;
        float lo = __bfloat162float(smem[pair_d]);
        float hi = __bfloat162float(smem[d]);
        float c = __bfloat162float(cos_cache[pos * head_dim + pair_d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + pair_d]);
        result = __float2bfloat16(lo * s + hi * c);
    }

    data[offset] = result;
}

// ============================================================================
// Kernel 2: Batch write K and V to KV cache
//
// Grid: (num_kv_heads, seq_len), Block: head_dim
// K is already normed+RoPE'd, V is raw.
// ============================================================================
__global__ void prefill_kv_cache_write_kernel(
    const __nv_bfloat16* __restrict__ k,   // [kv_dim, seq_len] normed+RoPE'd
    const __nv_bfloat16* __restrict__ v,   // [kv_dim, seq_len]
    __nv_bfloat16* __restrict__ k_cache,   // [num_kv_heads * max_seq * head_dim]
    __nv_bfloat16* __restrict__ v_cache,
    int head_dim, int kv_dim, int max_seq_len, int start_pos
) {
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    int src_offset = kv_head * head_dim + d + token * kv_dim;
    int dst_offset = kv_head * max_seq_len * head_dim + (start_pos + token) * head_dim + d;

    k_cache[dst_offset] = k[src_offset];
    v_cache[dst_offset] = v[src_offset];
}

// ============================================================================
// Kernel 3: Causal softmax (FP32 scores → BF16 probabilities)
//
// Scores layout: column-major [total_seq, seq_len] per head, from K^T @ Q.
// Column i = scores for query i, stored contiguously at col_offset = i * total_seq.
// Causal mask: key j is valid if j <= start_pos + query_idx.
//
// Grid: (num_heads, seq_len), Block: SOFTMAX_BLOCK (256)
// ============================================================================
__global__ void causal_softmax_kernel(
    const float* __restrict__ scores_fp32,
    __nv_bfloat16* __restrict__ out_bf16,
    int total_seq, int seq_len, int start_pos
) {
    int head = blockIdx.x;
    int query = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    long long stride = (long long)total_seq * seq_len;
    const float* col = scores_fp32 + head * stride + (long long)query * total_seq;
    __nv_bfloat16* out_col = out_bf16 + head * stride + (long long)query * total_seq;

    int valid_len = start_pos + query + 1;

    // Phase 1: find max
    float local_max = -INFINITY;
    for (int j = tid; j < valid_len; j += SOFTMAX_BLOCK) {
        local_max = fmaxf(local_max, col[j]);
    }
    local_max = warp_reduce_max(local_max);

    __shared__ float warp_maxes[SOFTMAX_NUM_WARPS];
    if (lane_id == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();

    __shared__ float s_max;
    if (warp_id == 0) {
        float v = (lane_id < SOFTMAX_NUM_WARPS) ? warp_maxes[lane_id] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane_id == 0) s_max = v;
    }
    __syncthreads();
    float global_max = s_max;

    // Phase 2: exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < valid_len; j += SOFTMAX_BLOCK) {
        local_sum += expf(col[j] - global_max);
    }
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float warp_sums[SOFTMAX_NUM_WARPS];
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    __shared__ float s_sum;
    if (warp_id == 0) {
        float v = (lane_id < SOFTMAX_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) s_sum = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Phase 3: normalize → BF16
    for (int j = tid; j < valid_len; j += SOFTMAX_BLOCK) {
        float val = expf(col[j] - global_max) * inv_sum;
        out_col[j] = __float2bfloat16(val);
    }
    // Zero masked entries
    for (int j = valid_len + tid; j < total_seq; j += SOFTMAX_BLOCK) {
        out_col[j] = __float2bfloat16(0.0f);
    }
}

// ============================================================================
// C API: Batched prefill attention
//
// Replaces the per-token attention loop with:
//   1. Per-head QK norm + RoPE (custom kernel)
//   2. KV cache batch write (custom kernel)
//   3. K^T @ Q attention scores (cuBLAS strided batched GEMM × num_kv_heads)
//   4. Causal softmax (custom kernel)
//   5. V @ softmax output (cuBLAS strided batched GEMM × num_kv_heads)
//
// Supports start_pos >= 0 for multi-turn by reading K/V from cache.
// ============================================================================
extern "C" {

void prefill_attention_cuda(
    __nv_bfloat16* q_batch,          // [q_dim, seq_len] modified in-place
    __nv_bfloat16* k_batch,          // [kv_dim, seq_len] modified in-place
    const __nv_bfloat16* v_batch,    // [kv_dim, seq_len]
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* k_cache,          // [num_kv_heads * max_seq * head_dim]
    __nv_bfloat16* v_cache,
    __nv_bfloat16* output,           // [q_dim, seq_len]
    float* scores_buf,               // temp [num_q_heads * total_seq * seq_len] FP32
    __nv_bfloat16* softmax_buf,      // temp [num_q_heads * total_seq * seq_len] BF16
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    float scale,
    float rms_eps,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int gqa_ratio = num_q_heads / num_kv_heads;
    int max_seq_len = 4096;
    int total_seq = start_pos + seq_len;

    // Step 1: QK norm + RoPE (in-place)
    dim3 norm_grid(num_q_heads + num_kv_heads, seq_len);
    prefill_qk_norm_rope_kernel<<<norm_grid, head_dim, 0, stream>>>(
        q_batch, k_batch, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim,
        seq_len, q_dim, kv_dim, start_pos, rms_eps
    );

    // Step 2: Write K, V to cache
    dim3 cache_grid(num_kv_heads, seq_len);
    prefill_kv_cache_write_kernel<<<cache_grid, head_dim, 0, stream>>>(
        k_batch, v_batch, k_cache, v_cache,
        head_dim, kv_dim, max_seq_len, start_pos
    );

    // Step 3: Attention scores C' = K_cache^T @ Q  [total_seq, seq_len] per head
    // Use cache for K (contains all positions 0..total_seq).
    // cuBLAS strided batched GEMM per GQA group (gqa_ratio Q heads share 1 K head).
    cublasSetStream(g_cublas_handle, stream);
    float zero = 0.0f;
    long long score_stride = (long long)total_seq * seq_len;

    for (int g = 0; g < num_kv_heads; g++) {
        // K_g from cache: [head_dim, total_seq] with lda = head_dim
        const __nv_bfloat16* K_g = k_cache + (long long)g * max_seq_len * head_dim;
        // Q heads in this group: [head_dim, seq_len] each, stride = head_dim between heads
        const __nv_bfloat16* Q_group = q_batch + g * gqa_ratio * head_dim;
        float* C_group = scores_buf + g * gqa_ratio * score_stride;

        cublasGemmStridedBatchedEx(
            g_cublas_handle,
            CUBLAS_OP_T,    // K^T
            CUBLAS_OP_N,    // Q
            total_seq,      // M
            seq_len,        // N
            head_dim,       // K
            &scale,
            K_g, CUDA_R_16BF, head_dim,
            0,              // strideA = 0 (broadcast same K)
            Q_group, CUDA_R_16BF, q_dim,
            (long long)head_dim,       // strideB between Q heads
            &zero,
            C_group, CUDA_R_32F, total_seq,
            score_stride,   // strideC
            gqa_ratio,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }

    // Step 4: Causal softmax (FP32 → BF16)
    dim3 softmax_grid(num_q_heads, seq_len);
    causal_softmax_kernel<<<softmax_grid, SOFTMAX_BLOCK, 0, stream>>>(
        scores_buf, softmax_buf, total_seq, seq_len, start_pos
    );

    // Step 5: Output = V_cache @ softmax  [head_dim, seq_len] per head
    float one = 1.0f;

    for (int g = 0; g < num_kv_heads; g++) {
        const __nv_bfloat16* V_g = v_cache + (long long)g * max_seq_len * head_dim;
        const __nv_bfloat16* S_group = softmax_buf + g * gqa_ratio * score_stride;
        __nv_bfloat16* O_group = output + g * gqa_ratio * head_dim;

        cublasGemmStridedBatchedEx(
            g_cublas_handle,
            CUBLAS_OP_N,    // V
            CUBLAS_OP_N,    // softmax
            head_dim,       // M
            seq_len,        // N
            total_seq,      // K
            &one,
            V_g, CUDA_R_16BF, head_dim,
            0,              // strideA = 0 (broadcast same V)
            S_group, CUDA_R_16BF, total_seq,
            score_stride,   // strideB
            &zero,
            O_group, CUDA_R_16BF, q_dim,
            (long long)head_dim,       // strideC between output heads
            gqa_ratio,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
}

} // extern "C"
