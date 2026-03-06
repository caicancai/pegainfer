//! Prefill benchmark for measuring TTFT at various prompt lengths.
//!
//! Usage:
//!   cargo test -r --test bench_prefill -- bench_prefill_1024 --nocapture

use pegainfer::model::{ModelRuntimeConfig, Qwen3Model};
use pegainfer::sampler::SamplingParams;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

fn bench_prefill(seq_len: usize, warmup: usize, iters: usize) {
    pegainfer::logging::init_stderr("warn");

    let mut model = Qwen3Model::from_safetensors_with_runtime(
        MODEL_PATH,
        ModelRuntimeConfig {
            enable_cuda_graph: true,
        },
    )
    .expect("Failed to load model");

    let greedy = SamplingParams::default();
    let mut rng = StdRng::seed_from_u64(42);

    // Synthetic prompt: valid token IDs (100..100+seq_len), wrapped
    let prompt: Vec<u32> = (0..seq_len).map(|i| ((i % 1000) + 100) as u32).collect();

    // Warmup
    eprintln!("[warmup] {} runs, seq_len={}", warmup, seq_len);
    for _ in 0..warmup {
        let _ = model
            .generate(&prompt, 1, &greedy, &mut rng)
            .expect("warmup failed");
    }

    // Benchmark: generate with max_tokens=1 (prefill only, no decode loop)
    eprintln!("[bench] {} runs, seq_len={}", iters, seq_len);
    let mut ttft_sum = 0.0f64;
    for _ in 0..iters {
        let start = Instant::now();
        let _ = model
            .generate(&prompt, 1, &greedy, &mut rng)
            .expect("bench failed");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        ttft_sum += elapsed;
    }
    let avg_ttft = ttft_sum / iters as f64;
    let throughput = seq_len as f64 / (avg_ttft / 1000.0);

    eprintln!(
        "[result] seq_len={}, TTFT={:.2}ms, prefill_throughput={:.0} tok/s",
        seq_len, avg_ttft, throughput
    );
}

#[test]
fn bench_prefill_1024() {
    bench_prefill(1024, 2, 5);
}

#[test]
#[ignore]
fn bench_prefill_sweep() {
    for &seq_len in &[1, 4, 16, 64, 256, 512, 1024] {
        bench_prefill(seq_len, 2, 5);
    }
}
