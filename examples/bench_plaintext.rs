//! Plaintext inference throughput benchmark for weirwood.
//!
//! Called by `benchmarks/run_benchmark.sh`; can also be run directly:
//!
//! ```sh
//! cargo run --release --example bench_plaintext
//! cargo run --release --example bench_plaintext -- tests/fixtures/trained_binary.json
//! ```

use std::time::Instant;

use weirwood::{eval::PlaintextEvaluator, model::Ensemble};

const WARMUP: usize = 1_000;
const ITERATIONS: usize = 100_000;

fn main() -> Result<(), weirwood::Error> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/trained_binary.ubj".to_string());

    let ensemble = if model_path.ends_with(".ubj") {
        Ensemble::from_ubj_file(&model_path)?
    } else {
        Ensemble::from_json_file(&model_path)?
    };

    let features: Vec<f32> = vec![0.7, 0.3];
    let eval = PlaintextEvaluator;

    // Warm up instruction / branch-predictor caches.
    for _ in 0..WARMUP {
        eval.predict_proba(&ensemble, &features);
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        eval.predict_proba(&ensemble, &features);
    }
    let elapsed = start.elapsed();

    let per_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
    let throughput = ITERATIONS as f64 / elapsed.as_secs_f64();

    println!("weirwood plaintext inference ({model_path})");
    println!("  iterations : {ITERATIONS}");
    println!("  total      : {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  per call   : {per_ns:.1} ns");
    println!("  throughput : {throughput:.0} inferences/sec");

    Ok(())
}
