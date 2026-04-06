//! Latency benchmark for FHE stump inference.
//!
//! Measures and prints a side-by-side table comparing:
//!   - weirwood plaintext (Rust)
//!   - FHE evaluation (weirwood + tfhe-rs, CPU)
//!
//! FHE latency is averaged over 10 runs.  The stump has 1 internal node so
//! each inference costs exactly 1 PBS operation (~410 ms on CPU), giving a
//! total FHE benchmark time of ~4 s.  Plaintext uses 10,000 iterations for a
//! stable per-call figure.
//!
//! Model: single decision stump (`tests/fixtures/stump_regression.json`).
//!        Depth-1 tree — one TFHE comparison per prediction.
//!
//! Expect ~30 s total (keygen ~800 ms + plaintext bench + 10 FHE runs).
//! Always run in release mode:
//!
//! ```sh
//! cargo run --release --example bench_fhe_stump
//! cargo run --release --example bench_fhe_stump -- tests/fixtures/stump_regression.json
//! ```

use std::time::Instant;

use weirwood::{
    eval::{Evaluator as _, PlaintextEvaluator},
    fhe::{ClientContext, FheEvaluator},
    model::WeirwoodTree,
};

const DEFAULT_MODEL: &str = "tests/fixtures/stump_regression.json";
const PLAINTEXT_WARMUP: usize = 1_000;
const PLAINTEXT_ITERS: usize = 10_000;
// FHE is benchmarked over this many iterations and averaged.
const FHE_ITERS: usize = 10;

fn main() -> Result<(), weirwood::Error> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let model = WeirwoodTree::from_json_file(&model_path)?;
    let features: Vec<f32> = vec![0.0]; // left-branch probe

    println!("weirwood · stump inference benchmark");
    println!("  model    : {model_path}");
    println!(
        "  stump    : {} tree(s), {} feature(s)",
        model.trees.len(),
        model.num_features,
    );
    println!();

    // -----------------------------------------------------------------------
    // Plaintext baseline
    // -----------------------------------------------------------------------
    let plain_eval = PlaintextEvaluator;

    for _ in 0..PLAINTEXT_WARMUP {
        plain_eval.predict(&model, &features);
    }

    let t = Instant::now();
    for _ in 0..PLAINTEXT_ITERS {
        plain_eval.predict(&model, &features);
    }
    let plain_elapsed = t.elapsed();
    let plain_ns = plain_elapsed.as_nanos() as f64 / PLAINTEXT_ITERS as f64;
    let plain_thru = PLAINTEXT_ITERS as f64 / plain_elapsed.as_secs_f64();

    // -----------------------------------------------------------------------
    // Client: key generation
    // -----------------------------------------------------------------------
    print!("Generating FHE keys … ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t_keygen = Instant::now();
    let client = ClientContext::generate()?;
    let keygen_ms = t_keygen.elapsed().as_secs_f64() * 1000.0;
    println!("{keygen_ms:.0} ms");

    // Client extracts the server context (ServerKey only — no private key).
    let server_ctx = client.server_context();

    // -----------------------------------------------------------------------
    // Server setup
    // -----------------------------------------------------------------------
    server_ctx.set_active();
    let fhe_eval = FheEvaluator::new(server_ctx);

    // -----------------------------------------------------------------------
    // Client: encryption latency
    // -----------------------------------------------------------------------
    let t = Instant::now();
    let enc_input = client.encrypt(&features);
    let enc_ms = t.elapsed().as_secs_f64() * 1000.0;

    // -----------------------------------------------------------------------
    // Server: inference latency (averaged over FHE_ITERS — bootstrapping dominates)
    // -----------------------------------------------------------------------
    print!("Running FHE inference ({FHE_ITERS} iterations, will average) … ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t = Instant::now();
    let mut enc_score = fhe_eval.predict(&model, &enc_input);
    for _ in 1..FHE_ITERS {
        enc_score = fhe_eval.predict(&model, &enc_input);
    }
    let fhe_elapsed = t.elapsed();
    let fhe_s = fhe_elapsed.as_secs_f64() / FHE_ITERS as f64;
    println!("done");

    // -----------------------------------------------------------------------
    // Client: decryption latency
    // -----------------------------------------------------------------------
    let t = Instant::now();
    let fhe_score = client.decrypt_score(&enc_score);
    let dec_ms = t.elapsed().as_secs_f64() * 1000.0;

    let plain_score = PlaintextEvaluator.predict(&model, &features);
    let delta = (fhe_score - plain_score).abs();

    // -----------------------------------------------------------------------
    // Results
    // -----------------------------------------------------------------------
    println!();
    println!("FHE phase breakdown");
    println!("  keygen    : {keygen_ms:.0} ms");
    println!("  encrypt   : {enc_ms:.3} ms");
    let pbs_ops: usize = model
        .trees
        .iter()
        .map(|t| t.nodes.iter().filter(|n| !n.is_leaf()).count())
        .sum();
    println!("  inference : {fhe_s:.2} s  (avg over {FHE_ITERS} runs, {pbs_ops} PBS op(s) each)");
    println!("  decrypt   : {dec_ms:.3} ms");
    println!();
    println!("Correctness check");
    println!("  plaintext  : {plain_score:.4}");
    println!("  FHE result : {fhe_score:.4}");
    println!("  |Δ|        : {delta:.4}  (≤ 0.01 expected with SCALE=100)");
    println!();

    // -----------------------------------------------------------------------
    // Comparison table (machine-parseable markers for run_benchmark_stump.sh)
    // -----------------------------------------------------------------------
    println!("BENCH_PLAIN_NS={plain_ns:.1}");
    println!("BENCH_PLAIN_THRU={plain_thru:.0}");
    println!("BENCH_FHE_KEYGEN_MS={keygen_ms:.0}");
    println!("BENCH_FHE_ENC_MS={enc_ms:.3}");
    println!("BENCH_FHE_INFER_S={fhe_s:.2}");
    println!("BENCH_FHE_THRU={:.2}", 1.0 / fhe_s);
    println!("BENCH_FHE_DEC_MS={dec_ms:.3}");
    println!("BENCH_FHE_SCORE={fhe_score:.4}");
    println!("BENCH_PLAIN_SCORE={plain_score:.4}");
    println!("BENCH_DELTA={delta:.4}");

    Ok(())
}
