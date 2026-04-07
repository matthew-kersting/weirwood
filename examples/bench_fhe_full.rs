//! Latency benchmark for FHE full-ensemble inference.
//!
//! Measures end-to-end FHE inference on the committed `trained_binary.ubj`
//! fixture (100 trees, max_depth=8, 177 internal nodes, 2 features,
//! `binary:logistic`) and compares against the plaintext Rust baseline.
//!
//! FHE latency is averaged over 10 runs.  With 177 PBS ops per inference at
//! ~0.52 s each, expect ~92 s per run and ~15 min total on CPU.
//! Plaintext uses 10,000 iterations for a stable per-call figure.
//!
//! Called by `benchmarks/run_benchmark.sh`; can also be run directly:
//!
//! ```sh
//! cargo run --release --example bench_fhe_full
//! cargo run --release --example bench_fhe_full -- tests/fixtures/trained_binary.ubj
//! ```
//!
//! WARNING: expect ~15 min total on CPU. Always run in release mode.

use std::time::Instant;

use weirwood::{
    eval::{Evaluator as _, PlaintextEvaluator},
    fhe::{ClientContext, FheEvaluator},
    model::WeirwoodTree,
};

const DEFAULT_MODEL: &str = "tests/fixtures/trained_binary.ubj";
const PLAINTEXT_WARMUP: usize = 1_000;
const PLAINTEXT_ITERS: usize = 10_000;
const FHE_ITERS: usize = 10;

fn main() -> Result<(), weirwood::Error> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let model = if model_path.ends_with(".ubj") {
        WeirwoodTree::from_ubj_file(&model_path)?
    } else {
        WeirwoodTree::from_json_file(&model_path)?
    };

    // Count internal nodes across all trees (= number of PBS operations per inference).
    let total_internal_nodes: usize = model
        .trees
        .iter()
        .map(|t| t.nodes.iter().filter(|n| !n.is_leaf()).count())
        .sum();

    let features: Vec<f32> = vec![0.7, 0.3];

    println!("weirwood · full-ensemble FHE inference benchmark");
    println!("  model       : {model_path}");
    println!(
        "  trees       : {}   depth ≤ {}   features : {}",
        model.trees.len(),
        max_depth(&model),
        model.num_features,
    );
    println!(
        "  PBS ops/infer: {total_internal_nodes}  (one per internal node, evaluated obliviously)"
    );
    println!("  objective   : {:?}", model.objective);
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
    // Server: FHE inference averaged over FHE_ITERS runs
    // -----------------------------------------------------------------------
    println!(
        "Running FHE inference ({FHE_ITERS} iterations, {total_internal_nodes} PBS ops each) … "
    );
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t = Instant::now();
    let mut enc_score = fhe_eval.predict(&model, &enc_input);
    for _ in 1..FHE_ITERS {
        enc_score = fhe_eval.predict(&model, &enc_input);
    }
    let fhe_elapsed = t.elapsed();
    let fhe_s = fhe_elapsed.as_secs_f64() / FHE_ITERS as f64;
    println!("done (avg {fhe_s:.1} s over {FHE_ITERS} runs)");

    // -----------------------------------------------------------------------
    // Client: decryption and correctness check
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
    println!(
        "  inference : {fhe_s:.1} s  (avg {FHE_ITERS} runs, {total_internal_nodes} PBS ops each)"
    );
    println!("  decrypt   : {dec_ms:.3} ms");
    println!();
    println!("Correctness check");
    println!("  plaintext  : {plain_score:.4}");
    println!("  FHE result : {fhe_score:.4}");
    println!(
        "  |Δ|        : {delta:.4}  (≤ {} expected with SCALE=100)",
        model.trees.len() as f32 * 0.5 / 100.0
    );
    println!();

    // -----------------------------------------------------------------------
    // Machine-parseable output for run_benchmark_fhe_full.sh
    // -----------------------------------------------------------------------
    println!("BENCH_PLAIN_NS={plain_ns:.1}");
    println!("BENCH_PLAIN_THRU={plain_thru:.0}");
    println!("BENCH_FHE_KEYGEN_MS={keygen_ms:.0}");
    println!("BENCH_FHE_ENC_MS={enc_ms:.3}");
    println!("BENCH_FHE_INFER_S={fhe_s:.2}");
    println!("BENCH_FHE_THRU={:.4}", 1.0 / fhe_s);
    println!("BENCH_FHE_DEC_MS={dec_ms:.3}");
    println!("BENCH_FHE_SCORE={fhe_score:.4}");
    println!("BENCH_PLAIN_SCORE={plain_score:.4}");
    println!("BENCH_DELTA={delta:.4}");
    println!("BENCH_PBS_OPS={total_internal_nodes}");

    Ok(())
}

fn max_depth(model: &WeirwoodTree) -> usize {
    model
        .trees
        .iter()
        .map(|tree| tree_depth(tree, 0, 0))
        .max()
        .unwrap_or(0)
}

fn tree_depth(tree: &weirwood::model::Tree, node_idx: usize, depth: usize) -> usize {
    let node = &tree.nodes[node_idx];
    if node.is_leaf() {
        depth
    } else {
        let left = tree_depth(tree, node.left_child as usize, depth + 1);
        let right = tree_depth(tree, node.right_child as usize, depth + 1);
        left.max(right)
    }
}
