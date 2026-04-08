//! End-to-end FHE inference example over a multi-tree XGBoost ensemble.
//!
//! Demonstrates full XGBoost inference — multiple trees of arbitrary depth —
//! running entirely in FHE, with **all scoring functions (sigmoid, softmax)
//! applied client-side** on the decrypted raw score.
//!
//! # Protocol
//!
//! ```text
//! ┌──────────── Client ─────────────┐        ┌───────── Server ──────────┐
//! │  1. Generate FHE keys            │        │                           │
//! │  2. Extract ServerContext ───────┼──────► │  3. Install server key    │
//! │  4. Encrypt features ────────────┼──────► │  5. Evaluate all trees    │
//! │                                  │ ◄───── │     (pure FHE, no plains) │
//! │  6. Decrypt raw score            │        │                           │
//! │  7. Apply sigmoid/identity       │        │                           │
//! └──────────────────────────────────┘        └───────────────────────────┘
//! ```
//!
//! Steps 5–6 happen over encrypted data; the server sees no feature values
//! or scores.  Activation functions (step 7) are computed client-side in
//! plaintext, after decryption.
//!
//! # Default model
//!
//! `tests/fixtures/trained_binary.ubj` — 100 trees, max_depth=8, 177 internal
//! nodes, 2 features, objective `binary:logistic`.  Pass any XGBoost JSON or
//! UBJ model file as a CLI argument, followed by optional feature values:
//!
//! ```sh
//! cargo run --release --example fhe_full_inference
//! cargo run --release --example fhe_full_inference -- path/to/model.ubj 0.7 0.3
//! ```
//!
//! **WARNING:** FHE key generation takes ~1 s; with Rayon tree-level parallelism
//! the default model (177 PBS ops, 100 trees) takes ~64 s per inference.  With 3
//! default test cases expect ~200 s (~3 min) total.  Always run in release mode.

use std::time::Instant;

use weirwood::{
    eval::{Evaluator as _, PlaintextEvaluator},
    fhe::{ClientContext, FheEvaluator},
    model::{Objective, WeirwoodTree},
};

const DEFAULT_MODEL: &str = "tests/fixtures/trained_binary.ubj";

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply the model's activation function to a raw ensemble score.
/// This runs client-side on plaintext after decryption.
fn apply_activation(objective: &Objective, raw_score: f32) -> f32 {
    match objective {
        Objective::BinaryLogistic => sigmoid(raw_score),
        Objective::RegSquaredError => raw_score,
        Objective::MultiSoftmax { .. } => raw_score, // softmax requires all class scores
        Objective::Other(_) => raw_score,
    }
}

fn main() -> Result<(), weirwood::Error> {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());

    // Remaining CLI args are treated as feature values (f32).
    let cli_features: Vec<f32> = args
        .map(|s| s.parse::<f32>().expect("feature values must be f32"))
        .collect();

    let model = if model_path.ends_with(".ubj") {
        WeirwoodTree::from_ubj_file(&model_path)?
    } else {
        WeirwoodTree::from_json_file(&model_path)?
    };

    println!("weirwood · full FHE XGBoost inference");
    println!("  model     : {model_path}");
    println!(
        "  trees     : {}   depth ≤ {}   features : {}",
        model.trees.len(),
        max_depth(&model),
        model.num_features,
    );
    println!("  objective : {:?}", model.objective);
    println!();

    // -----------------------------------------------------------------------
    // Client: key generation
    // -----------------------------------------------------------------------
    print!("Generating FHE keys … ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t_keygen = Instant::now();
    let client = ClientContext::generate()?;
    let keygen_ms = t_keygen.elapsed().as_secs_f64() * 1000.0;
    println!("{keygen_ms:.0} ms");
    println!();

    // Extract server context (ServerKey only — no private key material).
    let server_ctx = client.server_context();

    // -----------------------------------------------------------------------
    // Server setup
    // -----------------------------------------------------------------------
    server_ctx.set_active(); // install key on the calling thread (worker threads get it via start_handler)
    let evaluator = FheEvaluator::new(server_ctx);

    // -----------------------------------------------------------------------
    // Build test cases: CLI features (if any) + default probe vectors.
    // -----------------------------------------------------------------------
    let mut test_cases: Vec<Vec<f32>> = Vec::new();
    if !cli_features.is_empty() {
        test_cases.push(cli_features);
    } else {
        // Default probes for trained_binary.ubj (100 trees, max_depth=8, 2 features).
        // Model separates on feature[0] > 0.5; feature[1] has no effect.
        //   [0.0, 0.0] → proba ≈ 0.0004  (class 0 region)
        //   [1.0, 0.0] → proba ≈ 0.9966  (class 1 region)
        //   [0.3, 0.7] → proba ≈ 0.0008
        test_cases.push(vec![0.0, 0.0]);
        test_cases.push(vec![1.0, 0.0]);
        test_cases.push(vec![0.3, 0.7]);
    }

    // -----------------------------------------------------------------------
    // For each test point: client encrypts → server evaluates → client decrypts
    // -----------------------------------------------------------------------
    println!(
        "  {:<22}  {:<12}  {:<12}  {:<12}  {:<12}  {}",
        "features", "plain_raw", "plain_proba", "fhe_raw", "fhe_proba", "FHE latency"
    );
    println!("  {}", "-".repeat(88));

    for features in &test_cases {
        // Plaintext reference (both raw score and activated probability).
        let plain_raw: f32 = PlaintextEvaluator.predict(&model, features);
        let plain_proba: f32 = apply_activation(&model.objective, plain_raw);

        // Client: encrypt features with private key.
        let t_enc = Instant::now();
        let enc_input = client.encrypt(features);
        let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

        // Server: evaluate all trees in FHE — no plaintext data visible.
        let t_fhe = Instant::now();
        let enc_score = evaluator.predict(&model, &enc_input);
        let fhe_s = t_fhe.elapsed().as_secs_f64();

        // Client: decrypt raw score, then apply activation in plaintext.
        let t_dec = Instant::now();
        let fhe_raw: f32 = client.decrypt_score(&enc_score);
        let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        let fhe_proba: f32 = apply_activation(&model.objective, fhe_raw);

        let feature_str = format!(
            "[{}]",
            features
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "  {:<22}  {:<12.4}  {:<12.4}  {:<12.4}  {:<12.4}  {:.1} s",
            feature_str, plain_raw, plain_proba, fhe_raw, fhe_proba, fhe_s,
        );
        println!(
            "  {:<22}  (enc {enc_ms:.1} ms, FHE {fhe_s:.0} s, dec {dec_ms:.1} ms)",
            ""
        );
    }

    println!();
    println!("Key generation : {keygen_ms:.0} ms");
    println!();
    println!("Note: sigmoid / activation applied client-side on decrypted raw score.");
    Ok(())
}

/// Return the maximum depth of any tree in the ensemble (depth of a stump = 1).
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
