//! End-to-end FHE inference example on the stump regression fixture.
//!
//! Demonstrates the full privacy-preserving evaluation flow:
//!   1. Client generates FHE key material.
//!   2. Client encrypts a feature vector with their private key.
//!   3. Server evaluates the XGBoost stump entirely in FHE — it never sees
//!      plaintext features or the score.
//!   4. Client decrypts the result.
//!
//! The plaintext result from `PlaintextEvaluator` is shown alongside for
//! comparison.  The two scores must agree within fixed-point rounding error
//! (at most ±0.01 with SCALE=100).
//!
//! Model: single decision stump — feature[0] ≤ 1.5 → leaf −0.5, else +0.5;
//!        base_score = 1.0, so expected raw scores are 0.5 (left) and 1.5 (right).
//!
//! WARNING: FHE key generation takes ~1–3 s and inference ~2–8 min on CPU.
//!          Run in release mode:
//!
//! ```sh
//! cargo run --release --example fhe_stump_inference
//! cargo run --release --example fhe_stump_inference -- tests/fixtures/stump_regression.json
//! ```

use std::time::Instant;

use weirwood::{
    eval::{Evaluator as _, PlaintextEvaluator},
    fhe::{FheContext, FheEvaluator},
    model::WeirwoodTree,
};

const DEFAULT_MODEL: &str = "tests/fixtures/stump_regression.json";

// Feature vectors to probe: left branch, exact threshold, right branch.
const TEST_CASES: &[f32] = &[0.0, 1.5, 2.0];

fn main() -> Result<(), weirwood::Error> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    let model = WeirwoodTree::from_json_file(&model_path)?;

    println!("weirwood · FHE stump inference");
    println!("  model : {model_path}");
    println!(
        "  trees : {}   features : {}   objective : {:?}",
        model.trees.len(),
        model.num_features,
        model.objective,
    );
    println!();

    // -----------------------------------------------------------------------
    // Client: key generation
    // -----------------------------------------------------------------------
    print!("Generating FHE keys … ");
    std::io::Write::flush(&mut std::io::stdout()).ok();
    let t_keygen = Instant::now();
    let ctx = FheContext::generate()?;
    let keygen_ms = t_keygen.elapsed().as_secs_f64() * 1000.0;
    println!("{keygen_ms:.0} ms");
    println!();

    // Install the server key on this thread (in a real deployment the server
    // would receive only server_key() — the client key stays local).
    ctx.set_active();
    let evaluator = FheEvaluator::new(ctx);

    // -----------------------------------------------------------------------
    // For each test point: encrypt → FHE predict → decrypt, compare plaintext
    // -----------------------------------------------------------------------
    println!(
        "  {:<12}  {:<14}  {:<14}  {:<12}  {}",
        "feature[0]", "plaintext", "FHE result", "FHE latency", "match?"
    );
    println!("  {}", "-".repeat(70));

    for &v in TEST_CASES {
        let features = vec![v];

        // Plaintext reference
        let plain_score = PlaintextEvaluator.predict(&model, &features);

        // Client: encrypt
        let t_enc = Instant::now();
        let enc_input = evaluator.encrypt(&features);
        let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

        // Server: FHE evaluation
        let t_fhe = Instant::now();
        let enc_score = evaluator.predict(&model, &enc_input);
        let fhe_s = t_fhe.elapsed().as_secs_f64();

        // Client: decrypt
        let t_dec = Instant::now();
        let fhe_score = evaluator.decrypt_score(&enc_score);
        let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

        let delta = (fhe_score - plain_score).abs();
        let ok = if delta < 0.02 { "OK" } else { "MISMATCH" };

        println!(
            "  {:<12.2}  {:<14.4}  {:<14.4}  {:<12.2}  {} (|Δ|={:.4})",
            v,
            plain_score,
            fhe_score,
            format!("{fhe_s:.1} s"),
            ok,
            delta,
        );

        println!(
            "             (enc {enc_ms:.1} ms, FHE {:.0} s, dec {dec_ms:.1} ms)",
            fhe_s,
        );
    }

    println!();
    println!("Key generation : {keygen_ms:.0} ms");
    Ok(())
}
