//! End-to-end FHE inference example on the stump regression fixture.
//!
//! Demonstrates the full privacy-preserving evaluation flow across the two
//! logical parties:
//!
//!   **Client**
//!   1. Generates FHE key material (`ClientContext::generate`).
//!   2. Extracts a `ServerContext` (server key only) to share with the server.
//!   3. Encrypts a feature vector with the private key.
//!   4. Decrypts the inference result returned by the server.
//!
//!   **Server**
//!   5. Creates an `FheEvaluator` (installs the server key on worker threads).
//!   6. Evaluates the XGBoost stump entirely in FHE — it never sees
//!      plaintext features or the score.
//!
//! In this single-process demo both parties run in the same `main`, but the
//! types make the boundary explicit: `FheEvaluator` holds only a
//! `ServerContext` and cannot decrypt anything.
//!
//! The plaintext result from `PlaintextEvaluator` is shown alongside for
//! comparison.  The two scores must agree within fixed-point rounding error
//! (at most ±0.01 with SCALE=100).
//!
//! Model: single decision stump — feature[0] ≤ 1.5 → leaf −0.5, else +0.5;
//!        base_score = 1.0, so expected raw scores are 0.5 (left) and 1.5 (right).
//!
//! WARNING: FHE key generation takes ~1 s; each inference costs 1 PBS op
//!          (~410 ms on CPU), so 3 test cases ≈ 2 s of FHE time.
//!          Run in release mode:
//!
//! ```sh
//! cargo run --release --example fhe_stump_inference
//! cargo run --release --example fhe_stump_inference -- tests/fixtures/stump_regression.json
//! ```

use std::time::Instant;

use weirwood::{
    eval::{Evaluator as _, PlaintextEvaluator},
    fhe::{ClientContext, FheEvaluator},
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
    let client = ClientContext::generate()?;
    let keygen_ms = t_keygen.elapsed().as_secs_f64() * 1000.0;
    println!("{keygen_ms:.0} ms");
    println!();

    // Client extracts the server context (ServerKey only — no private key).
    // In a real deployment this is what gets sent over the wire to the server.
    let server_ctx = client.server_context();

    // -----------------------------------------------------------------------
    // Server setup
    // -----------------------------------------------------------------------
    server_ctx.set_active(); // install key on the calling thread (worker threads get it via start_handler)
    let evaluator = FheEvaluator::new(server_ctx);

    // -----------------------------------------------------------------------
    // For each test point: client encrypts → server evaluates → client decrypts
    // -----------------------------------------------------------------------
    println!(
        "  {:<12}  {:<14}  {:<14}  {:<12}  {}",
        "feature[0]", "plaintext", "FHE result", "FHE latency", "match?"
    );
    println!("  {}", "-".repeat(70));

    for &v in TEST_CASES {
        let features = vec![v];

        // Plaintext reference (client-side)
        let plain_score = PlaintextEvaluator.predict(&model, &features);

        // Client: encrypt
        let t_enc = Instant::now();
        let enc_input = client.encrypt(&features);
        let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

        // Server: FHE evaluation
        let t_fhe = Instant::now();
        let enc_score = evaluator.predict(&model, &enc_input);
        let fhe_s = t_fhe.elapsed().as_secs_f64();

        // Client: decrypt
        let t_dec = Instant::now();
        let fhe_score = client.decrypt_score(&enc_score);
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
