mod client;
mod evaluator;
mod server;

pub use client::{ClientContext, EncryptedInput, EncryptedScore, SCALE};
pub use evaluator::FheEvaluator;
pub use server::ServerContext;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tfhe::prelude::*;

    // -----------------------------------------------------------------------
    // Key management model (for readers of these tests)
    //
    // PRIVATE — ClientContext.client_key
    //   Used to encrypt inputs and decrypt outputs.
    //   Never leaves the client.
    //
    // PUBLIC  — ServerContext (obtained via ClientContext::server_context())
    //   Used by the inference server to perform homomorphic operations.
    //   Safe to share; reveals nothing about plaintexts or the client key.
    //
    // FHE key generation is intentionally expensive (~1-3 s per call).
    // -----------------------------------------------------------------------

    /// Maximum rounding error introduced by the fixed-point encoding.
    /// Any `f32` is stored as `round(v * SCALE) / SCALE`, so the error
    /// is at most half a unit of least precision: `0.5 / SCALE`.
    const FIXED_POINT_EPSILON: f32 = 0.5 / SCALE;

    /// Encrypt a single feature with the **private** key, then decrypt it
    /// with the same private key, and assert the value is recovered within
    /// fixed-point rounding error.
    fn assert_round_trip(client: &ClientContext, original: f32) {
        // Encryption uses the private ClientKey.
        let ciphertext: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = client.encrypt(&[original]);

        // Decryption also uses the private ClientKey.
        // The server never sees this step.
        let raw: i16 = ciphertext[0].decrypt(&client.client_key);
        let recovered: f32 = raw as f32 / SCALE;

        // The expected decoded value is the rounded fixed-point representation,
        // not necessarily `original` exactly.
        let expected: f32 = (original * SCALE)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16 as f32
            / SCALE;

        approx::assert_abs_diff_eq!(recovered, expected, epsilon = 1e-6);
        // Also sanity-check against the original within the rounding bound.
        approx::assert_abs_diff_eq!(recovered, original, epsilon = FIXED_POINT_EPSILON + 1e-6);
    }

    // -----------------------------------------------------------------------
    // Round-trip: encrypt with private key, decrypt with private key
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_typical_inference_vector() {
        // Represents a realistic feature vector passed to an XGBoost model
        // (e.g. age=35, income=0.72 normalised, debt_ratio=-0.15, score=320.0).
        let client: ClientContext = ClientContext::generate().unwrap();
        let features: &[f32] = &[35.0, 0.72, -0.15, 3.20];
        let ciphertext: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = client.encrypt(features);

        for (i, &original) in features.iter().enumerate() {
            let raw: i16 = ciphertext[i].decrypt(&client.client_key);
            let recovered = raw as f32 / SCALE;
            approx::assert_abs_diff_eq!(recovered, original, epsilon = FIXED_POINT_EPSILON + 1e-6);
        }
    }

    #[test]
    fn round_trip_negative_features() {
        let client: ClientContext = ClientContext::generate().unwrap();
        for &v in &[-1.0_f32, -0.5, -100.0, -327.0] {
            assert_round_trip(&client, v);
        }
    }

    #[test]
    fn round_trip_zero() {
        let client: ClientContext = ClientContext::generate().unwrap();
        assert_round_trip(&client, 0.0);
    }

    #[test]
    fn round_trip_positive_features() {
        let client: ClientContext = ClientContext::generate().unwrap();
        for &v in &[0.01_f32, 1.0, 50.5, 327.0] {
            assert_round_trip(&client, v);
        }
    }

    // -----------------------------------------------------------------------
    // Fixed-point precision
    // -----------------------------------------------------------------------

    #[test]
    fn fixed_point_rounding_within_one_ulp() {
        // A value with sub-cent precision (1.234) should decode as 1.23,
        // not 1.234 — the third decimal is lost in encoding.
        let client: ClientContext = ClientContext::generate().unwrap();

        let ciphertext: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = client.encrypt(&[1.234]);
        let raw: i16 = ciphertext[0].decrypt(&client.client_key);
        let recovered: f32 = raw as f32 / SCALE;

        // 1.234 * 100 = 123.4 → rounds to 123 → 1.23
        approx::assert_abs_diff_eq!(recovered, 1.23, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // FHE circuit evaluation: FheEvaluator::predict matches PlaintextEvaluator
    // -----------------------------------------------------------------------

    /// Maximum rounding error for a single accumulated term (leaf or base_score).
    /// Each f32 is quantised to `round(v * SCALE) / SCALE`, so the error per
    /// term is at most `0.5 / SCALE`.  Two terms (leaf + base_score) gives
    /// the bound used below.
    const CIRCUIT_EPSILON: f32 = 2.0 * (0.5 / SCALE);

    /// Verify that `FheEvaluator::predict` agrees with `PlaintextEvaluator`
    /// on the stump regression fixture (threshold 1.5, leaves ±0.5, base 1.0).
    ///
    /// This test performs real TFHE bootstrapping and takes ~5–30 s depending
    /// on the machine.  It is the integration proof that the FHE circuit
    /// correctly implements the decision-tree evaluation.
    #[test]
    fn fhe_stump_matches_plaintext() {
        use crate::eval::{Evaluator as _, PlaintextEvaluator};
        use crate::model::WeirwoodTree;

        // --- Client ---
        let client: ClientContext = ClientContext::generate().unwrap();
        let server_ctx: ServerContext = client.server_context();

        // Install the server key on the calling thread; FheEvaluator::new installs
        // it on worker threads via start_handler.
        server_ctx.set_active();

        // --- Server setup ---
        let evaluator: FheEvaluator = FheEvaluator::new(server_ctx);

        let model: WeirwoodTree =
            WeirwoodTree::from_json_file("tests/fixtures/stump_regression.json").unwrap();

        // Probe both branches and the exact boundary.
        // stump: feature[0] <= 1.5 → left (leaf = -0.5), else right (leaf = 0.5)
        // base_score = 1.0  →  expected scores: 0.5 (left) and 1.5 (right)
        let test_cases: &[f32] = &[
            0.0, // clearly left  → plaintext 0.5
            1.5, // at threshold  → plaintext 0.5  (≤ goes left)
            2.0, // clearly right → plaintext 1.5
        ];

        for &feature_val in test_cases {
            let features: Vec<f32> = vec![feature_val];

            // --- Client: plaintext reference and encryption ---
            let plaintext_score: f32 = PlaintextEvaluator.predict(&model, &features);
            let encrypted_input: EncryptedInput = client.encrypt(&features);

            // --- Server: FHE evaluation ---
            let encrypted_score: EncryptedScore = evaluator.predict(&model, &encrypted_input);

            // --- Client: decryption and comparison ---
            let fhe_score: f32 = client.decrypt_score(&encrypted_score);

            approx::assert_abs_diff_eq!(fhe_score, plaintext_score, epsilon = CIRCUIT_EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // FHE circuit evaluation: multi-tree ensemble
    // -----------------------------------------------------------------------

    /// Verify that `FheEvaluator::predict` correctly accumulates scores across
    /// multiple trees on the `two_trees_binary.json` fixture.
    ///
    /// Model layout:
    ///   Tree 1: feature[0] <= 1.5 → left(-0.3), right(+0.3)
    ///   Tree 2: feature[1] <= 2.0 → left(-0.2), right(+0.2)
    ///   base_score = 0.0
    ///
    /// Expected raw scores:
    ///   [0.0, 0.0] → -0.3 + -0.2 = -0.5
    ///   [2.0, 3.0] →  0.3 +  0.2 =  0.5
    ///   [0.0, 3.0] → -0.3 +  0.2 = -0.1
    ///   [2.0, 0.0] →  0.3 + -0.2 =  0.1
    ///
    /// Rounding error per tree is at most 0.5/SCALE; with 2 trees and no
    /// base_score the bound is 2 * 0.5/SCALE = 0.01.
    #[test]
    fn fhe_two_trees_matches_plaintext() {
        use crate::eval::{Evaluator as _, PlaintextEvaluator};
        use crate::model::WeirwoodTree;

        // epsilon: 2 leaf quantizations, no base_score contribution
        const EPSILON: f32 = 2.0 * (0.5 / SCALE);

        // --- Client ---
        let client: ClientContext = ClientContext::generate().unwrap();
        let server_ctx: ServerContext = client.server_context();

        // Install the server key on the calling thread; FheEvaluator::new installs
        // it on worker threads via start_handler.
        server_ctx.set_active();

        // --- Server setup ---
        let evaluator: FheEvaluator = FheEvaluator::new(server_ctx);

        let model: WeirwoodTree =
            WeirwoodTree::from_json_file("tests/fixtures/two_trees_binary.json").unwrap();

        let test_cases: &[[f32; 2]] = &[
            [0.0, 0.0], // both left  → -0.5
            [2.0, 3.0], // both right → +0.5
            [0.0, 3.0], // left, right → -0.1
            [2.0, 0.0], // right, left → +0.1
        ];

        for features in test_cases {
            let features_vec: Vec<f32> = features.to_vec();
            let plaintext_score: f32 = PlaintextEvaluator.predict(&model, &features_vec);
            let encrypted_input: EncryptedInput = client.encrypt(&features_vec);
            let encrypted_score: EncryptedScore = evaluator.predict(&model, &encrypted_input);
            let fhe_score: f32 = client.decrypt_score(&encrypted_score);

            approx::assert_abs_diff_eq!(fhe_score, plaintext_score, epsilon = EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // Key separation: a different private key cannot decrypt the ciphertext
    // -----------------------------------------------------------------------

    #[test]
    fn wrong_private_key_gives_garbage() {
        // Simulate two independent users.  client_alice holds the private key
        // used to encrypt; client_bob is a completely unrelated key pair.
        // Decrypting alice's ciphertext with bob's private key must not
        // recover the original value.
        let client_alice: ClientContext = ClientContext::generate().unwrap();
        let client_bob: ClientContext = ClientContext::generate().unwrap();

        let original: i16 = 42; // scaled value
        let ciphertext: tfhe::FheInt<tfhe::FheInt16Id> =
            tfhe::FheInt16::encrypt(original, &client_alice.client_key);

        let decrypted_by_alice: i16 = ciphertext.decrypt(&client_alice.client_key);
        let decrypted_by_bob: i16 = ciphertext.decrypt(&client_bob.client_key);

        assert_eq!(
            decrypted_by_alice, original,
            "alice should recover her own ciphertext"
        );
        assert_ne!(
            decrypted_by_bob, original,
            "bob's key should not decrypt alice's ciphertext \
             (collision negligible with 128-bit security)"
        );
    }
}
