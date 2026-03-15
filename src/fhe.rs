//! FHE context and encrypted evaluator.
//!
//! # Key management model
//!
//! TFHE-rs uses a *symmetric* key model under the hood, but the roles map
//! naturally onto the public-key mental model:
//!
//! | Concept         | TFHE-rs type  | Who holds it          |
//! |-----------------|---------------|-----------------------|
//! | Private key     | `ClientKey`   | Client only           |
//! | Public key      | `ServerKey`   | Server (or anyone)    |
//!
//! The `ClientKey` is used for both **encryption** and **decryption**.
//! The `ServerKey` is used for **homomorphic operations** (future inference).
//! In a real deployment the client keeps the `ClientKey` locally and only
//! sends the `ServerKey` plus the encrypted input to the server.
//!
//! # Encoding
//!
//! `f32` features are encoded as fixed-point `i16` values scaled by
//! [`SCALE`] before encryption.  This gives two decimal places of precision
//! and supports feature values in the range `[-327.0, 327.0]`.  The same
//! scale factor is applied to leaf values by the future [`FheEvaluator`], so
//! [`FheContext::decrypt_score`] simply divides the decrypted integer by
//! [`SCALE`] to recover the original float range.
//!
//! # Status
//!
//! Key generation, encryption, and decryption are complete.
//! The encrypted evaluator ([`FheEvaluator`]) is stubbed — it will implement
//! [`crate::eval::Evaluator`] once the circuit translation is finished.
//! Use [`crate::eval::PlaintextEvaluator`] for inference today.

use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheInt16, FheInt32, ServerKey, generate_keys, set_server_key};

use crate::{Error, eval::Evaluator, model::WeirwoodTree};

// ---------------------------------------------------------------------------
// Constants and type aliases
// ---------------------------------------------------------------------------

/// Fixed-point scale factor applied to `f32` features before encryption.
///
/// An `f32` value `v` is stored as `round(v * SCALE)` clamped to `i16`.
/// The [`FheEvaluator`] must scale plaintext leaf values by the same factor
/// so that [`FheContext::decrypt_score`] produces the correct result.
pub const SCALE: f32 = 100.0;

/// An encrypted feature vector produced by [`FheContext::encrypt`].
///
/// Each element is an `FheInt16` representing one feature scaled by [`SCALE`].
pub type EncryptedInput = Vec<FheInt16>;

/// An encrypted raw ensemble score produced by [`FheEvaluator`].
///
/// Stored as a scaled `FheInt32`; decrypt and divide by [`SCALE`] to
/// recover the original float score.
pub type EncryptedScore = FheInt32;

// ---------------------------------------------------------------------------
// FheContext
// ---------------------------------------------------------------------------

/// Holds the FHE key material and scheme configuration.
///
/// # Example
///
/// ```no_run
/// use weirwood::fhe::FheContext;
///
/// let ctx = FheContext::generate()?;
/// ctx.set_active();
///
/// let features = vec![1.5_f32, 0.3, -2.1];
/// let ciphertext = ctx.encrypt(&features);
///
/// // ... send server_key() and ciphertext to the inference server ...
///
/// // let score: EncryptedScore = server.run_inference(...);
/// // let result = ctx.decrypt_score(&score);
/// # Ok::<(), weirwood::Error>(())
/// ```
pub struct FheContext {
    pub(crate) client_key: tfhe::ClientKey,
    pub(crate) server_key: ServerKey,
}

impl FheContext {
    /// Generate a fresh keypair with default 128-bit security parameters.
    pub fn generate() -> Result<Self, Error> {
        let config: tfhe::Config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);
        Ok(FheContext {
            client_key,
            server_key,
        })
    }

    /// Return a clone of the server key.
    ///
    /// This is the "public" component: share it with the inference server
    /// while keeping the `FheContext` (and its `ClientKey`) local.
    pub fn server_key(&self) -> ServerKey {
        self.server_key.clone()
    }

    /// Install the server key as the thread-local active key for TFHE-rs
    /// operations.  Must be called on the thread that will run inference
    /// before invoking [`FheEvaluator::predict`].
    pub fn set_active(&self) {
        set_server_key(self.server_key.clone());
    }

    /// Encrypt a plaintext feature vector using the private key.
    ///
    /// Each `f32` is multiplied by [`SCALE`], rounded, clamped to `i16`,
    /// and then encrypted.  The resulting [`EncryptedInput`] can be sent to
    /// the inference server alongside the [`server_key`](Self::server_key).
    pub fn encrypt(&self, features: &[f32]) -> EncryptedInput {
        features
            .iter()
            .map(|&v| {
                let scaled = (v * SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                FheInt16::encrypt(scaled, &self.client_key)
            })
            .collect()
    }

    /// Decrypt an encrypted ensemble score back to `f32`.
    ///
    /// The [`FheEvaluator`] is expected to produce leaf-value sums where every
    /// leaf is scaled by [`SCALE`], so this method divides the decrypted
    /// integer by `SCALE` to recover the original float range.
    pub fn decrypt_score(&self, score: &EncryptedScore) -> f32 {
        let raw: i32 = score.decrypt(&self.client_key);
        raw as f32 / SCALE
    }
}

// ---------------------------------------------------------------------------
// FheEvaluator
// ---------------------------------------------------------------------------

/// Encrypted evaluator — runs XGBoost inference entirely in FHE.
///
/// Accepts an [`EncryptedInput`] from [`FheContext::encrypt`] and returns an
/// [`EncryptedScore`] that can be decrypted with [`FheContext::decrypt_score`]
/// or the convenience wrapper [`FheEvaluator::decrypt_score`].
pub struct FheEvaluator {
    pub(crate) ctx: FheContext,
}

impl FheEvaluator {
    pub fn new(ctx: FheContext) -> Self {
        FheEvaluator { ctx }
    }

    /// Decrypt an encrypted score produced by [`Self::predict`].
    ///
    /// Delegates to [`FheContext::decrypt_score`].  In a real deployment only
    /// the client (who holds the private key) can call this.
    pub fn decrypt_score(&self, score: &EncryptedScore) -> f32 {
        self.ctx.decrypt_score(score)
    }
}

// ---------------------------------------------------------------------------
// FHE circuit helpers
// ---------------------------------------------------------------------------

/// Recursively evaluate one tree node in FHE, returning a scaled `FheInt16`.
///
/// Internal nodes perform a bootstrapped comparison between an encrypted
/// feature and a plaintext threshold, then use `if_then_else` to select the
/// left or right sub-result.  Leaf nodes return a trivially-encrypted
/// (unrandomised) constant — the scaled leaf weight.
///
/// The return type is `FheInt16` rather than `FheInt32` because
/// `FheBool::if_then_else` on 32-bit ciphertexts triggers an integer overflow
/// inside tfhe-rs's PBS accumulator size calculation (the 16-block radix
/// representation overflows a usize intermediate).  Single-tree leaf values
/// scaled by [`SCALE`] easily fit in `i16` (range ±327), so this is safe.
/// The caller widens to `FheInt32` before accumulating across trees.
fn eval_node(tree: &crate::model::Tree, node_idx: usize, features: &EncryptedInput) -> FheInt16 {
    let node: &crate::model::Node = &tree.nodes[node_idx];
    if node.is_leaf() {
        // Trivially encrypt the scaled leaf weight.  No secret material is
        // involved; this just wraps the plaintext in the ciphertext format so
        // it can be combined with real ciphertexts via homomorphic operations.
        let scaled: i16 = (node.leaf_value * SCALE).round() as i16;
        FheInt16::encrypt_trivial(scaled)
    } else {
        // Scale the plaintext threshold to match the fixed-point encoding of
        // the encrypted features.
        let threshold: i16 = (node.split_threshold * SCALE).round() as i16;
        // Wrap the threshold as a trivial ciphertext so the comparison uses
        // the ciphertext-vs-ciphertext path (comparison.rs).  The scalar path
        // (scalar_comparison.rs) has a bug in tfhe-rs 0.10.0: its borrow-
        // propagation state_fn returns values up to 124, and 124 × delta
        // (= 124 × 2^59) overflows u64 in fill_accumulator.  The ciphertext
        // path keeps state values ≤ 16, so 16 × 2^59 = 2^63 stays in range.
        let threshold_ct: FheInt16 = FheInt16::encrypt_trivial(threshold);
        // One programmable-bootstrapping comparison: encrypted feature vs
        // trivially-encrypted threshold.  Returns FheBool (encrypted 0 or 1).
        let go_left: tfhe::FheBool = features[node.split_feature as usize].le(&threshold_ct);
        let left_score: FheInt16 = eval_node(tree, node.left_child as usize, features);
        let right_score: FheInt16 = eval_node(tree, node.right_child as usize, features);
        // Oblivious mux: selects left_score when go_left=1, right_score
        // otherwise, without revealing the branch taken.
        go_left.if_then_else(&left_score, &right_score)
    }
}

impl Evaluator for FheEvaluator {
    type Input = EncryptedInput;
    type Output = EncryptedScore;

    /// Run XGBoost inference over an encrypted feature vector.
    ///
    /// Each tree is evaluated by recursively applying bootstrapped comparisons
    /// and oblivious muxes (see [`eval_node`]).  Tree scores are accumulated
    /// into a single [`EncryptedScore`] together with the scaled `base_score`.
    ///
    /// The server key must be installed via [`FheContext::set_active`] on the
    /// calling thread before this method is invoked.
    fn predict(
        &self,
        weirwood_tree: &WeirwoodTree,
        encrypted_features: &EncryptedInput,
    ) -> EncryptedScore {
        // Start with the scaled base_score as a trivial ciphertext so that
        // subsequent additions stay in the same ciphertext domain.
        let base_scaled = (weirwood_tree.base_score * SCALE).round() as i32;
        let mut total: FheInt32 = FheInt32::encrypt_trivial(base_scaled);

        for tree in &weirwood_tree.trees {
            // eval_node returns FheInt16; widen before adding to the i32 accumulator.
            let tree_score: FheInt16 = eval_node(tree, 0, encrypted_features);
            let tree_score_i32: FheInt32 = FheInt32::cast_from(tree_score);
            total = total + tree_score_i32;
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Key management model (for readers of these tests)
    //
    // PRIVATE — FheContext.client_key
    //   Used to encrypt inputs and decrypt outputs.
    //   Never leaves the client.
    //
    // PUBLIC  — FheContext.server_key()
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
    fn assert_round_trip(ctx: &FheContext, original: f32) {
        // Encryption uses the private ClientKey.
        let ciphertext: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = ctx.encrypt(&[original]);

        // Decryption also uses the private ClientKey.
        // The server never sees this step.
        let raw: i16 = ciphertext[0].decrypt(&ctx.client_key);
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
        let ctx: FheContext = FheContext::generate().unwrap();
        let features: &[f32] = &[35.0, 0.72, -0.15, 3.20];
        let ciphertext: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = ctx.encrypt(features);

        for (i, &original) in features.iter().enumerate() {
            let raw: i16 = ciphertext[i].decrypt(&ctx.client_key);
            let recovered = raw as f32 / SCALE;
            approx::assert_abs_diff_eq!(recovered, original, epsilon = FIXED_POINT_EPSILON + 1e-6);
        }
    }

    #[test]
    fn round_trip_negative_features() {
        let ctx: FheContext = FheContext::generate().unwrap();
        for &v in &[-1.0_f32, -0.5, -100.0, -327.0] {
            assert_round_trip(&ctx, v);
        }
    }

    #[test]
    fn round_trip_zero() {
        let ctx: FheContext = FheContext::generate().unwrap();
        assert_round_trip(&ctx, 0.0);
    }

    #[test]
    fn round_trip_positive_features() {
        let ctx: FheContext = FheContext::generate().unwrap();
        for &v in &[0.01_f32, 1.0, 50.5, 327.0] {
            assert_round_trip(&ctx, v);
        }
    }

    // -----------------------------------------------------------------------
    // Fixed-point precision
    // -----------------------------------------------------------------------

    #[test]
    fn fixed_point_rounding_within_one_ulp() {
        // A value with sub-cent precision (1.234) should decode as 1.23,
        // not 1.234 — the third decimal is lost in encoding.
        let ctx: FheContext = FheContext::generate().unwrap();

        let ciphertext: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = ctx.encrypt(&[1.234]);
        let raw: i16 = ciphertext[0].decrypt(&ctx.client_key);
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

        let ctx: FheContext = FheContext::generate().unwrap();
        ctx.set_active();

        let model: WeirwoodTree = WeirwoodTree::from_json_file("tests/fixtures/stump_regression.json").unwrap();
        let evaluator: FheEvaluator = FheEvaluator::new(ctx);

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

            let plaintext_score: f32 = PlaintextEvaluator.predict(&model, &features);

            let encrypted_input: Vec<tfhe::FheInt<tfhe::FheInt16Id>> = evaluator.ctx.encrypt(&features);
            let encrypted_score: tfhe::FheInt<tfhe::FheInt32Id> = evaluator.predict(&model, &encrypted_input);
            let fhe_score: f32 = evaluator.decrypt_score(&encrypted_score);

            approx::assert_abs_diff_eq!(fhe_score, plaintext_score, epsilon = CIRCUIT_EPSILON);
        }
    }

    // -----------------------------------------------------------------------
    // Key separation: a different private key cannot decrypt the ciphertext
    // -----------------------------------------------------------------------

    #[test]
    fn wrong_private_key_gives_garbage() {
        // Simulate two independent users.  ctx_alice holds the private key
        // used to encrypt; ctx_bob is a completely unrelated key pair.
        // Decrypting alice's ciphertext with bob's private key must not
        // recover the original value.
        let ctx_alice: FheContext = FheContext::generate().unwrap();
        let ctx_bob: FheContext = FheContext::generate().unwrap();

        let original: i16 = 42; // scaled value
        let ciphertext: tfhe::FheInt<tfhe::FheInt16Id> =
            FheInt16::encrypt(original, &ctx_alice.client_key);

        let decrypted_by_alice: i16 = ciphertext.decrypt(&ctx_alice.client_key);
        let decrypted_by_bob: i16 = ciphertext.decrypt(&ctx_bob.client_key);

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
