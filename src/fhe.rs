//! FHE context and encrypted evaluator.
//!
//! This module is only available with the `tfhe-backend` feature flag.
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

use tfhe::{
    ConfigBuilder, FheInt16, FheInt32, ServerKey,
    generate_keys, set_server_key,
};
use tfhe::prelude::*;

use crate::{
    Error,
    eval::Evaluator,
    model::Ensemble,
};

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
/// # #[cfg(feature = "tfhe-backend")]
/// # {
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
/// # }
/// # Ok::<(), weirwood::Error>(())
/// ```
pub struct FheContext {
    pub(crate) client_key: tfhe::ClientKey,
    pub(crate) server_key: ServerKey,
}

impl FheContext {
    /// Generate a fresh keypair with default 128-bit security parameters.
    pub fn generate() -> Result<Self, Error> {
        let config = ConfigBuilder::default().build();
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
                let scaled = (v * SCALE)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
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
/// [`EncryptedScore`] that can be decrypted with [`FheContext::decrypt_score`].
///
/// **Not yet implemented.** Use [`crate::eval::PlaintextEvaluator`] for
/// inference until the circuit translation is complete.
pub struct FheEvaluator {
    #[allow(dead_code)]
    ctx: FheContext,
}

impl FheEvaluator {
    pub fn new(ctx: FheContext) -> Self {
        FheEvaluator { ctx }
    }
}

impl Evaluator for FheEvaluator {
    type Input = EncryptedInput;
    type Output = EncryptedScore;

    fn predict(&self, _ensemble: &Ensemble, _input: &EncryptedInput) -> EncryptedScore {
        todo!("FHE inference circuit not yet implemented")
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
        let ciphertext = ctx.encrypt(&[original]);

        // Decryption also uses the private ClientKey.
        // The server never sees this step.
        let raw: i16 = ciphertext[0].decrypt(&ctx.client_key);
        let recovered = raw as f32 / SCALE;

        // The expected decoded value is the rounded fixed-point representation,
        // not necessarily `original` exactly.
        let expected = (original * SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16 as f32 / SCALE;

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
        let ctx = FheContext::generate().unwrap();
        let features: &[f32] = &[35.0, 0.72, -0.15, 3.20];
        let ciphertext = ctx.encrypt(features);

        for (i, &original) in features.iter().enumerate() {
            let raw: i16 = ciphertext[i].decrypt(&ctx.client_key);
            let recovered = raw as f32 / SCALE;
            approx::assert_abs_diff_eq!(recovered, original, epsilon = FIXED_POINT_EPSILON + 1e-6);
        }
    }

    #[test]
    fn round_trip_negative_features() {
        let ctx = FheContext::generate().unwrap();
        for &v in &[-1.0_f32, -0.5, -100.0, -327.0] {
            assert_round_trip(&ctx, v);
        }
    }

    #[test]
    fn round_trip_zero() {
        let ctx = FheContext::generate().unwrap();
        assert_round_trip(&ctx, 0.0);
    }

    #[test]
    fn round_trip_positive_features() {
        let ctx = FheContext::generate().unwrap();
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
        let ctx = FheContext::generate().unwrap();

        let ciphertext = ctx.encrypt(&[1.234]);
        let raw: i16 = ciphertext[0].decrypt(&ctx.client_key);
        let recovered = raw as f32 / SCALE;

        // 1.234 * 100 = 123.4 → rounds to 123 → 1.23
        approx::assert_abs_diff_eq!(recovered, 1.23, epsilon = 1e-6);
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
        let ctx_alice = FheContext::generate().unwrap();
        let ctx_bob = FheContext::generate().unwrap();

        let original: i16 = 42; // scaled value
        let ciphertext = FheInt16::encrypt(original, &ctx_alice.client_key);

        let decrypted_by_alice: i16 = ciphertext.decrypt(&ctx_alice.client_key);
        let decrypted_by_bob: i16 = ciphertext.decrypt(&ctx_bob.client_key);

        assert_eq!(decrypted_by_alice, original, "alice should recover her own ciphertext");
        assert_ne!(
            decrypted_by_bob, original,
            "bob's key should not decrypt alice's ciphertext \
             (collision negligible with 128-bit security)"
        );
    }
}
