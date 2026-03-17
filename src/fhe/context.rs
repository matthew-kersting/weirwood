//! FHE key management and fixed-point encoding.
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
//! The `ServerKey` is used for **homomorphic operations** (inference).
//! In a real deployment the client keeps the `ClientKey` locally and only
//! sends the `ServerKey` plus the encrypted input to the server.
//!
//! # Encoding
//!
//! `f32` features are encoded as fixed-point `i16` values scaled by
//! [`SCALE`] before encryption.  This gives two decimal places of precision
//! and supports feature values in the range `[-327.0, 327.0]`.  The same
//! scale factor is applied to leaf values by [`FheEvaluator`], so
//! [`FheContext::decrypt_score`] simply divides the decrypted integer by
//! [`SCALE`] to recover the original float range.

use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheInt16, ServerKey, generate_keys, set_server_key};

use crate::Error;

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
pub type EncryptedScore = tfhe::FheInt32;

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
