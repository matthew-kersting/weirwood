//! Client-side FHE context: key generation, encryption, and decryption.
//!
//! # Role in the two-party model
//!
//! The client generates key material, encrypts feature vectors with the
//! private [`ClientKey`], and decrypts the inference result.  The private key
//! **never leaves the client**.  Only the [`ServerContext`] (which contains
//! the [`ServerKey`] alone) is shared with the inference server.
//!
//! ```text
//! ┌──────────── Client ─────────────┐        ┌───────── Server ──────────┐
//! │  ClientContext::generate()      │        │                           │
//! │      ├─ ClientKey  (private)    │        │  ServerContext             │
//! │      └─ ServerKey  ─────────────┼──────► │      └─ ServerKey         │
//! │                                 │        │                           │
//! │  client.encrypt(&features) ─────┼──────► │  FheEvaluator::predict()  │
//! │                                 │ ◄───── │                           │
//! │  client.decrypt_score(&score)   │        │                           │
//! └─────────────────────────────────┘        └───────────────────────────┘
//! ```
//!
//! # Encoding
//!
//! `f32` features are encoded as fixed-point `i16` values scaled by
//! [`SCALE`] before encryption.  This gives two decimal places of precision
//! and supports feature values in the range `[-327.0, 327.0]`.  The same
//! scale factor is applied to leaf values by [`FheEvaluator`], so
//! [`ClientContext::decrypt_score`] simply divides the decrypted integer by
//! [`SCALE`] to recover the original float range.
//!
//! [`FheEvaluator`]: super::evaluator::FheEvaluator
//! [`ServerContext`]: super::server::ServerContext
//! [`ClientKey`]: tfhe::ClientKey
//! [`ServerKey`]: tfhe::ServerKey

use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheInt16, generate_keys};

use crate::Error;

use super::server::ServerContext;

/// Fixed-point scale factor applied to `f32` features before encryption.
///
/// An `f32` value `v` is stored as `round(v * SCALE)` clamped to `i16`.
/// The [`FheEvaluator`](super::evaluator::FheEvaluator) must scale plaintext
/// leaf values by the same factor so that [`ClientContext::decrypt_score`]
/// produces the correct result.
pub const SCALE: f32 = 100.0;

/// An encrypted feature vector produced by [`ClientContext::encrypt`].
///
/// Each element is an `FheInt16` representing one feature scaled by [`SCALE`].
pub type EncryptedInput = Vec<FheInt16>;

/// An encrypted raw ensemble score produced by [`FheEvaluator`].
///
/// Stored as a scaled `FheInt32`; decrypt and divide by [`SCALE`] to
/// recover the original float score.
///
/// [`FheEvaluator`]: super::evaluator::FheEvaluator
pub type EncryptedScore = tfhe::FheInt32;

// ---------------------------------------------------------------------------
// ClientContext
// ---------------------------------------------------------------------------

/// Client-side FHE context — holds the private key and is never shared.
///
/// Responsible for key generation, feature encryption, and score decryption.
/// Call [`server_context`](Self::server_context) to obtain a [`ServerContext`]
/// that can safely be handed to the inference server.
///
/// # Example
///
/// ```no_run
/// use weirwood::fhe::{ClientContext, FheEvaluator};
/// use weirwood::eval::Evaluator as _;
/// use weirwood::model::WeirwoodTree;
///
/// // --- Client ---
/// let client = ClientContext::generate()?;
/// let server_ctx = client.server_context(); // only the ServerKey is shared
///
/// let model = WeirwoodTree::from_json_file("model.json")?;
/// let features = vec![1.5_f32, 0.3, -2.1];
/// let ciphertext = client.encrypt(&features);
///
/// // --- "Send server_ctx and ciphertext to the server" ---
///
/// // --- Server ---
/// let evaluator = FheEvaluator::new(server_ctx); // installs key on worker threads
/// let encrypted_score = evaluator.predict(&model, &ciphertext);
///
/// // --- "Send encrypted_score back to the client" ---
///
/// // --- Client ---
/// let score = client.decrypt_score(&encrypted_score);
/// # Ok::<(), weirwood::Error>(())
/// ```
pub struct ClientContext {
    pub(crate) client_key: tfhe::ClientKey,
    server_key: tfhe::ServerKey,
}

impl ClientContext {
    /// Generate a fresh keypair with default 128-bit security parameters.
    ///
    /// Key generation is intentionally expensive (~1–3 s).
    pub fn generate() -> Result<Self, Error> {
        let config: tfhe::Config = ConfigBuilder::default().build();
        let (client_key, server_key) = generate_keys(config);
        Ok(ClientContext {
            client_key,
            server_key,
        })
    }

    /// Extract the server-side context to share with the inference server.
    ///
    /// Clones the server key — this is a one-time, expected cost (~50–200 MB
    /// of key material).  The private client key is **not** included.
    ///
    /// [`ServerKey`]: tfhe::ServerKey
    pub fn server_context(&self) -> ServerContext {
        ServerContext::from_key(self.server_key.clone())
    }

    /// Encrypt a plaintext feature vector using the private key.
    ///
    /// Each `f32` is multiplied by [`SCALE`], rounded, clamped to `i16`,
    /// and then encrypted.  The resulting [`EncryptedInput`] can be sent to
    /// the inference server alongside the [`ServerContext`].
    ///
    /// [`ServerContext`]: super::server::ServerContext
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
    /// The [`FheEvaluator`](super::evaluator::FheEvaluator) produces leaf-value
    /// sums where every leaf is scaled by [`SCALE`], so this method divides the
    /// decrypted integer by [`SCALE`] to recover the original float range.
    pub fn decrypt_score(&self, score: &EncryptedScore) -> f32 {
        let raw: i32 = score.decrypt(&self.client_key);
        raw as f32 / SCALE
    }
}
