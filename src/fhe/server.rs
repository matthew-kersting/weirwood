//! Server-side FHE context: holds only the server key.
//!
//! [`ServerContext`] is the only thing the inference server needs.  It
//! contains the [`ServerKey`] required to perform homomorphic operations, but
//! **no private key material** — so sharing it with the server does not
//! compromise input privacy.
//!
//! In a real deployment the client serializes this context and sends it to the
//! server once; the server installs it via [`ServerContext::set_active`] and
//! then evaluates any number of encrypted inputs.
//!
//! [`ServerKey`]: tfhe::ServerKey

use tfhe::{ServerKey, set_server_key};

// ---------------------------------------------------------------------------
// ServerContext
// ---------------------------------------------------------------------------

/// Server-side FHE context — contains only the [`ServerKey`], no private key.
///
/// Obtain a `ServerContext` from [`ClientContext::server_context`]; do not
/// construct one directly.  Pass it to [`FheEvaluator::new`] to create an
/// evaluator that can run inference on encrypted inputs.
///
/// [`ClientContext::server_context`]: super::client::ClientContext::server_context
/// [`FheEvaluator::new`]: super::evaluator::FheEvaluator::new
/// [`ServerKey`]: tfhe::ServerKey
pub struct ServerContext {
    pub(crate) server_key: ServerKey,
}

impl ServerContext {
    /// Construct from an owned [`ServerKey`].
    ///
    /// This is `pub(crate)` — callers obtain a `ServerContext` via
    /// [`ClientContext::server_context`], which enforces the key-separation
    /// invariant.
    ///
    /// [`ClientContext::server_context`]: super::client::ClientContext::server_context
    pub(crate) fn from_key(server_key: ServerKey) -> Self {
        ServerContext { server_key }
    }

    /// Install the server key as the thread-local active key for TFHE-rs
    /// operations.
    ///
    /// Must be called on the thread that will run inference before invoking
    /// [`FheEvaluator::predict`].  Each call clones the server key internally
    /// (unavoidable due to the tfhe-rs API); avoid calling this in a hot loop.
    ///
    /// [`FheEvaluator::predict`]: super::evaluator::FheEvaluator::predict
    pub fn set_active(&self) {
        set_server_key(self.server_key.clone());
    }
}
