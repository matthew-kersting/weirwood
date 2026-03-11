//! FHE context and encrypted evaluator.
//!
//! This module is only available with the `tfhe-backend` feature flag.
//!
//! # Status
//!
//! Stub — key generation scaffolding is in place; the encrypted evaluator is
//! not yet implemented. The plaintext [`crate::eval::PlaintextEvaluator`] is
//! the correct choice for inference today.

use tfhe::{ConfigBuilder, FheUint8, ServerKey, generate_keys, set_server_key};

use crate::Error;

/// Holds the FHE key material and scheme configuration.
///
/// The `ClientKey` is kept here for decryption after inference; in a real
/// deployment the client would hold it locally and only share the `ServerKey`.
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

    /// Install the server key as the thread-local active key for TFHE-rs operations.
    pub fn set_active(&self) {
        set_server_key(self.server_key.clone());
    }
}

/// Placeholder for the encrypted evaluator.
///
/// Will implement [`crate::eval::Evaluator`] with `Input = Vec<FheUint8>`
/// and `Output = FheUint8` once the circuit translation is complete.
pub struct FheEvaluator {
    #[allow(dead_code)]
    ctx: FheContext,
}

impl FheEvaluator {
    pub fn new(ctx: FheContext) -> Self {
        FheEvaluator { ctx }
    }
}
