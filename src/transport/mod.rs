//! gRPC transport for privacy-preserving inference, plus the
//! `safe_serialize` / `safe_deserialize` helpers that move FHE artifacts
//! across the wire.
//!
//! # gRPC contract
//!
//! The Protocol Buffer service definition lives in `proto/inference.proto`
//! and is compiled to Rust by `tonic-build` at crate-build time. The
//! generated server trait and client stub are re-exported here for
//! convenience:
//!
//! ```text
//! use weirwood::transport::{InferenceServiceServer, InferenceService,
//!                           InferenceServiceClient};
//! ```
//!
//! Implement [`InferenceService`] and serve it under
//! `tonic::transport::Server`; see `examples/server.rs` for the minimal
//! wiring. A high-level convenience client that bundles
//! key-generation → InitSession → encrypt → Predict → decrypt → activation
//! is provided by [`WeirwoodClient`] (gated on the `transport` feature).
//!
//! # Payload encoding
//!
//! All ciphertext bytes use `tfhe-rs`' versioned `safe_serialize` /
//! `safe_deserialize` format. Helpers are exposed for callers that want to
//! drive the protocol manually (e.g. proxying through a different transport).

pub mod rpc;

mod client;
pub use client::WeirwoodClient;

/// gRPC server-side trait — implement this and wrap with
/// [`InferenceServiceServer::new`] to plug into `tonic::transport::Server`.
pub use rpc::inference_service_server::{InferenceService, InferenceServiceServer};

/// gRPC client stub — typically used by [`WeirwoodClient`], but exposed for
/// callers who want to drive the protocol themselves.
pub use rpc::inference_service_client::InferenceServiceClient;

pub use rpc::{InitSessionRequest, InitSessionResponse, PredictRequest, PredictResponse};

use std::io::Cursor;

use tfhe::named::Named;
use tfhe::safe_serialization::{safe_deserialize, safe_serialize};
use tfhe::{Unversionize, Versionize};

use crate::Error;
use crate::eval::fhe::{EncryptedScore, ServerContext};

/// Maximum payload size accepted by [`safe_serialize`] / [`safe_deserialize`].
/// 512 MB is generous headroom over the ~180 MB serialized `ServerKey`.
const SIZE_LIMIT_BYTES: u64 = 512 * 1024 * 1024;

/// Maximum gRPC message size for both encoding and decoding ends.
/// Mirrored on the server (`InferenceServiceServer`) and the client
/// (`WeirwoodClient` / `InferenceServiceClient`) so the large `InitSession`
/// payload (~180 MB) isn't rejected by tonic's 4 MB default.
pub const MAX_GRPC_MESSAGE_BYTES: usize = 512 * 1024 * 1024;

fn safe_to_bytes<T>(value: &T, what: &str) -> Result<Vec<u8>, Error>
where
    T: serde::Serialize + Versionize + Named,
{
    let mut buf = Vec::new();
    safe_serialize(value, &mut buf, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Fhe(format!("failed to serialize {what}: {e}")))?;
    Ok(buf)
}

fn safe_from_bytes<T>(bytes: &[u8], what: &str) -> Result<T, Error>
where
    T: serde::de::DeserializeOwned + Unversionize + Named,
{
    safe_deserialize(&mut Cursor::new(bytes), SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Fhe(format!("failed to deserialize {what}: {e}")))
}

/// Serialize a [`ServerContext`]'s server key (~100–200 MB) for transport.
pub fn serialize_server_context(ctx: &ServerContext) -> Result<Vec<u8>, Error> {
    safe_to_bytes(&ctx.server_key, "server key")
}

/// Deserialize a [`ServerContext`] from bytes produced by [`serialize_server_context`].
pub fn deserialize_server_context(bytes: &[u8]) -> Result<ServerContext, Error> {
    let server_key: tfhe::ServerKey = safe_from_bytes(bytes, "server key")?;
    Ok(ServerContext::from_key(server_key))
}

/// Serialize a single encrypted feature (`FheInt32`).
pub fn serialize_feature(feature: &tfhe::FheInt32) -> Result<Vec<u8>, Error> {
    safe_to_bytes(feature, "feature")
}

/// Deserialize a single encrypted feature.
pub fn deserialize_feature(bytes: &[u8]) -> Result<tfhe::FheInt32, Error> {
    safe_from_bytes(bytes, "feature")
}

/// Serialize an encrypted score.
pub fn serialize_score(score: &EncryptedScore) -> Result<Vec<u8>, Error> {
    safe_to_bytes(score, "encrypted score")
}

/// Deserialize an encrypted score.
pub fn deserialize_score(bytes: &[u8]) -> Result<EncryptedScore, Error> {
    safe_from_bytes(bytes, "encrypted score")
}
