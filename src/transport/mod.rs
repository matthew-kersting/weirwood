//! Serialization helpers and gRPC types for transporting FHE types over the wire.
//!
//! Provides functions to serialize and deserialize `ServerContext` and encrypted
//! data types using TFHE's native serialization format. Intended for use with
//! gRPC or other network protocols.

pub mod rpc;

use std::io::{Cursor, Read, Write};

use tfhe::safe_serialization::{safe_deserialize, safe_serialize};

use crate::eval::fhe::{EncryptedInput, EncryptedScore, ServerContext};
use crate::Error;

const SIZE_LIMIT_BYTES: u64 = 512 * 1024 * 1024;

/// Serialize a `ServerKey` from a `ServerContext` to bytes using TFHE's safe serialization.
///
/// The resulting bytes (~100-200 MB) are suitable for transmission over gRPC
/// or other network protocols. The serialization includes versioning information.
pub fn serialize_server_context(ctx: &ServerContext) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    safe_serialize(&ctx.server_key, &mut buf, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Other(format!("failed to serialize server key: {}", e)))?;
    Ok(buf)
}

/// Deserialize a `ServerContext` from bytes using TFHE's safe deserialization.
///
/// Expects bytes produced by `serialize_server_context`.
pub fn deserialize_server_context(bytes: &[u8]) -> Result<ServerContext, Error> {
    let mut reader = Cursor::new(bytes);
    let server_key: tfhe::ServerKey = safe_deserialize(&mut reader, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Other(format!("failed to deserialize server key: {}", e)))?;
    Ok(ServerContext::from_key(server_key))
}

/// Serialize a single encrypted feature (`FheInt32`) to bytes.
pub fn serialize_feature(feature: &tfhe::FheInt32) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    safe_serialize(feature, &mut buf, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Other(format!("failed to serialize feature: {}", e)))?;
    Ok(buf)
}

/// Deserialize a single encrypted feature from bytes.
pub fn deserialize_feature(bytes: &[u8]) -> Result<tfhe::FheInt32, Error> {
    let mut reader = Cursor::new(bytes);
    safe_deserialize(&mut reader, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Other(format!("failed to deserialize feature: {}", e)))
}

/// Serialize an encrypted score to bytes.
pub fn serialize_score(score: &EncryptedScore) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    safe_serialize(score, &mut buf, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Other(format!("failed to serialize encrypted score: {}", e)))?;
    Ok(buf)
}

/// Deserialize an encrypted score from bytes.
pub fn deserialize_score(bytes: &[u8]) -> Result<EncryptedScore, Error> {
    let mut reader = Cursor::new(bytes);
    safe_deserialize(&mut reader, SIZE_LIMIT_BYTES)
        .map_err(|e| Error::Other(format!("failed to deserialize encrypted score: {}", e)))
}

/// Serialize an encrypted input (feature vector) to bytes.
///
/// Each feature is serialized individually and concatenated, with a 4-byte
/// length prefix indicating the number of features. This avoids relying on
/// `Named` for `Vec<FheInt32>`, which is not implemented by tfhe-rs.
pub fn serialize_encrypted_input(input: &EncryptedInput) -> Result<Vec<u8>, Error> {
    let mut buf = Vec::new();
    let num_features = input.len() as u32;
    buf.extend_from_slice(&num_features.to_le_bytes());
    for feature in input {
        let feature_bytes = serialize_feature(feature)?;
        buf.extend_from_slice(&(feature_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&feature_bytes);
    }
    Ok(buf)
}

/// Deserialize an encrypted input (feature vector) from bytes.
///
/// Expects the format produced by `serialize_encrypted_input`.
pub fn deserialize_encrypted_input(bytes: &[u8]) -> Result<EncryptedInput, Error> {
    let mut reader = Cursor::new(bytes);

    let mut num_features_bytes = [0u8; 4];
    reader
        .read_exact(&mut num_features_bytes)
        .map_err(|e| Error::Other(format!("failed to read feature count: {}", e)))?;
    let num_features = u32::from_le_bytes(num_features_bytes) as usize;

    let mut features = Vec::with_capacity(num_features);
    for _ in 0..num_features {
        let mut len_bytes = [0u8; 4];
        reader
            .read_exact(&mut len_bytes)
            .map_err(|e| Error::Other(format!("failed to read feature length: {}", e)))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut feature_bytes = vec![0u8; len];
        reader
            .read_exact(&mut feature_bytes)
            .map_err(|e| Error::Other(format!("failed to read feature bytes: {}", e)))?;

        let feature = deserialize_feature(&feature_bytes)?;
        features.push(feature);
    }

    Ok(features)
}
