//! Measure the on-the-wire byte sizes of the artifacts that cross the network
//! in a `transport`-feature deployment: the serialized ServerKey, one encrypted
//! feature (`FheInt32`), and the full encrypted feature vector for the bundled
//! Breast Cancer Wisconsin fixture (30 features). The encrypted score is a
//! single `FheInt32` and is approximately the same size as one encrypted
//! feature.
//!
//! Used to populate the size figures quoted in the paper.
//!
//! Usage:
//!   cargo run --release --example measure_transport_sizes --features transport

use weirwood::{
    fhe::ClientContext,
    model::WeirwoodTree,
    transport::{serialize_encrypted_input, serialize_feature, serialize_server_context},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating ClientContext (this takes ~1-3 s)...");
    let client = ClientContext::generate()?;
    let server_ctx = client.server_context();

    let server_key_bytes = serialize_server_context(&server_ctx)?;
    println!(
        "ServerKey serialized: {} bytes ({:.2} MB)",
        server_key_bytes.len(),
        server_key_bytes.len() as f64 / 1_000_000.0
    );

    let model = WeirwoodTree::from_ubj_file("tests/fixtures/trained_binary.ubj")?;
    let features = vec![0.5_f32; model.num_features];
    let encrypted = client.encrypt(&features);

    let one_feature = serialize_feature(&encrypted[0])?;
    println!(
        "One encrypted feature (FheInt32): {} bytes ({:.2} KB)",
        one_feature.len(),
        one_feature.len() as f64 / 1024.0
    );

    let all_features = serialize_encrypted_input(&encrypted)?;
    println!(
        "Full encrypted input ({} features): {} bytes ({:.2} MB)",
        encrypted.len(),
        all_features.len(),
        all_features.len() as f64 / 1_000_000.0
    );

    println!(
        "EncryptedScore: a single FheInt32, approximately {:.2} KB (same shape as one feature).",
        one_feature.len() as f64 / 1024.0
    );

    Ok(())
}
