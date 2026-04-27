//! Simple TCP-based client for privacy-preserving XGBoost inference.
//!
//! Connects to a remote server, uploads FHE keys, encrypts features locally,
//! and receives encrypted predictions without the server ever seeing plaintext data.
//!
//! Usage:
//!   cargo run --release --example client --features transport [--server 127.0.0.1:9999]

use std::io::{Read, Write};
use std::net::TcpStream;

use prost::Message as _;

use weirwood::{
    eval::Evaluator as _,
    fhe::ClientContext,
    model::WeirwoodTree,
    transport::rpc::{InitSessionRequest, InitSessionResponse, PredictRequest, PredictResponse},
    transport::{
        deserialize_score, serialize_encrypted_input, serialize_feature, serialize_server_context,
    },
};

const DEFAULT_MODEL: &str = "tests/fixtures/trained_binary.ubj";

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut server_addr = "127.0.0.1:9999".to_string();

    while let Some(arg) = args.next() {
        if arg == "--server" {
            server_addr = args.next().expect("--server requires an address");
        }
    }

    // --- Load model ---
    println!("Loading model from {}…", DEFAULT_MODEL);
    let model = WeirwoodTree::from_ubj_file(DEFAULT_MODEL)?;
    println!(
        "Model loaded: {} trees, {} features, objective: {:?}",
        model.trees.len(),
        model.num_features,
        model.objective
    );

    // --- Generate FHE keys ---
    println!();
    println!("Generating FHE keys (this takes ~1–3 seconds)…");
    let client = ClientContext::generate()?;
    println!("Keys generated.");

    // --- Connect to server ---
    println!();
    println!("Connecting to server at {}…", server_addr);
    let mut stream = TcpStream::connect(&server_addr)?;
    println!("Connected.");

    // --- InitSession RPC ---
    println!();
    println!("Uploading server key to remote server…");
    let server_key_bytes = serialize_server_context(&client.server_context())?;
    println!("Server key is {} MB", server_key_bytes.len() / 1_000_000);

    let init_req = InitSessionRequest {
        server_key: server_key_bytes.into(),
    };

    let mut init_req_bytes = Vec::new();
    init_req.encode(&mut init_req_bytes)?;

    // Send: msg_type (0 = InitSession), length, data
    stream.write_all(&[0u8])?;
    stream.write_all(&(init_req_bytes.len() as u32).to_le_bytes())?;
    stream.write_all(&init_req_bytes)?;

    // Receive response
    let mut len_bytes = [0u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;

    let init_resp = InitSessionResponse::decode(buf.as_slice())?;
    let session_id = init_resp.session_id;
    println!("Session {} created on server.", session_id);

    // --- Run test inferences ---
    // Create feature vectors with proper number of features for the model
    println!();
    let mut make_features = |first_val: f32| -> Vec<f32> {
        let mut v = vec![first_val];
        v.extend(std::iter::repeat(0.0_f32).take(model.num_features - 1));
        v
    };

    let test_cases = vec![
        ("Feature[0] = 0.0 (class 0 region)", make_features(0.0)),
        ("Feature[0] = 1.0 (class 1 region)", make_features(1.0)),
        ("Feature[0] = 0.3 (boundary region)", make_features(0.3)),
    ];

    println!("Running {} test inferences…", test_cases.len());
    println!(
        "{:<30} {:<15} {:<15} {:<15}",
        "Test case", "Plaintext", "FHE", "Δ"
    );
    println!("{}", "-".repeat(75));

    for (name, features) in test_cases {
        // Plaintext reference
        let plaintext_score = weirwood::eval::PlaintextEvaluator.predict(&model, &features);

        // Encrypt locally
        let encrypted_input = client.encrypt(&features);

        // Serialize each feature
        let mut feature_bytes = Vec::new();
        for feature in &encrypted_input {
            feature_bytes.push(serialize_feature(feature)?);
        }

        // Send Predict RPC
        let predict_req = PredictRequest {
            session_id: session_id.clone(),
            features: feature_bytes,
        };

        let mut predict_req_bytes = Vec::new();
        predict_req.encode(&mut predict_req_bytes)?;

        stream.write_all(&[1u8])?; // msg_type = 1 (Predict)
        stream.write_all(&(predict_req_bytes.len() as u32).to_le_bytes())?;
        stream.write_all(&predict_req_bytes)?;

        // Receive response
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut buf = vec![0u8; len];
        stream.read_exact(&mut buf)?;

        let predict_resp = PredictResponse::decode(buf.as_slice())?;

        // Decrypt locally
        let encrypted_score_bytes = predict_resp.encrypted_score;
        let encrypted_score = weirwood::transport::deserialize_score(&encrypted_score_bytes)?;
        let fhe_score = client.decrypt_score(&encrypted_score);

        // Display results
        let delta = (fhe_score - plaintext_score).abs();
        println!(
            "{:<30} {:<15.6} {:<15.6} {:<15.6}",
            name, plaintext_score, fhe_score, delta
        );
    }

    println!();
    println!("All tests completed. Server never saw plaintext features or scores.");
    Ok(())
}
