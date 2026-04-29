//! End-to-end gRPC client demo using the high-level [`WeirwoodClient`].
//!
//! Shows the one-call flow most users want: connect → predict_proba. The
//! protocol-level types (InferenceServiceClient, InitSession/Predict request
//! and response messages) remain available at `weirwood::transport::*` for
//! callers that need finer control.
//!
//! Usage:
//!   cargo run --release --example client --features transport \
//!     [-- --server http://127.0.0.1:9999]

use weirwood::{eval::PlaintextEvaluator, model::WeirwoodTree, transport::WeirwoodClient};

const DEFAULT_MODEL: &str = "tests/fixtures/trained_binary.ubj";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut server_addr = "http://127.0.0.1:9999".to_string();

    while let Some(arg) = args.next() {
        if arg == "--server" {
            server_addr = args.next().expect("--server requires an address");
        }
    }

    println!("Loading model from {DEFAULT_MODEL}…");
    let model = WeirwoodTree::from_file(DEFAULT_MODEL)?;
    println!(
        "Model loaded: {} trees, {} features, objective: {:?}",
        model.trees.len(),
        model.num_features,
        model.objective
    );

    println!();
    println!("Connecting to {server_addr} (this includes ~1-3 s of FHE keygen)…");
    let mut client = WeirwoodClient::connect(server_addr).await?;
    println!("Session established.");

    let make_features = |first_val: f32| -> Vec<f32> {
        let mut v = vec![first_val];
        v.extend(std::iter::repeat_n(0.0_f32, model.num_features - 1));
        v
    };

    let test_cases = vec![
        ("Feature[0] = 0.0 (class 0 region)", make_features(0.0)),
        ("Feature[0] = 1.0 (class 1 region)", make_features(1.0)),
        ("Feature[0] = 0.3 (boundary region)", make_features(0.3)),
    ];

    println!();
    println!("Running {} test inferences…", test_cases.len());
    println!(
        "{:<35} {:<15} {:<15} {:<10}",
        "Test case", "Plaintext", "FHE", "Δ"
    );
    println!("{}", "-".repeat(78));

    for (name, features) in test_cases {
        let plaintext = PlaintextEvaluator.predict_proba(&model, &features)?;
        let fhe = client.predict_proba(&model, &features).await?;
        let delta = (fhe - plaintext).abs();
        println!("{name:<35} {plaintext:<15.6} {fhe:<15.6} {delta:<10.6}");
    }

    println!();
    println!("All tests completed. Server never saw plaintext features or scores.");
    Ok(())
}
