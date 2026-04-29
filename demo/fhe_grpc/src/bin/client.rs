//! Minimal FHE inference gRPC client.
//!
//! Connects to the local server, generates an FHE keypair, uploads the public
//! `ServerKey`, and runs encrypted inference on two real samples from the
//! Breast Cancer Wisconsin held-out test split. Expected probabilities
//! P(benign=1) are from XGBoost.
//!
//! The client never loads the XGBoost model — the server reports its shape
//! (number of features, objective) at session-init time.
//!
//! Run with (server must be running first):
//!   cargo run --release --bin client
//!
//! Each FHE inference takes ~4 minutes on CPU.

use weirwood::transport::WeirwoodClient;

const SERVER_URL: &str = "http://127.0.0.1:9999";

struct Sample {
    label: &'static str,
    expected_proba: f32,
    features: [f32; 30],
}

const SAMPLES: &[Sample] = &[
    Sample {
        label: "malignant tumor",
        expected_proba: 0.00134412,
        features: [
            1.568513, 2.164016, 1.742866, 1.620764, -0.265376, 1.915932, 1.092455, 1.698896,
            0.309589, -0.069884, 1.658233, -0.034559, 2.199280, 1.609795, -0.232337, 1.198715,
            0.187957, 0.559817, -0.162073, 0.523832, 1.862160, 1.708208, 2.170623, 1.913701,
            -0.187741, 1.744274, 0.688560, 1.202144, -0.140431, 0.900362,
        ],
    },
    Sample {
        label: "benign tumor",
        expected_proba: 0.99849713,
        features: [
            -0.840276, -0.597067, -0.874173, -0.776169, -1.090346, -1.225691, -0.924639, -0.891008,
            -1.052578, -0.187341, -0.962401, -0.440763, -0.940773, -0.708094, -0.383363, -1.044917,
            -0.716089, -0.872824, -0.078685, -0.511433, -0.943849, -0.861492, -0.973363, -0.810993,
            -1.302037, -1.203076, -1.069908, -1.101241, -0.814291, -0.713236,
        ],
    },
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Connecting to {SERVER_URL} (includes ~1–3 s of FHE keygen)…");
    let mut client = WeirwoodClient::connect(SERVER_URL).await?;
    println!(
        "  server reports objective {:?}, {} features",
        client.objective(),
        client.num_features()
    );

    for sample in SAMPLES {
        println!("\n=== {} ===", sample.label);
        println!("  expected P(benign) : {:.4}", sample.expected_proba);
        println!("  running encrypted inference (~4 min on CPU)…");
        let proba = client.predict_proba(&sample.features).await?;
        println!("  FHE prediction     : {proba:.4}");
    }

    println!("\nServer never saw plaintext features or scores.");
    Ok(())
}
