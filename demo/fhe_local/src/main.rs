//! In-process FHE inference demo.
//!
//! Runs the whole client+server pipeline on one machine: load model, generate
//! keys, encrypt features, evaluate the encrypted XGBoost ensemble, decrypt,
//! apply sigmoid. The two test cases come from the held-out Breast Cancer
//! Wisconsin test split; expected probabilities P(benign=1) are from XGBoost.
//!
//! Run with:
//!   cargo run --release
//!
//! Each FHE inference takes ~4 minutes on CPU, so the full demo is ~8 minutes.

use weirwood::{
    eval::{Evaluator as _, PlaintextEvaluator},
    fhe::{ClientContext, FheEvaluator},
    model::{Objective, WeirwoodTree},
};

const MODEL_PATH: &str = "model.ubj";

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

/// Apply the model's activation to a raw FHE-decrypted ensemble score.
/// The local demo can do this directly because it has the model in hand.
fn activate(objective: &Objective, raw: f32) -> f32 {
    match objective {
        Objective::BinaryLogistic => 1.0 / (1.0 + (-raw).exp()),
        _ => raw,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from {MODEL_PATH}…");
    let model = WeirwoodTree::from_file(MODEL_PATH)?;
    println!(
        "  {} trees, {} features, objective {:?}",
        model.trees.len(),
        model.num_features,
        model.objective
    );

    println!("\nGenerating FHE keypair (~1–3 s)…");
    let client = ClientContext::generate()?;
    let evaluator = FheEvaluator::try_new(&model, client.server_context())?;

    for sample in SAMPLES {
        println!("\n=== {} ===", sample.label);

        let plaintext_proba = PlaintextEvaluator.predict_proba(&model, &sample.features)?;

        println!("  encrypting + evaluating in FHE (~4 min on CPU)…");
        let encrypted_features = client.encrypt(&sample.features);
        let encrypted_score = evaluator.predict(&model, &encrypted_features);
        let fhe_proba = activate(&model.objective, client.decrypt_score(&encrypted_score));

        println!("  expected P(benign) : {:.4}", sample.expected_proba);
        println!("  plaintext          : {plaintext_proba:.4}");
        println!("  FHE                : {fhe_proba:.4}");
    }

    Ok(())
}
