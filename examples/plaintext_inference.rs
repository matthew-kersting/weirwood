//! Minimal end-to-end example of plaintext inference with weirwood.
//!
//! Run against the committed test fixture:
//!
//! ```sh
//! cargo run --example plaintext_inference -- tests/fixtures/trained_binary.ubj 0.7 0.3
//! cargo run --example plaintext_inference -- tests/fixtures/trained_binary.json 0.0 0.0
//! ```
//!
//! If no features are supplied the example runs the built-in test vectors.

use std::env;

use weirwood::{eval::PlaintextEvaluator, model::WeirwoodTree};

fn main() -> Result<(), weirwood::Error> {
    let cli_args: Vec<String> = env::args().collect();

    if cli_args.len() < 2 {
        eprintln!("Usage: plaintext_inference <model.json|model.ubj> [feature1 feature2 ...]");
        std::process::exit(1);
    }

    let model_path: &String = &cli_args[1];
    let weirwood_tree: WeirwoodTree = if model_path.ends_with(".ubj") {
        WeirwoodTree::from_ubj_file(model_path)?
    } else {
        WeirwoodTree::from_json_file(model_path)?
    };

    println!(
        "Loaded: {} trees, {} features, objective {:?}",
        weirwood_tree.trees.len(),
        weirwood_tree.num_features,
        weirwood_tree.objective,
    );

    if cli_args.len() > 2 {
        let features: Vec<f32> = cli_args[2..]
            .iter()
            .map(|feature_str| feature_str.parse::<f32>().expect("feature must be a float"))
            .collect();

        let predicted_score: f32 = PlaintextEvaluator.predict_proba(&weirwood_tree, &features);
        println!("predict_proba({features:?}) = {predicted_score:.6}");
    } else {
        println!("No features supplied — running built-in test vectors:\n");
        let test_vectors: &[&[f32]] = &[&[0.0, 0.0], &[0.5, 0.5], &[1.0, 1.0], &[0.7, 0.3]];
        for feature_vector in test_vectors {
            let predicted_score: f32 =
                PlaintextEvaluator.predict_proba(&weirwood_tree, &feature_vector.to_vec());
            println!("  {feature_vector:?}  ->  {predicted_score:.6}");
        }
    }

    Ok(())
}
