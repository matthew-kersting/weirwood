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

use weirwood::{eval::PlaintextEvaluator, model::Ensemble};

fn main() -> Result<(), weirwood::Error> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: plaintext_inference <model.json|model.ubj> [feature1 feature2 ...]"
        );
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ensemble = if model_path.ends_with(".ubj") {
        Ensemble::from_ubj_file(model_path)?
    } else {
        Ensemble::from_json_file(model_path)?
    };

    println!(
        "Loaded: {} trees, {} features, objective {:?}",
        ensemble.trees.len(),
        ensemble.num_features,
        ensemble.objective,
    );

    if args.len() > 2 {
        let features: Vec<f32> = args[2..]
            .iter()
            .map(|s| s.parse::<f32>().expect("feature must be a float"))
            .collect();

        let score = PlaintextEvaluator.predict_proba(&ensemble, &features);
        println!("predict_proba({features:?}) = {score:.6}");
    } else {
        println!("No features supplied — running built-in test vectors:\n");
        let vectors: &[&[f32]] = &[
            &[0.0, 0.0],
            &[0.5, 0.5],
            &[1.0, 1.0],
            &[0.7, 0.3],
        ];
        for v in vectors {
            let score = PlaintextEvaluator.predict_proba(&ensemble, &v.to_vec());
            println!("  {v:?}  ->  {score:.6}");
        }
    }

    Ok(())
}
