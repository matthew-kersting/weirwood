//! Evaluators for running inference over a loaded [`Ensemble`].
//!
//! [`PlaintextEvaluator`] runs standard floating-point inference and is useful
//! for verifying model loading and as a correctness reference for the FHE path.
//!
//! The FHE evaluator lives in the [`crate::fhe`] module behind the
//! `tfhe-backend` feature flag.

use crate::model::{Ensemble, Objective};

/// Trait implemented by any inference backend (plaintext or encrypted).
pub trait Evaluator {
    /// The type of a single feature vector the evaluator accepts.
    type Input;
    /// The type of the raw pre-activation score returned.
    type Output;

    /// Run inference and return the raw ensemble score.
    ///
    /// For classification the caller is responsible for applying the activation
    /// (sigmoid, softmax) to the returned score. [`PlaintextEvaluator`] provides
    /// [`PlaintextEvaluator::predict_proba`] as a convenience.
    fn predict(&self, ensemble: &Ensemble, input: &Self::Input) -> Self::Output;
}

/// Plaintext f32 evaluator — no encryption, useful for testing and benchmarking.
pub struct PlaintextEvaluator;

impl Evaluator for PlaintextEvaluator {
    type Input = Vec<f32>;
    type Output = f32;

    fn predict(&self, ensemble: &Ensemble, features: &Vec<f32>) -> f32 {
        let raw: f32 = ensemble.trees.iter().map(|t| t.evaluate(features)).sum();
        raw + ensemble.base_score
    }
}

impl PlaintextEvaluator {
    /// Predict and apply the appropriate activation for the ensemble's objective.
    ///
    /// - `BinaryLogistic` → sigmoid
    /// - `RegSquaredError` → identity
    /// - `MultiSoftmax` → softmax over per-class scores (returns only class 0 for now)
    pub fn predict_proba(&self, ensemble: &Ensemble, features: &Vec<f32>) -> f32 {
        let score = self.predict(ensemble, features);
        match &ensemble.objective {
            Objective::BinaryLogistic => sigmoid(score),
            Objective::RegSquaredError => score,
            Objective::MultiSoftmax { .. } => sigmoid(score), // placeholder
            Objective::Other(_) => score,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Node, Objective, Tree};

    fn tiny_ensemble() -> Ensemble {
        // Single stump: if features[0] <= 1.0, leaf = -0.5, else leaf = 0.5
        let nodes = vec![
            Node {
                split_feature: 0,
                split_threshold: 1.0,
                left_child: 1,
                right_child: 2,
                leaf_value: 0.0,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: -0.5,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: 0.5,
            },
        ];
        Ensemble {
            trees: vec![Tree { nodes }],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 1,
        }
    }

    #[test]
    fn plaintext_left_branch() {
        let e = tiny_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![0.5]);
        approx::assert_abs_diff_eq!(score, -0.5, epsilon = 1e-6);
    }

    #[test]
    fn plaintext_right_branch() {
        let e = tiny_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![2.0]);
        approx::assert_abs_diff_eq!(score, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sigmoid_sanity() {
        let e = tiny_ensemble();
        let p = PlaintextEvaluator.predict_proba(&e, &vec![2.0]);
        assert!(p > 0.5 && p < 1.0);
    }
}
