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
    use crate::model::{Ensemble, Node, Objective, Tree};

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Single stump: feature[0] <= 1.0 → left (-0.5), else right (0.5).
    fn tiny_ensemble() -> Ensemble {
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

    /// Two-level tree over 2 features:
    ///   Node 0: feature[0] <= 5.0 → left (node 1), right (node 2, leaf=1.0)
    ///   Node 1: feature[1] <= 2.0 → left (node 3, leaf=-1.0), right (node 4, leaf=0.5)
    ///   Node 2: leaf = 1.0
    ///   Node 3: leaf = -1.0
    ///   Node 4: leaf = 0.5
    fn deep_ensemble() -> Ensemble {
        let nodes = vec![
            Node {
                split_feature: 0,
                split_threshold: 5.0,
                left_child: 1,
                right_child: 2,
                leaf_value: 0.0,
            },
            Node {
                split_feature: 1,
                split_threshold: 2.0,
                left_child: 3,
                right_child: 4,
                leaf_value: 0.0,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: 1.0,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: -1.0,
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
            num_features: 2,
        }
    }

    // ---------------------------------------------------------------------------
    // Routing tests
    // ---------------------------------------------------------------------------

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

    /// A feature value exactly equal to the threshold must go LEFT
    /// (the split condition is `feature <= threshold`).
    #[test]
    fn boundary_at_threshold_goes_left() {
        let e = tiny_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![1.0]);
        approx::assert_abs_diff_eq!(score, -0.5, epsilon = 1e-6);
    }

    #[test]
    fn just_above_threshold_goes_right() {
        let e = tiny_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![1.0001]);
        approx::assert_abs_diff_eq!(score, 0.5, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------------------
    // Multi-level routing
    // ---------------------------------------------------------------------------

    #[test]
    fn depth2_left_left() {
        // feature[0]=1.0 (<=5→left), feature[1]=1.0 (<=2→left) → leaf -1.0
        let e = deep_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![1.0, 1.0]);
        approx::assert_abs_diff_eq!(score, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn depth2_left_right() {
        // feature[0]=1.0 (<=5→left), feature[1]=3.0 (>2→right) → leaf 0.5
        let e = deep_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![1.0, 3.0]);
        approx::assert_abs_diff_eq!(score, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn depth2_right() {
        // feature[0]=6.0 (>5→right) → leaf 1.0 (never looks at feature[1])
        let e = deep_ensemble();
        let score = PlaintextEvaluator.predict(&e, &vec![6.0, 99.0]);
        approx::assert_abs_diff_eq!(score, 1.0, epsilon = 1e-6);
    }

    /// Tree uses feature[1], not feature[0] — verifies split_feature indexing.
    #[test]
    fn correct_feature_index_used() {
        let nodes = vec![
            Node {
                split_feature: 1,
                split_threshold: 0.5,
                left_child: 1,
                right_child: 2,
                leaf_value: 0.0,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: -1.0,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: 1.0,
            },
        ];
        let e = Ensemble {
            trees: vec![Tree { nodes }],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 2,
        };
        // feature[0] is irrelevant; split is on feature[1]
        let left = PlaintextEvaluator.predict(&e, &vec![999.0, 0.0]);
        let right = PlaintextEvaluator.predict(&e, &vec![0.0, 1.0]);
        approx::assert_abs_diff_eq!(left, -1.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(right, 1.0, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------------------
    // Multi-tree summation and base_score
    // ---------------------------------------------------------------------------

    #[test]
    fn two_trees_sum_correctly() {
        // Tree 1: feature[0] <= 1.0 → -0.3 else 0.3
        // Tree 2: feature[0] <= 1.0 → -0.2 else 0.2
        let make_stump = |left: f32, right: f32| Tree {
            nodes: vec![
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
                    leaf_value: left,
                },
                Node {
                    split_feature: 0,
                    split_threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    leaf_value: right,
                },
            ],
        };
        let e = Ensemble {
            trees: vec![make_stump(-0.3, 0.3), make_stump(-0.2, 0.2)],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 1,
        };
        // Both left: -0.3 + -0.2 = -0.5
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&e, &vec![0.0]),
            -0.5,
            epsilon = 1e-6
        );
        // Both right: 0.3 + 0.2 = 0.5
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&e, &vec![2.0]),
            0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn base_score_is_added_to_raw() {
        let mut e = tiny_ensemble();
        e.base_score = 2.0;
        // left branch gives -0.5 + 2.0 = 1.5
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&e, &vec![0.5]),
            1.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn zero_trees_returns_base_score() {
        let e = Ensemble {
            trees: vec![],
            objective: Objective::BinaryLogistic,
            base_score: 0.5,
            num_features: 1,
        };
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&e, &vec![1.0]),
            0.5,
            epsilon = 1e-6
        );
    }

    // ---------------------------------------------------------------------------
    // Activation functions and predict_proba
    // ---------------------------------------------------------------------------

    #[test]
    fn sigmoid_sanity() {
        let e = tiny_ensemble();
        let p = PlaintextEvaluator.predict_proba(&e, &vec![2.0]);
        assert!(p > 0.5 && p < 1.0);
    }

    #[test]
    fn sigmoid_of_zero_is_half() {
        // base_score=0, both leaf values are 0 → raw=0 → sigmoid(0)=0.5
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
                leaf_value: 0.0,
            },
            Node {
                split_feature: 0,
                split_threshold: 0.0,
                left_child: -1,
                right_child: -1,
                leaf_value: 0.0,
            },
        ];
        let e = Ensemble {
            trees: vec![Tree { nodes }],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 1,
        };
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&e, &vec![0.5]),
            0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn sigmoid_known_values() {
        // sigmoid(-0.5) ≈ 0.37754066
        // sigmoid( 0.5) ≈ 0.62245934
        let e = tiny_ensemble();
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&e, &vec![0.0]), // left → raw=-0.5
            0.37754066_f32,
            epsilon = 1e-5
        );
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&e, &vec![2.0]), // right → raw=0.5
            0.62245934_f32,
            epsilon = 1e-5
        );
    }

    #[test]
    fn regression_predict_proba_is_raw_score() {
        let mut e = tiny_ensemble();
        e.objective = Objective::RegSquaredError;
        e.base_score = 1.0;
        // right branch: 0.5 + base 1.0 = 1.5 — no activation applied
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&e, &vec![2.0]),
            1.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn other_objective_predict_proba_is_raw_score() {
        let mut e = tiny_ensemble();
        e.objective = Objective::Other("custom:loss".into());
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&e, &vec![0.0]),
            -0.5,
            epsilon = 1e-6
        );
    }
}
