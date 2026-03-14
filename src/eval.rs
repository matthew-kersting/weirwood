//! Evaluators for running inference over a loaded [`WeirwoodTree`].
//!
//! [`PlaintextEvaluator`] runs standard floating-point inference and is useful
//! for verifying model loading and as a correctness reference for the FHE path.
//!
//! The FHE evaluator lives in the [`crate::fhe`] module.

use crate::model::{Objective, WeirwoodTree};

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
    fn predict(&self, weirwood_tree: &WeirwoodTree, input: &Self::Input) -> Self::Output;
}

/// Plaintext f32 evaluator — no encryption, useful for testing and benchmarking.
pub struct PlaintextEvaluator;

impl Evaluator for PlaintextEvaluator {
    type Input = Vec<f32>;
    type Output = f32;

    fn predict(&self, weirwood_tree: &WeirwoodTree, features: &Vec<f32>) -> f32 {
        let raw_score: f32 = weirwood_tree
            .trees
            .iter()
            .map(|decision_tree| decision_tree.evaluate(features))
            .sum();
        raw_score + weirwood_tree.base_score
    }
}

impl PlaintextEvaluator {
    /// Predict and apply the appropriate activation for the model's objective.
    ///
    /// - `BinaryLogistic` → sigmoid
    /// - `RegSquaredError` → identity
    /// - `MultiSoftmax` → softmax over per-class scores (returns only class 0 for now)
    pub fn predict_proba(&self, weirwood_tree: &WeirwoodTree, features: &Vec<f32>) -> f32 {
        let raw_score: f32 = self.predict(weirwood_tree, features);
        match &weirwood_tree.objective {
            Objective::BinaryLogistic => sigmoid(raw_score),
            Objective::RegSquaredError => raw_score,
            Objective::MultiSoftmax { .. } => sigmoid(raw_score), // placeholder
            Objective::Other(_) => raw_score,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Node, Objective, Tree, WeirwoodTree};

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    /// Single stump: feature[0] <= 1.0 → left (-0.5), else right (0.5).
    fn tiny_tree() -> WeirwoodTree {
        let nodes: Vec<Node> = vec![
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
        WeirwoodTree {
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
    fn deep_tree() -> WeirwoodTree {
        let nodes: Vec<Node> = vec![
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
        WeirwoodTree {
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
        let tree: WeirwoodTree = tiny_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![0.5]);
        approx::assert_abs_diff_eq!(score, -0.5, epsilon = 1e-6);
    }

    #[test]
    fn plaintext_right_branch() {
        let tree: WeirwoodTree = tiny_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![2.0]);
        approx::assert_abs_diff_eq!(score, 0.5, epsilon = 1e-6);
    }

    /// A feature value exactly equal to the threshold must go LEFT
    /// (the split condition is `feature <= threshold`).
    #[test]
    fn boundary_at_threshold_goes_left() {
        let tree: WeirwoodTree = tiny_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![1.0]);
        approx::assert_abs_diff_eq!(score, -0.5, epsilon = 1e-6);
    }

    #[test]
    fn just_above_threshold_goes_right() {
        let tree: WeirwoodTree = tiny_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![1.0001]);
        approx::assert_abs_diff_eq!(score, 0.5, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------------------
    // Multi-level routing
    // ---------------------------------------------------------------------------

    #[test]
    fn depth2_left_left() {
        // feature[0]=1.0 (<=5→left), feature[1]=1.0 (<=2→left) → leaf -1.0
        let tree: WeirwoodTree = deep_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![1.0, 1.0]);
        approx::assert_abs_diff_eq!(score, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn depth2_left_right() {
        // feature[0]=1.0 (<=5→left), feature[1]=3.0 (>2→right) → leaf 0.5
        let tree: WeirwoodTree = deep_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![1.0, 3.0]);
        approx::assert_abs_diff_eq!(score, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn depth2_right() {
        // feature[0]=6.0 (>5→right) → leaf 1.0 (never looks at feature[1])
        let tree: WeirwoodTree = deep_tree();
        let score: f32 = PlaintextEvaluator.predict(&tree, &vec![6.0, 99.0]);
        approx::assert_abs_diff_eq!(score, 1.0, epsilon = 1e-6);
    }

    /// Tree uses feature[1], not feature[0] — verifies split_feature indexing.
    #[test]
    fn correct_feature_index_used() {
        let nodes: Vec<Node> = vec![
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
        let tree: WeirwoodTree = WeirwoodTree {
            trees: vec![Tree { nodes }],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 2,
        };
        // feature[0] is irrelevant; split is on feature[1]
        let left_score: f32 = PlaintextEvaluator.predict(&tree, &vec![999.0, 0.0]);
        let right_score: f32 = PlaintextEvaluator.predict(&tree, &vec![0.0, 1.0]);
        approx::assert_abs_diff_eq!(left_score, -1.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(right_score, 1.0, epsilon = 1e-6);
    }

    // ---------------------------------------------------------------------------
    // Multi-tree summation and base_score
    // ---------------------------------------------------------------------------

    #[test]
    fn two_trees_sum_correctly() {
        // Tree 1: feature[0] <= 1.0 → -0.3 else 0.3
        // Tree 2: feature[0] <= 1.0 → -0.2 else 0.2
        let make_stump = |left_leaf: f32, right_leaf: f32| Tree {
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
                    leaf_value: left_leaf,
                },
                Node {
                    split_feature: 0,
                    split_threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    leaf_value: right_leaf,
                },
            ],
        };
        let tree: WeirwoodTree = WeirwoodTree {
            trees: vec![make_stump(-0.3, 0.3), make_stump(-0.2, 0.2)],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 1,
        };
        // Both left: -0.3 + -0.2 = -0.5
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&tree, &vec![0.0]),
            -0.5,
            epsilon = 1e-6
        );
        // Both right: 0.3 + 0.2 = 0.5
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&tree, &vec![2.0]),
            0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn base_score_is_added_to_raw() {
        let mut tree: WeirwoodTree = tiny_tree();
        tree.base_score = 2.0;
        // left branch gives -0.5 + 2.0 = 1.5
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&tree, &vec![0.5]),
            1.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn zero_trees_returns_base_score() {
        let tree: WeirwoodTree = WeirwoodTree {
            trees: vec![],
            objective: Objective::BinaryLogistic,
            base_score: 0.5,
            num_features: 1,
        };
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict(&tree, &vec![1.0]),
            0.5,
            epsilon = 1e-6
        );
    }

    // ---------------------------------------------------------------------------
    // Activation functions and predict_proba
    // ---------------------------------------------------------------------------

    #[test]
    fn sigmoid_sanity() {
        let tree: WeirwoodTree = tiny_tree();
        let probability: f32 = PlaintextEvaluator.predict_proba(&tree, &vec![2.0]);
        assert!(probability > 0.5 && probability < 1.0);
    }

    #[test]
    fn sigmoid_of_zero_is_half() {
        // base_score=0, both leaf values are 0 → raw=0 → sigmoid(0)=0.5
        let nodes: Vec<Node> = vec![
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
        let tree: WeirwoodTree = WeirwoodTree {
            trees: vec![Tree { nodes }],
            objective: Objective::BinaryLogistic,
            base_score: 0.0,
            num_features: 1,
        };
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&tree, &vec![0.5]),
            0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn sigmoid_known_values() {
        // sigmoid(-0.5) ≈ 0.37754066
        // sigmoid( 0.5) ≈ 0.62245934
        let tree: WeirwoodTree = tiny_tree();
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&tree, &vec![0.0]), // left → raw=-0.5
            0.37754066_f32,
            epsilon = 1e-5
        );
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&tree, &vec![2.0]), // right → raw=0.5
            0.62245934_f32,
            epsilon = 1e-5
        );
    }

    #[test]
    fn regression_predict_proba_is_raw_score() {
        let mut tree: WeirwoodTree = tiny_tree();
        tree.objective = Objective::RegSquaredError;
        tree.base_score = 1.0;
        // right branch: 0.5 + base 1.0 = 1.5 — no activation applied
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&tree, &vec![2.0]),
            1.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn other_objective_predict_proba_is_raw_score() {
        let mut tree: WeirwoodTree = tiny_tree();
        tree.objective = Objective::Other("custom:loss".into());
        approx::assert_abs_diff_eq!(
            PlaintextEvaluator.predict_proba(&tree, &vec![0.0]),
            -0.5,
            epsilon = 1e-6
        );
    }
}
