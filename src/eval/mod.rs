//! Evaluators for running inference over a loaded [`WeirwoodTree`].
//!
//! [`PlaintextEvaluator`] runs standard floating-point inference and is useful
//! for verifying model loading and as a correctness reference for the FHE path.
//! The FHE evaluator lives in the [`fhe`] submodule.

pub mod fhe;

use crate::model::{Objective, WeirwoodTree};

/// Trait implemented by any inference backend (plaintext or encrypted).
///
/// `Input` is `?Sized` so backends can choose the idiomatic borrow shape
/// (e.g. `[f32]` instead of `Vec<f32>`).  Callers pass `&features` and the
/// usual `Vec → slice` deref coercion just works.
pub trait Evaluator {
    /// The type of a single feature vector the evaluator accepts.
    type Input: ?Sized;
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
    type Input = [f32];
    type Output = f32;

    fn predict(&self, weirwood_tree: &WeirwoodTree, features: &[f32]) -> f32 {
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
    /// - `MultiSoftmax` → **panics**; multi-class returns a vector of class
    ///   probabilities, which doesn't fit this method's `f32` return type.
    ///   Use [`Self::predict_multiclass_proba`] instead.
    pub fn predict_proba(&self, weirwood_tree: &WeirwoodTree, features: &[f32]) -> f32 {
        match &weirwood_tree.objective {
            Objective::BinaryLogistic => sigmoid(self.predict(weirwood_tree, features)),
            Objective::RegSquaredError => self.predict(weirwood_tree, features),
            Objective::MultiSoftmax { num_class } => panic!(
                "predict_proba returns a single f32; multi:softmax with num_class={} \
                 produces a vector of class probabilities — use predict_multiclass_proba instead",
                num_class
            ),
            Objective::Other(_) => self.predict(weirwood_tree, features),
        }
    }

    /// Per-class raw (pre-activation) scores for an XGBoost `multi:softmax` /
    /// `multi:softprob` model.
    ///
    /// XGBoost multi-class models are trained as one regression tree per class
    /// per boosting round, with trees interleaved by class:
    /// `tree[i]` contributes to class `i % num_class`. The returned vector has
    /// length `num_class`; entry `k` is the sum of leaf values from every tree
    /// belonging to class `k` (the global `base_score` is added to every class).
    ///
    /// Panics if the model's objective is not `MultiSoftmax`.
    pub fn predict_multiclass(&self, weirwood_tree: &WeirwoodTree, features: &[f32]) -> Vec<f32> {
        let num_class = match &weirwood_tree.objective {
            Objective::MultiSoftmax { num_class } => *num_class,
            other => panic!(
                "predict_multiclass requires a MultiSoftmax objective, got {:?}",
                other
            ),
        };
        assert!(num_class > 0, "MultiSoftmax model must have num_class > 0");

        let mut per_class = vec![weirwood_tree.base_score; num_class];
        for (i, decision_tree) in weirwood_tree.trees.iter().enumerate() {
            per_class[i % num_class] += decision_tree.evaluate(features);
        }
        per_class
    }

    /// Class probabilities for a `multi:softmax` / `multi:softprob` model.
    ///
    /// Computes per-class raw scores via [`Self::predict_multiclass`] and
    /// applies a numerically-stable softmax. Output sums to 1.
    pub fn predict_multiclass_proba(
        &self,
        weirwood_tree: &WeirwoodTree,
        features: &[f32],
    ) -> Vec<f32> {
        softmax(&self.predict_multiclass(weirwood_tree, features))
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable softmax: subtract the max before exponentiating to
/// avoid overflow on large logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
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

    // ---------------------------------------------------------------------------
    // Softmax / multi-class
    // ---------------------------------------------------------------------------

    #[test]
    fn softmax_uniform_logits_is_uniform() {
        let p = softmax(&[1.0, 1.0, 1.0]);
        approx::assert_abs_diff_eq!(p[0], 1.0 / 3.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(p[1], 1.0 / 3.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(p[2], 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[-2.0, 0.5, 3.1]);
        let sum: f32 = p.iter().sum();
        approx::assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn softmax_is_numerically_stable_on_large_logits() {
        // Without the max-subtraction trick, exp(1000) overflows to +inf and
        // the result becomes NaN. A correct implementation handles this.
        let p = softmax(&[1000.0, 1000.0, 999.0]);
        let sum: f32 = p.iter().sum();
        assert!(sum.is_finite(), "softmax must not overflow on large logits");
        approx::assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-5);
        // Classes 0 and 1 have equal logits and should share the bulk of mass.
        approx::assert_abs_diff_eq!(p[0], p[1], epsilon = 1e-6);
        assert!(p[2] < p[0]);
    }

    /// Three-class model with one stump per class, interleaved by class index.
    /// Class k's stump returns +1.0 when `feature[0] == k as f32`, else 0.0,
    /// so `predict_multiclass` should return `1.0` at the "selected" class
    /// and `0.0` elsewhere.
    fn three_class_model() -> WeirwoodTree {
        let stump_for_value = |target: f32| Tree {
            // Stump: feature[0] <= target - 0.5 → 0.0, else (feature[0] <= target + 0.5 → 1.0, else 0.0).
            // We approximate with two splits.
            nodes: vec![
                Node {
                    split_feature: 0,
                    split_threshold: target - 0.5,
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
                    split_threshold: target + 0.5,
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
                    leaf_value: 0.0,
                },
            ],
        };
        WeirwoodTree {
            // Interleaved: tree[0] → class 0, tree[1] → class 1, tree[2] → class 2.
            trees: vec![
                stump_for_value(0.0),
                stump_for_value(1.0),
                stump_for_value(2.0),
            ],
            objective: Objective::MultiSoftmax { num_class: 3 },
            base_score: 0.0,
            num_features: 1,
        }
    }

    #[test]
    fn multiclass_routes_trees_by_class_index() {
        let tree = three_class_model();
        // feature=1.0 → only the class-1 stump fires.
        let raw = PlaintextEvaluator.predict_multiclass(&tree, &vec![1.0]);
        assert_eq!(raw.len(), 3);
        approx::assert_abs_diff_eq!(raw[0], 0.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(raw[1], 1.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(raw[2], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn multiclass_proba_argmax_matches_dominant_class() {
        let tree = three_class_model();
        let p = PlaintextEvaluator.predict_multiclass_proba(&tree, &vec![2.0]);
        let sum: f32 = p.iter().sum();
        approx::assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        let argmax = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, 2);
    }

    #[test]
    fn multiclass_base_score_is_added_to_every_class() {
        let mut tree = three_class_model();
        tree.base_score = 0.25;
        // feature far from every stump's selected value → all stumps return 0.0;
        // each class's raw score should be exactly base_score.
        let raw = PlaintextEvaluator.predict_multiclass(&tree, &vec![10.0]);
        approx::assert_abs_diff_eq!(raw[0], 0.25, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(raw[1], 0.25, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(raw[2], 0.25, epsilon = 1e-6);
        // Equal logits → uniform softmax.
        let p = PlaintextEvaluator.predict_multiclass_proba(&tree, &vec![10.0]);
        approx::assert_abs_diff_eq!(p[0], 1.0 / 3.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(p[1], 1.0 / 3.0, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(p[2], 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "predict_multiclass_proba")]
    fn predict_proba_panics_on_multiclass_objective() {
        let mut tree = tiny_tree();
        tree.objective = Objective::MultiSoftmax { num_class: 3 };
        let _ = PlaintextEvaluator.predict_proba(&tree, &vec![0.5]);
    }

    #[test]
    #[should_panic(expected = "MultiSoftmax")]
    fn predict_multiclass_panics_on_binary_objective() {
        let tree = tiny_tree(); // BinaryLogistic
        let _ = PlaintextEvaluator.predict_multiclass(&tree, &vec![0.5]);
    }
}
