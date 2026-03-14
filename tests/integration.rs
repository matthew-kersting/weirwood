//! Integration tests for weirwood model loading and plaintext inference.
//!
//! The hand-crafted JSON fixtures in `tests/fixtures/` have a known, manually
//! verified structure so expected outputs can be computed exactly.
//!
//! `trained_binary.json` / `.ubj` / `_expected.json` were produced by
//! XGBoost's Python API and are committed as reference fixtures.

use weirwood::{
    eval::{Evaluator, PlaintextEvaluator},
    model::WeirwoodTree,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_fixture_json(fixture_name: &str) -> WeirwoodTree {
    let fixture_path: String = format!("tests/fixtures/{fixture_name}");
    WeirwoodTree::from_json_file(&fixture_path)
        .unwrap_or_else(|error| panic!("failed to load {fixture_path}: {error}"))
}

// ---------------------------------------------------------------------------
// Model loading — structural sanity
// ---------------------------------------------------------------------------

#[test]
fn two_trees_binary_loads_correctly() {
    let loaded_tree: WeirwoodTree = load_fixture_json("two_trees_binary.json");
    assert_eq!(loaded_tree.num_features, 2);
    assert_eq!(loaded_tree.trees.len(), 2);
    assert_eq!(loaded_tree.base_score, 0.0);
}

#[test]
fn stump_regression_loads_correctly() {
    let loaded_tree: WeirwoodTree = load_fixture_json("stump_regression.json");
    assert_eq!(loaded_tree.num_features, 1);
    assert_eq!(loaded_tree.trees.len(), 1);
    approx::assert_abs_diff_eq!(loaded_tree.base_score, 1.0, epsilon = 1e-6);
}

#[test]
fn missing_file_returns_io_error() {
    let result: Result<WeirwoodTree, weirwood::Error> =
        WeirwoodTree::from_json_file("tests/fixtures/does_not_exist.json");
    assert!(matches!(result, Err(weirwood::Error::Io(_))));
}

#[test]
fn invalid_json_returns_parse_error() {
    let result: Result<WeirwoodTree, weirwood::Error> =
        WeirwoodTree::from_json_bytes(b"not json at all {{{");
    assert!(matches!(result, Err(weirwood::Error::Json(_))));
}

#[test]
fn empty_trees_array_is_valid() {
    // A model with zero trees is odd but structurally legal.
    let json: &str = r#"{
        "learner": {
            "learner_model_param": { "base_score": "2.5", "num_class": "0", "num_feature": "3" },
            "objective": { "name": "reg:squarederror" },
            "gradient_booster": { "model": { "trees": [] } }
        }
    }"#;
    let loaded_tree: WeirwoodTree = WeirwoodTree::from_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(loaded_tree.trees.len(), 0);
    approx::assert_abs_diff_eq!(loaded_tree.base_score, 2.5, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// two_trees_binary.json — binary:logistic
//
// Tree 1: feature[0] <= 1.5 → left(-0.3), right(0.3)
// Tree 2: feature[1] <= 2.0 → left(-0.2), right(0.2)
// base_score = 0.0
//
// Expected raw scores and probabilities (sigmoid):
//   [0.0, 0.0] → raw=-0.5  proba=sigmoid(-0.5)≈0.37754
//   [2.0, 3.0] → raw= 0.5  proba=sigmoid( 0.5)≈0.62246
//   [0.0, 3.0] → raw=-0.1  proba=sigmoid(-0.1)≈0.47502
//   [2.0, 0.0] → raw= 0.1  proba=sigmoid( 0.1)≈0.52498
//   [1.5, 2.0] → raw=-0.5  (boundary: both features exactly at threshold → left)
// ---------------------------------------------------------------------------

#[test]
fn two_trees_raw_scores() {
    let loaded_tree: WeirwoodTree = load_fixture_json("two_trees_binary.json");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;

    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![0.0, 0.0]),
        -0.5,
        epsilon = 1e-6
    );
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![2.0, 3.0]),
        0.5,
        epsilon = 1e-6
    );
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![0.0, 3.0]),
        -0.1,
        epsilon = 1e-6
    );
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![2.0, 0.0]),
        0.1,
        epsilon = 1e-6
    );
}

#[test]
fn two_trees_boundary_conditions() {
    let loaded_tree: WeirwoodTree = load_fixture_json("two_trees_binary.json");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;

    // Both features exactly at threshold → both go left
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![1.5, 2.0]),
        -0.5,
        epsilon = 1e-6
    );
    // Just above both thresholds → both go right
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![1.501, 2.001]),
        0.5,
        epsilon = 1e-5
    );
}

#[test]
fn two_trees_predict_proba() {
    let loaded_tree: WeirwoodTree = load_fixture_json("two_trees_binary.json");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;

    // sigmoid(-0.5) = 1 / (1 + e^0.5) ≈ 0.37754066
    approx::assert_abs_diff_eq!(
        evaluator.predict_proba(&loaded_tree, &vec![0.0, 0.0]),
        0.37754066_f32,
        epsilon = 1e-5
    );
    // sigmoid(0.5) ≈ 0.62245934
    approx::assert_abs_diff_eq!(
        evaluator.predict_proba(&loaded_tree, &vec![2.0, 3.0]),
        0.62245934_f32,
        epsilon = 1e-5
    );
    // sigmoid(-0.1) ≈ 0.47502081
    approx::assert_abs_diff_eq!(
        evaluator.predict_proba(&loaded_tree, &vec![0.0, 3.0]),
        0.47502081_f32,
        epsilon = 1e-5
    );
    // sigmoid(0.1) ≈ 0.52497919
    approx::assert_abs_diff_eq!(
        evaluator.predict_proba(&loaded_tree, &vec![2.0, 0.0]),
        0.52497919_f32,
        epsilon = 1e-5
    );
}

// ---------------------------------------------------------------------------
// stump_regression.json — reg:squarederror
//
// Tree 1: feature[0] <= 1.5 → left(-0.5), right(0.5)
// base_score = 1.0
//
// predict_proba == predict (identity activation)
//   [0.0] → 1.0 + (-0.5) = 0.5
//   [2.0] → 1.0 +   0.5  = 1.5
//   [1.5] → 0.5  (boundary → left)
//   [1.6] → 1.5
// ---------------------------------------------------------------------------

#[test]
fn regression_raw_scores() {
    let loaded_tree: WeirwoodTree = load_fixture_json("stump_regression.json");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;

    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![0.0]),
        0.5,
        epsilon = 1e-6
    );
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![2.0]),
        1.5,
        epsilon = 1e-6
    );
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![1.5]),
        0.5,
        epsilon = 1e-6
    ); // boundary → left
    approx::assert_abs_diff_eq!(
        evaluator.predict(&loaded_tree, &vec![1.6]),
        1.5,
        epsilon = 1e-6
    );
}

#[test]
fn regression_predict_proba_is_identity() {
    let loaded_tree: WeirwoodTree = load_fixture_json("stump_regression.json");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;

    // For regression the activation is identity, so predict == predict_proba.
    let raw_score: f32 = evaluator.predict(&loaded_tree, &vec![2.0]);
    let predicted_proba: f32 = evaluator.predict_proba(&loaded_tree, &vec![2.0]);
    approx::assert_abs_diff_eq!(raw_score, predicted_proba, epsilon = 1e-9);
}

// ---------------------------------------------------------------------------
// UBJ end-to-end tests
//
// Fixtures were produced by XGBoost's Python API and committed to the repo.
// The model was trained on a simple binary-classification dataset separable
// on feature[0] with a threshold near 0.5.
//
// Expected probabilities (P(class=1)) verified against XGBoost Python output:
//   [0.0, 0.0] -> 0.42555749
//   [0.5, 0.5] -> 0.42555749
//   [1.0, 1.0] -> 0.57444251
//   [0.0, 1.0] -> 0.42555749
//   [1.0, 0.0] -> 0.57444251
//   [0.3, 0.7] -> 0.42555749
//   [0.7, 0.3] -> 0.42555749
// ---------------------------------------------------------------------------

const TRAINED_TEST_VECTORS: &[[f32; 2]] = &[
    [0.0, 0.0],
    [0.5, 0.5],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.3, 0.7],
    [0.7, 0.3],
];

const TRAINED_EXPECTED_PROBA: &[f32] = &[
    0.42555749, 0.42555749, 0.57444251, 0.42555749, 0.57444251, 0.42555749, 0.42555749,
];

#[test]
fn trained_json_matches_expected_outputs() {
    let loaded_tree: WeirwoodTree = load_fixture_json("trained_binary.json");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;
    for (features, &expected_proba) in TRAINED_TEST_VECTORS.iter().zip(TRAINED_EXPECTED_PROBA) {
        let predicted_proba: f32 = evaluator.predict_proba(&loaded_tree, &features.to_vec());
        approx::assert_abs_diff_eq!(predicted_proba, expected_proba, epsilon = 1e-5);
    }
}

#[test]
fn trained_ubj_matches_expected_outputs() {
    let loaded_tree: WeirwoodTree =
        WeirwoodTree::from_ubj_file("tests/fixtures/trained_binary.ubj")
            .expect("load trained_binary.ubj");
    let evaluator: PlaintextEvaluator = PlaintextEvaluator;
    for (features, &expected_proba) in TRAINED_TEST_VECTORS.iter().zip(TRAINED_EXPECTED_PROBA) {
        let predicted_proba: f32 = evaluator.predict_proba(&loaded_tree, &features.to_vec());
        approx::assert_abs_diff_eq!(predicted_proba, expected_proba, epsilon = 1e-5);
    }
}

#[test]
fn ubj_and_json_loaders_produce_identical_predictions() {
    let json_loaded_tree: WeirwoodTree = load_fixture_json("trained_binary.json");
    let ubj_loaded_tree: WeirwoodTree =
        WeirwoodTree::from_ubj_file("tests/fixtures/trained_binary.ubj")
            .expect("load trained_binary.ubj");

    assert_eq!(json_loaded_tree.num_features, ubj_loaded_tree.num_features);
    assert_eq!(json_loaded_tree.trees.len(), ubj_loaded_tree.trees.len());
    approx::assert_abs_diff_eq!(
        json_loaded_tree.base_score,
        ubj_loaded_tree.base_score,
        epsilon = 1e-6
    );

    let evaluator: PlaintextEvaluator = PlaintextEvaluator;
    for features in TRAINED_TEST_VECTORS {
        let json_predicted_proba: f32 =
            evaluator.predict_proba(&json_loaded_tree, &features.to_vec());
        let ubj_predicted_proba: f32 =
            evaluator.predict_proba(&ubj_loaded_tree, &features.to_vec());
        approx::assert_abs_diff_eq!(json_predicted_proba, ubj_predicted_proba, epsilon = 1e-6);
    }
}
