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
// Fixtures were produced by benchmarks/train_model.py (100 trees, max_depth=8,
// seed=42) trained on the Breast Cancer Wisconsin dataset (sklearn built-in,
// 569 samples, 30 features, binary: malignant=0 / benign=1).  Features are
// StandardScaler-normalized (mean=0, std=1) before training and inference.
//
// Test vectors are the first 7 samples from the held-out test split.
// Expected probabilities P(benign=1) verified against XGBoost Python output.
// ---------------------------------------------------------------------------

const TRAINED_TEST_VECTORS: &[[f32; 30]] = &[
    [1.568513_f32, 2.164016_f32, 1.742866_f32, 1.620764_f32, -0.265376_f32, 1.915932_f32, 1.092455_f32, 1.698896_f32, 0.309589_f32, -0.069884_f32, 1.658233_f32, -0.034559_f32, 2.199280_f32, 1.609795_f32, -0.232337_f32, 1.198715_f32, 0.187957_f32, 0.559817_f32, -0.162073_f32, 0.523832_f32, 1.862160_f32, 1.708208_f32, 2.170623_f32, 1.913701_f32, -0.187741_f32, 1.744274_f32, 0.688560_f32, 1.202144_f32, -0.140431_f32, 0.900362_f32],
    [-0.840276_f32, -0.597067_f32, -0.874173_f32, -0.776169_f32, -1.090346_f32, -1.225691_f32, -0.924639_f32, -0.891008_f32, -1.052578_f32, -0.187341_f32, -0.962401_f32, -0.440763_f32, -0.940773_f32, -0.708094_f32, -0.383363_f32, -1.044917_f32, -0.716089_f32, -0.872824_f32, -0.078685_f32, -0.511433_f32, -0.943849_f32, -0.861492_f32, -0.973363_f32, -0.810993_f32, -1.302037_f32, -1.203076_f32, -1.069908_f32, -1.101241_f32, -0.814291_f32, -0.713236_f32],
    [-0.070723_f32, 1.191387_f32, 0.032028_f32, -0.152775_f32, 1.490195_f32, 1.201416_f32, 0.569537_f32, 0.494989_f32, 1.671757_f32, 1.335406_f32, 0.284559_f32, 0.541730_f32, 0.076827_f32, -0.004644_f32, 0.828998_f32, 0.705937_f32, 0.068271_f32, 0.714206_f32, 0.406793_f32, 1.151473_f32, -0.035096_f32, 1.172719_f32, -0.018858_f32, -0.146703_f32, 2.089687_f32, 0.890832_f32, 0.285220_f32, 0.572558_f32, 1.149974_f32, 1.868411_f32],
    [0.695970_f32, -0.217560_f32, 0.623587_f32, 0.550139_f32, 0.051519_f32, -0.358921_f32, -0.374411_f32, 0.000156_f32, -1.109632_f32, -0.963397_f32, -0.224542_f32, 0.385834_f32, -0.242758_f32, -0.135765_f32, 0.055347_f32, -0.405931_f32, -0.440162_f32, -0.213724_f32, -0.438511_f32, -0.646828_f32, 0.409835_f32, -0.031727_f32, 0.318913_f32, 0.253429_f32, 0.065307_f32, -0.541172_f32, -0.511259_f32, -0.339891_f32, -0.797251_f32, -1.047159_f32],
    [2.143533_f32, 0.714163_f32, 2.091086_f32, 2.410164_f32, 1.103374_f32, 0.225037_f32, 1.894590_f32, 2.329011_f32, -0.285913_f32, -0.921448_f32, 2.904446_f32, 0.065285_f32, 2.460528_f32, 2.863472_f32, 1.026960_f32, 0.182801_f32, 0.600826_f32, 2.048006_f32, -1.101046_f32, 0.159059_f32, 1.946110_f32, 0.121040_f32, 1.793646_f32, 2.096566_f32, 0.384946_f32, -0.269423_f32, 0.621800_f32, 1.614373_f32, -1.314652_f32, -0.695833_f32],
    [-0.256674_f32, -0.233468_f32, -0.301683_f32, -0.322845_f32, -1.599908_f32, -0.821672_f32, -0.497242_f32, -0.505223_f32, -1.255833_f32, -1.024922_f32, -0.735138_f32, -0.934900_f32, -0.804822_f32, -0.535839_f32, -0.355904_f32, -0.575070_f32, -0.523045_f32, -0.510884_f32, -1.125034_f32, -0.765298_f32, -0.268056_f32, -0.282586_f32, -0.347581_f32, -0.328482_f32, -0.183301_f32, -0.362094_f32, -0.412511_f32, -0.140972_f32, -1.043559_f32, -0.799164_f32],
    [-1.484527_f32, -0.853859_f32, -1.443347_f32, -1.182769_f32, -0.984714_f32, -0.502946_f32, -0.514142_f32, -0.630989_f32, 0.441527_f32, 0.474054_f32, 0.520046_f32, -0.032807_f32, 0.737811_f32, -0.217266_f32, 1.228115_f32, 0.187138_f32, -0.012757_f32, 0.539125_f32, 0.867141_f32, 0.135653_f32, -1.300424_f32, -1.277983_f32, -1.247199_f32, -1.026629_f32, -1.368629_f32, -0.811043_f32, -0.844548_f32, -0.993612_f32, -0.735287_f32, -0.523977_f32],
];

const TRAINED_EXPECTED_PROBA: &[f32] = &[
    0.00134412, 0.99849713, 0.00319455, 0.18419895, 0.00168573, 0.99729341, 0.99910814,
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
