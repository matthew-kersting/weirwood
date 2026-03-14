//! XGBoost model loading and internal representation.
//!
//! Supports XGBoost models saved in JSON (`booster.save_model("model.json")`)
//! or Universal Binary JSON (`booster.save_model("model.ubj")`) format.

use std::path::Path;

use serde::Deserialize;

use crate::Error;

// ---------------------------------------------------------------------------
// Public IR types
// ---------------------------------------------------------------------------

/// A single node in a regression tree.
///
/// Leaf nodes have `left_child == right_child == -1`; their prediction
/// contribution is `leaf_value`. Internal nodes route left when
/// `features[split_feature] <= split_threshold`, right otherwise.
#[derive(Debug, Clone)]
pub struct Node {
    pub split_feature: u32,
    pub split_threshold: f32,
    pub left_child: i32,
    pub right_child: i32,
    /// Leaf weight; only meaningful when `is_leaf()` returns true.
    pub leaf_value: f32,
}

impl Node {
    pub fn is_leaf(&self) -> bool {
        self.left_child == -1
    }
}

/// A single regression tree in the ensemble.
#[derive(Debug, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    /// Walk the tree for a plaintext feature vector and return the leaf value.
    pub fn evaluate(&self, features: &[f32]) -> f32 {
        let mut idx: usize = 0usize;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf() {
                return node.leaf_value;
            }
            let feature_val = features[node.split_feature as usize];
            idx = if feature_val <= node.split_threshold {
                node.left_child as usize
            } else {
                node.right_child as usize
            };
        }
    }
}

/// The prediction task the ensemble was trained for.
#[derive(Debug, Clone)]
pub enum Objective {
    BinaryLogistic,
    RegSquaredError,
    MultiSoftmax {
        num_class: usize,
    },
    /// Catch-all for objectives not yet explicitly handled.
    Other(String),
}

impl Objective {
    fn from_str(s: &str, num_class: usize) -> Self {
        match s {
            "binary:logistic" => Self::BinaryLogistic,
            "reg:squarederror" | "reg:linear" => Self::RegSquaredError,
            "multi:softmax" | "multi:softprob" => Self::MultiSoftmax { num_class },
            other => Self::Other(other.to_owned()),
        }
    }
}

/// The full boosted tree ensemble, ready for inference.
#[derive(Debug, Clone)]
pub struct Ensemble {
    pub trees: Vec<Tree>,
    pub objective: Objective,
    /// Global bias added before the activation function.
    pub base_score: f32,
    pub num_features: usize,
}

impl Ensemble {
    /// Load from an XGBoost JSON model file.
    ///
    /// Save from Python with `booster.save_model("model.json")`.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes = std::fs::read(path)?;
        Self::from_json_bytes(&bytes)
    }

    /// Load from raw JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let raw: RawModel = serde_json::from_slice(bytes)?;
        Self::from_raw(raw)
    }

    /// Load from an XGBoost UBJ (Universal Binary JSON) model file.
    ///
    /// Save from Python with `booster.save_model("model.ubj")`.
    pub fn from_ubj_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes: Vec<u8> = std::fs::read(path)?;
        Self::from_ubj_bytes(&bytes)
    }

    /// Load from raw UBJ bytes.
    pub fn from_ubj_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let value: serde_json::Value = crate::ubj::parse(bytes)?;
        let raw: RawModel = serde_json::from_value(value)?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawModel) -> Result<Self, Error> {
        let learner: RawLearner = raw.learner;

        let num_features = learner
            .learner_model_param
            .num_feature
            .parse::<usize>()
            .map_err(|_| Error::Format("invalid num_feature".into()))?;

        let base_score: f32 = parse_base_score(&learner.learner_model_param.base_score)?;

        let num_class: usize = learner
            .learner_model_param
            .num_class
            .parse::<usize>()
            .unwrap_or(0);

        let objective: Objective = Objective::from_str(&learner.objective.name, num_class);

        let raw_trees = learner
            .gradient_booster
            .model
            .trees
            .ok_or_else(|| Error::Format("missing trees array".into()))?;

        let trees = raw_trees
            .into_iter()
            .map(tree_from_raw)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Ensemble {
            trees,
            objective,
            base_score,
            num_features,
        })
    }
}

/// Parse the `base_score` field.
///
/// XGBoost >= 1.6 stores `base_score` in **probability space** wrapped in
/// brackets, e.g. `"[5E-1]"`.  The raw-score contribution for inference is
/// `logit(p) = ln(p / (1 - p))`.  For the default `p = 0.5` this is exactly
/// zero, meaning the bias has already been absorbed into the tree leaf weights.
///
/// Older versions store a plain float string (e.g. `"0.5"`) that is already
/// in raw-score (logit) space and is added directly.
fn parse_base_score(s: &str) -> Result<f32, Error> {
    let trimmed: &str = s.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let prob = trimmed[1..trimmed.len() - 1]
            .parse::<f32>()
            .map_err(|_| Error::Format(format!("invalid base_score: {s:?}")))?;
        // Convert from probability space to logit (raw-score) space.
        Ok((prob / (1.0 - prob)).ln())
    } else {
        trimmed
            .parse::<f32>()
            .map_err(|_| Error::Format(format!("invalid base_score: {s:?}")))
    }
}

fn tree_from_raw(raw: RawTree) -> Result<Tree, Error> {
    let n = raw.left_children.len();
    if raw.right_children.len() != n
        || raw.split_conditions.len() != n
        || raw.split_indices.len() != n
        || raw.base_weights.len() != n
    {
        return Err(Error::Format(
            "tree arrays have inconsistent lengths".into(),
        ));
    }

    let nodes: Vec<Node> = (0..n)
        .map(|i| Node {
            split_feature: raw.split_indices[i],
            split_threshold: raw.split_conditions[i],
            left_child: raw.left_children[i],
            right_child: raw.right_children[i],
            leaf_value: raw.base_weights[i],
        })
        .collect();

    Ok(Tree { nodes })
}

// ---------------------------------------------------------------------------
// Serde types for the XGBoost JSON format
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawModel {
    learner: RawLearner,
}

#[derive(Deserialize)]
struct RawLearner {
    gradient_booster: RawBooster,
    learner_model_param: RawModelParam,
    objective: RawObjective,
}

#[derive(Deserialize)]
struct RawBooster {
    model: RawBoosterModel,
}

#[derive(Deserialize)]
struct RawBoosterModel {
    trees: Option<Vec<RawTree>>,
}

#[derive(Deserialize)]
struct RawModelParam {
    /// Stored as a string in the JSON format ("5E-1", "4", etc.)
    #[serde(default = "default_base_score")]
    base_score: String,
    #[serde(default = "default_zero_str")]
    num_class: String,
    #[serde(default = "default_zero_str")]
    num_feature: String,
}

fn default_base_score() -> String {
    "0.5".into()
}
fn default_zero_str() -> String {
    "0".into()
}

#[derive(Deserialize)]
struct RawObjective {
    name: String,
}

#[derive(Deserialize)]
struct RawTree {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_conditions: Vec<f32>,
    split_indices: Vec<u32>,
    base_weights: Vec<f32>,
}
