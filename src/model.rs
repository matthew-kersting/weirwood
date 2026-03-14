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

/// A single regression tree in the boosted ensemble.
#[derive(Debug, Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    /// Walk the tree for a plaintext feature vector and return the leaf value.
    pub fn evaluate(&self, features: &[f32]) -> f32 {
        let mut current_node_index: usize = 0;
        loop {
            let current_node: &Node = &self.nodes[current_node_index];
            if current_node.is_leaf() {
                return current_node.leaf_value;
            }
            let feature_value: f32 = features[current_node.split_feature as usize];
            current_node_index = if feature_value <= current_node.split_threshold {
                current_node.left_child as usize
            } else {
                current_node.right_child as usize
            };
        }
    }
}

/// The prediction task the model was trained for.
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
    fn from_str(objective_name: &str, num_class: usize) -> Self {
        match objective_name {
            "binary:logistic" => Self::BinaryLogistic,
            "reg:squarederror" | "reg:linear" => Self::RegSquaredError,
            "multi:softmax" | "multi:softprob" => Self::MultiSoftmax { num_class },
            other => Self::Other(other.to_owned()),
        }
    }
}

/// A fully loaded XGBoost boosted tree ensemble, ready for inference.
#[derive(Debug, Clone)]
pub struct WeirwoodTree {
    pub trees: Vec<Tree>,
    pub objective: Objective,
    /// Global bias added before the activation function.
    pub base_score: f32,
    pub num_features: usize,
}

impl WeirwoodTree {
    /// Load from an XGBoost JSON model file.
    ///
    /// Save from Python with `booster.save_model("model.json")`.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let bytes: Vec<u8> = std::fs::read(path)?;
        Self::from_json_bytes(&bytes)
    }

    /// Load from raw JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let raw_model: RawModel = serde_json::from_slice(bytes)?;
        Self::from_raw(raw_model)
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
        let json_value: serde_json::Value = crate::ubj::parse(bytes)?;
        let raw_model: RawModel = serde_json::from_value(json_value)?;
        Self::from_raw(raw_model)
    }

    fn from_raw(raw_model: RawModel) -> Result<Self, Error> {
        let learner: RawLearner = raw_model.learner;

        let num_features: usize = learner
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

        let serialized_trees: Vec<RawTree> = learner
            .gradient_booster
            .model
            .trees
            .ok_or_else(|| Error::Format("missing trees array".into()))?;

        let trees: Vec<Tree> = serialized_trees
            .into_iter()
            .map(tree_from_raw)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(WeirwoodTree {
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
fn parse_base_score(raw_base_score: &str) -> Result<f32, Error> {
    let trimmed_score: &str = raw_base_score.trim();
    if trimmed_score.starts_with('[') && trimmed_score.ends_with(']') {
        let base_probability: f32 = trimmed_score[1..trimmed_score.len() - 1]
            .parse::<f32>()
            .map_err(|_| Error::Format(format!("invalid base_score: {raw_base_score:?}")))?;
        // Convert from probability space to logit (raw-score) space.
        Ok((base_probability / (1.0 - base_probability)).ln())
    } else {
        trimmed_score
            .parse::<f32>()
            .map_err(|_| Error::Format(format!("invalid base_score: {raw_base_score:?}")))
    }
}

fn tree_from_raw(raw_tree: RawTree) -> Result<Tree, Error> {
    let node_count: usize = raw_tree.left_children.len();
    if raw_tree.right_children.len() != node_count
        || raw_tree.split_conditions.len() != node_count
        || raw_tree.split_indices.len() != node_count
        || raw_tree.base_weights.len() != node_count
    {
        return Err(Error::Format(
            "tree arrays have inconsistent lengths".into(),
        ));
    }

    let nodes: Vec<Node> = (0..node_count)
        .map(|node_index| Node {
            split_feature: raw_tree.split_indices[node_index],
            split_threshold: raw_tree.split_conditions[node_index],
            left_child: raw_tree.left_children[node_index],
            right_child: raw_tree.right_children[node_index],
            leaf_value: raw_tree.base_weights[node_index],
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
