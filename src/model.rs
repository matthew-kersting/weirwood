//! XGBoost model loading and internal representation.
//!
//! Supports XGBoost models saved in JSON (`booster.save_model("model.json")`)
//! or Universal Binary JSON (`booster.save_model("model.ubj")`) format.

use std::path::Path;

use serde::Deserialize;

use crate::Error;
use crate::eval::fhe::SCALE as FHE_SCALE;

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

    /// Depth of this tree, where a stump (root + 2 leaves) is depth 1.
    pub fn depth(&self) -> usize {
        fn walk(tree: &Tree, node_idx: usize, depth: usize) -> usize {
            let node = &tree.nodes[node_idx];
            if node.is_leaf() {
                depth
            } else {
                let left = walk(tree, node.left_child as usize, depth + 1);
                let right = walk(tree, node.right_child as usize, depth + 1);
                left.max(right)
            }
        }
        walk(self, 0, 0)
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
    /// Parse XGBoost's textual objective name (e.g. `"binary:logistic"`) into
    /// a typed [`Objective`]. `num_class` is only consulted for
    /// `multi:softmax` / `multi:softprob` — pass `0` otherwise.
    pub fn from_str(objective_name: &str, num_class: usize) -> Self {
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
    /// Load an XGBoost model from a path, dispatching by extension:
    /// `.ubj` → Universal Binary JSON, anything else → JSON.
    ///
    /// Save from Python with either `booster.save_model("model.json")` or
    /// `booster.save_model("model.ubj")`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        if path.extension().is_some_and(|e| e == "ubj") {
            Self::from_ubj_file(path)
        } else {
            Self::from_json_file(path)
        }
    }

    /// Load from an XGBoost JSON model file.
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

    /// Maximum [`Tree::depth`] across the ensemble; 0 if there are no trees.
    pub fn max_depth(&self) -> usize {
        self.trees.iter().map(Tree::depth).max().unwrap_or(0)
    }

    /// Inspect the loaded tree for issues that would only manifest under FHE
    /// evaluation but are silently fine for plaintext use.
    ///
    /// Returns an empty vector when the model is safe to evaluate under FHE.
    /// Callers using only [`PlaintextEvaluator`](crate::eval::PlaintextEvaluator)
    /// can ignore the result; callers about to construct an
    /// [`FheEvaluator`](crate::eval::fhe::FheEvaluator) should treat any
    /// warning as a hard error (FHE evaluation will produce wrong results at
    /// affected nodes).
    pub fn validate_for_fhe(&self) -> Vec<LoadWarning> {
        let mut warnings = Vec::new();
        for (tree_index, tree) in self.trees.iter().enumerate() {
            for (node_index, node) in tree.nodes.iter().enumerate() {
                if node.is_leaf() {
                    continue;
                }
                let scaled = node.split_threshold * FHE_SCALE;
                if scaled > i32::MAX as f32 || scaled < i32::MIN as f32 {
                    warnings.push(LoadWarning::ThresholdOverflowsFheRange {
                        tree_index,
                        node_index,
                        threshold: node.split_threshold,
                    });
                }
            }
        }
        warnings
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
            .map(|raw_tree| tree_from_raw(raw_tree, num_features))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(WeirwoodTree {
            trees,
            objective,
            base_score,
            num_features,
        })
    }
}

/// A non-fatal issue discovered while loading a [`WeirwoodTree`].
///
/// Returned by [`WeirwoodTree::validate_for_fhe`] so applications can decide
/// how to react (warn, log, abort) instead of the library writing to stderr
/// behind their backs.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadWarning {
    /// An internal node's threshold, after scaling by the FHE fixed-point
    /// factor, falls outside the `i32` range. FHE evaluation at this node
    /// will compare the encrypted feature against a clamped threshold and
    /// produce an incorrect routing decision. Plaintext evaluation is
    /// unaffected.
    ThresholdOverflowsFheRange {
        tree_index: usize,
        node_index: usize,
        threshold: f32,
    },
}

impl std::fmt::Display for LoadWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadWarning::ThresholdOverflowsFheRange {
                tree_index,
                node_index,
                threshold,
            } => write!(
                f,
                "tree {tree_index}, node {node_index}: split_threshold {threshold} \
                 exceeds i32 range after FHE scaling; encrypted comparisons at this \
                 node will be incorrect"
            ),
        }
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

fn tree_from_raw(raw_tree: RawTree, num_features: usize) -> Result<Tree, Error> {
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

    for (i, node) in nodes.iter().enumerate() {
        if node.is_leaf() {
            continue;
        }
        if node.split_feature as usize >= num_features {
            return Err(Error::Format(format!(
                "node {i}: split_feature {} >= num_features {num_features}",
                node.split_feature
            )));
        }
        if node.left_child < 0 || node.left_child as usize >= node_count {
            return Err(Error::Format(format!(
                "node {i}: left_child {} is out of bounds (node_count={node_count})",
                node.left_child
            )));
        }
        if node.right_child < 0 || node.right_child as usize >= node_count {
            return Err(Error::Format(format!(
                "node {i}: right_child {} is out of bounds (node_count={node_count})",
                node.right_child
            )));
        }
    }

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
