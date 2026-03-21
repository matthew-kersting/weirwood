//! FHE circuit evaluator — runs XGBoost inference entirely in FHE.

use tfhe::prelude::*;
use tfhe::{FheInt16, FheInt32};

use crate::eval::Evaluator;
use crate::model::WeirwoodTree;

use super::client::{EncryptedInput, EncryptedScore, SCALE};
use super::server::ServerContext;

// ---------------------------------------------------------------------------
// FheEvaluator
// ---------------------------------------------------------------------------

/// Encrypted evaluator — runs XGBoost inference entirely in FHE.
///
/// Constructed from a [`ServerContext`] (which contains only the server key,
/// no private key material).  In a real deployment the server receives a
/// `ServerContext` from the client, creates an `FheEvaluator`, and evaluates
/// any number of [`EncryptedInput`]s without ever learning the plaintext
/// features or scores.
///
/// # Example
///
/// ```no_run
/// use weirwood::fhe::{ClientContext, FheEvaluator};
/// use weirwood::eval::Evaluator as _;
/// use weirwood::model::WeirwoodTree;
///
/// let client = ClientContext::generate()?;
/// let server_ctx = client.server_context();
/// server_ctx.set_active();
///
/// let model = WeirwoodTree::from_json_file("model.json")?;
/// let ciphertext = client.encrypt(&[1.5_f32, 0.3]);
///
/// let evaluator = FheEvaluator::new(server_ctx);
/// let encrypted_score = evaluator.predict(&model, &ciphertext);
///
/// let score = client.decrypt_score(&encrypted_score);
/// # Ok::<(), weirwood::Error>(())
/// ```
pub struct FheEvaluator {
    // ServerContext holds only the ServerKey — no private key material.
    _ctx: ServerContext,
}

impl FheEvaluator {
    /// Create a new evaluator from a [`ServerContext`].
    ///
    /// The server key must already be installed via [`ServerContext::set_active`]
    /// on the calling thread before [`predict`](Self::predict) is invoked.
    pub fn new(ctx: ServerContext) -> Self {
        FheEvaluator { _ctx: ctx }
    }
}

// ---------------------------------------------------------------------------
// FHE circuit helpers
// ---------------------------------------------------------------------------

/// Recursively evaluate one tree node in FHE, returning a scaled `FheInt16`.
///
/// Internal nodes perform a bootstrapped comparison between an encrypted
/// feature and a plaintext threshold, then use `if_then_else` to select the
/// left or right sub-result.  Leaf nodes return a trivially-encrypted
/// (unrandomised) constant — the scaled leaf weight.
///
/// The return type is `FheInt16` rather than `FheInt32` because
/// `FheBool::if_then_else` on 32-bit ciphertexts triggers an integer overflow
/// inside tfhe-rs's PBS accumulator size calculation (the 16-block radix
/// representation overflows a usize intermediate).  Single-tree leaf values
/// scaled by [`SCALE`] easily fit in `i16` (range ±327), so this is safe.
/// The caller widens to `FheInt32` before accumulating across trees.
fn eval_node(tree: &crate::model::Tree, node_idx: usize, features: &EncryptedInput) -> FheInt16 {
    let node: &crate::model::Node = &tree.nodes[node_idx];
    if node.is_leaf() {
        // Trivially encrypt the scaled leaf weight.  No secret material is
        // involved; this just wraps the plaintext in the ciphertext format so
        // it can be combined with real ciphertexts via homomorphic operations.
        let scaled: i16 = (node.leaf_value * SCALE)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        FheInt16::encrypt_trivial(scaled)
    } else {
        // Scale the plaintext threshold to match the fixed-point encoding of
        // the encrypted features.
        let threshold: i16 = (node.split_threshold * SCALE).round() as i16;
        // Wrap the threshold as a trivial ciphertext so the comparison uses
        // the ciphertext-vs-ciphertext path (comparison.rs).  The scalar path
        // (scalar_comparison.rs) has a bug in tfhe-rs 0.10.0: its borrow-
        // propagation state_fn returns values up to 124, and 124 × delta
        // (= 124 × 2^59) overflows u64 in fill_accumulator.  The ciphertext
        // path keeps state values ≤ 16, so 16 × 2^59 = 2^63 stays in range.
        let threshold_ct: FheInt16 = FheInt16::encrypt_trivial(threshold);
        // One programmable-bootstrapping comparison: encrypted feature vs
        // trivially-encrypted threshold.  Returns FheBool (encrypted 0 or 1).
        let go_left: tfhe::FheBool = features[node.split_feature as usize].le(&threshold_ct);
        let left_score: FheInt16 = eval_node(tree, node.left_child as usize, features);
        let right_score: FheInt16 = eval_node(tree, node.right_child as usize, features);
        // Oblivious mux: selects left_score when go_left=1, right_score
        // otherwise, without revealing the branch taken.
        go_left.if_then_else(&left_score, &right_score)
    }
}

impl Evaluator for FheEvaluator {
    type Input = EncryptedInput;
    type Output = EncryptedScore;

    /// Run XGBoost inference over an encrypted feature vector.
    ///
    /// Each tree is evaluated by recursively applying bootstrapped comparisons
    /// and oblivious muxes.  Tree scores are accumulated
    /// into a single [`EncryptedScore`] together with the scaled `base_score`.
    ///
    /// The server key must be installed via [`ServerContext::set_active`] on
    /// the calling thread before this method is invoked.
    fn predict(
        &self,
        weirwood_tree: &WeirwoodTree,
        encrypted_features: &EncryptedInput,
    ) -> EncryptedScore {
        // Start with the scaled base_score as a trivial ciphertext so that
        // subsequent additions stay in the same ciphertext domain.
        let base_scaled = (weirwood_tree.base_score * SCALE)
            .round()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        let mut total: FheInt32 = FheInt32::encrypt_trivial(base_scaled);

        for tree in &weirwood_tree.trees {
            // eval_node returns FheInt16; widen before adding to the i32 accumulator.
            let tree_score: FheInt16 = eval_node(tree, 0, encrypted_features);
            let tree_score_i32: FheInt32 = FheInt32::cast_from(tree_score);
            total += tree_score_i32;
        }
        total
    }
}
