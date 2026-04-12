//! FHE circuit evaluator — runs XGBoost inference entirely in FHE.

use rayon::prelude::*;
use tfhe::FheInt32;
use tfhe::prelude::*;

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
///
/// let model = WeirwoodTree::from_json_file("model.json")?;
/// let ciphertext = client.encrypt(&[1.5_f32, 0.3]);
///
/// // FheEvaluator::new installs the server key on its worker threads.
/// let evaluator = FheEvaluator::new(server_ctx);
/// let encrypted_score = evaluator.predict(&model, &ciphertext);
///
/// let score = client.decrypt_score(&encrypted_score);
/// # Ok::<(), weirwood::Error>(())
/// ```
pub struct FheEvaluator {
    // Dedicated Rayon thread pool for parallel tree evaluation.  Using a
    // private pool (rather than the global one) lets us install the server
    // key on exactly these threads at construction time, avoiding interference
    // between concurrent evaluators that hold different server keys.
    thread_pool: rayon::ThreadPool,
}

impl FheEvaluator {
    /// Create a new evaluator from a [`ServerContext`].
    ///
    /// Builds a dedicated Rayon thread pool and installs the server key on
    /// every worker thread.  The one-time key broadcast happens here so that
    /// [`predict`](Self::predict) calls pay no per-call broadcast overhead.
    pub fn new(ctx: ServerContext) -> Self {
        // Clone the server key into the start_handler closure.  The handler
        // runs once per worker thread when it is spawned (Rayon spawns lazily),
        // installing the key into that thread's local storage before any work
        // is dispatched.
        let sk = ctx.server_key.clone();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .start_handler(move |_| tfhe::set_server_key(sk.clone()))
            .build()
            .expect("failed to build Rayon thread pool for FHE evaluation");
        FheEvaluator { thread_pool }
    }
}

// ---------------------------------------------------------------------------
// FHE circuit helpers
// ---------------------------------------------------------------------------

/// Recursively evaluate one tree node in FHE, returning a scaled `FheInt32`.
///
/// Internal nodes perform a bootstrapped comparison (~1.1 s per PBS op on CPU)
/// between an encrypted feature and a plaintext threshold, then use
/// `if_then_else` to select the left or right sub-result.  Leaf nodes return a
/// trivially-encrypted (unrandomised) constant — the scaled leaf weight.
fn eval_node(tree: &crate::model::Tree, node_idx: usize, features: &EncryptedInput) -> FheInt32 {
    let node: &crate::model::Node = &tree.nodes[node_idx];
    if node.is_leaf() {
        // Trivially encrypt the scaled leaf weight.  No secret material is
        // involved; this just wraps the plaintext in the ciphertext format so
        // it can be combined with real ciphertexts via homomorphic operations.
        let scaled: i32 = (node.leaf_value * SCALE)
            .round()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        FheInt32::encrypt_trivial(scaled)
    } else {
        // Scale the plaintext threshold to match the fixed-point encoding of
        // the encrypted features.
        let threshold: i32 = (node.split_threshold * SCALE).round() as i32;
        // One programmable-bootstrapping comparison: encrypted feature vs
        // scalar plaintext threshold.  Returns FheBool (encrypted 0 or 1).
        let go_left: tfhe::FheBool = features[node.split_feature as usize].le(threshold);
        let left_score: FheInt32 = eval_node(tree, node.left_child as usize, features);
        let right_score: FheInt32 = eval_node(tree, node.right_child as usize, features);
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
    /// Trees are evaluated in parallel using the evaluator's dedicated Rayon
    /// thread pool — each tree is independent, so bootstrapped comparisons
    /// across trees run concurrently.  Tree scores are then accumulated
    /// sequentially into a single [`EncryptedScore`] together with the scaled
    /// `base_score` (accumulation is cheap: only homomorphic additions, no PBS).
    ///
    /// The server key must be installed on the **calling thread** via
    /// [`ServerContext::set_active`] before this method is invoked.  Worker
    /// threads in the evaluator's private pool have the key installed
    /// automatically at construction time.
    fn predict(
        &self,
        weirwood_tree: &WeirwoodTree,
        encrypted_features: &EncryptedInput,
    ) -> EncryptedScore {
        // Evaluate all trees in parallel inside the evaluator's private thread
        // pool.  The calling thread also participates in task execution via
        // install(), so it must have the server key set (see set_active()).
        // Pool worker threads already have it from the start_handler in new().
        let tree_scores: Vec<FheInt32> = self.thread_pool.install(|| {
            weirwood_tree
                .trees
                .par_iter()
                .map(|tree| eval_node(tree, 0, encrypted_features))
                .collect()
        });

        let base_scaled = (weirwood_tree.base_score * SCALE)
            .round()
            .clamp(i32::MIN as f32, i32::MAX as f32) as i32;
        let mut total: FheInt32 = FheInt32::encrypt_trivial(base_scaled);
        for score in tree_scores {
            total += score;
        }
        total
    }
}
