//! FHE circuit evaluator — runs XGBoost inference entirely in FHE.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use tfhe::FheInt32;
use tfhe::ServerKey;
use tfhe::prelude::*;

use crate::Error;
use crate::eval::Evaluator;
use crate::model::WeirwoodTree;

use super::client::{EncryptedScore, encode_fixed_point};
use super::server::ServerContext;

/// Process-wide counter used to give each [`FheEvaluator`] a unique id so that
/// the calling-thread key-installation cache can detect when a different
/// evaluator's key needs to be installed in its place.
static EVALUATOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Id of the [`FheEvaluator`] whose `ServerKey` is currently installed in
    /// this thread's TFHE-rs thread-local slot. `None` means no key has been
    /// installed by `weirwood` on this thread.
    static INSTALLED_EVALUATOR_ID: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Encrypted evaluator — runs XGBoost inference entirely in FHE.
///
/// Constructed from a [`ServerContext`] (which contains only the server key,
/// no private key material). In a real deployment the server receives a
/// `ServerContext` from the client, creates an `FheEvaluator`, and evaluates
/// any number of `EncryptedInput`s without ever learning the plaintext
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
/// // FheEvaluator::new installs the server key on its worker threads;
/// // predict() lazily installs it on the calling thread on first use.
/// let evaluator = FheEvaluator::new(server_ctx);
/// let encrypted_score = evaluator.predict(&model, &ciphertext);
///
/// let score = client.decrypt_score(&encrypted_score);
/// # Ok::<(), weirwood::Error>(())
/// ```
pub struct FheEvaluator {
    id: u64,
    /// `Arc` so cheap clones can be handed to the worker `start_handler`
    /// without copying the ~100-200 MB key blob. `set_server_key` itself
    /// consumes an owned `ServerKey`, paid once per (thread, evaluator) pair.
    server_key: Arc<ServerKey>,
    /// Private pool so the server key can be pre-installed on these specific
    /// threads, avoiding interference between concurrent evaluators that hold
    /// different server keys.
    thread_pool: rayon::ThreadPool,
}

impl FheEvaluator {
    /// Create a new evaluator from a [`ServerContext`], validating the model
    /// for FHE evaluation first.
    ///
    /// Returns `Err(Error::Format)` if `model.validate_for_fhe()` reports any
    /// warnings (e.g. a split threshold that overflows the fixed-point range
    /// — FHE evaluation at such a node would produce wrong results). Use
    /// [`new`](Self::new) directly to skip validation.
    pub fn try_new(model: &WeirwoodTree, ctx: ServerContext) -> Result<Self, Error> {
        let warnings = model.validate_for_fhe();
        if !warnings.is_empty() {
            let summary = warnings
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(Error::Format(format!(
                "model is unsafe for FHE evaluation ({} issue(s)): {summary}",
                warnings.len()
            )));
        }
        Ok(Self::new(ctx))
    }

    /// Create a new evaluator from a [`ServerContext`] **without** running
    /// FHE-safety validation on a model.
    ///
    /// Builds a dedicated Rayon thread pool and installs the server key on
    /// every worker thread. The one-time key broadcast happens here so that
    /// [`predict`](Self::predict) calls pay no per-call broadcast overhead on
    /// pool workers. The calling thread is handled lazily inside `predict`.
    ///
    /// Prefer [`try_new`](Self::try_new) when you have a model in hand;
    /// `new` exists for advanced callers who validate separately.
    pub fn new(ctx: ServerContext) -> Self {
        let id = EVALUATOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let server_key = Arc::new(ctx.server_key);
        let server_key_for_workers = Arc::clone(&server_key);
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .start_handler(move |_| {
                tfhe::set_server_key((*server_key_for_workers).clone());
            })
            .build()
            .expect("failed to build Rayon thread pool for FHE evaluation");
        FheEvaluator {
            id,
            server_key,
            thread_pool,
        }
    }

    /// Install this evaluator's server key on the calling thread, but only if
    /// it isn't already there.
    ///
    /// `set_server_key` consumes the key by value, so the first install on a
    /// given thread pays one ~100-200 MB clone of the key blob.  Subsequent
    /// `predict` calls on the same `(thread, evaluator)` pair are O(1).
    fn ensure_key_installed(&self) {
        INSTALLED_EVALUATOR_ID.with(|cell| {
            if cell.get() != Some(self.id) {
                tfhe::set_server_key((*self.server_key).clone());
                cell.set(Some(self.id));
            }
        });
    }
}

/// Recursively evaluate one tree node in FHE, returning a scaled `FheInt32`.
///
/// Internal nodes spend one programmable bootstrap (~1.1 s per PBS on CPU) on
/// the `feature <= threshold` comparison, then use `if_then_else` (an oblivious
/// mux that doesn't reveal the branch taken) to combine the sub-results. Leaf
/// nodes return a trivially-encrypted constant — no secret material is
/// involved, the trivial encryption just packages the plaintext as a
/// ciphertext so it can be added to real ciphertexts.
fn eval_node(tree: &crate::model::Tree, node_idx: usize, features: &[FheInt32]) -> FheInt32 {
    let node: &crate::model::Node = &tree.nodes[node_idx];
    if node.is_leaf() {
        return FheInt32::encrypt_trivial(encode_fixed_point(node.leaf_value));
    }
    let threshold: i32 = encode_fixed_point(node.split_threshold);
    let go_left: tfhe::FheBool = features[node.split_feature as usize].le(threshold);
    let left_score: FheInt32 = eval_node(tree, node.left_child as usize, features);
    let right_score: FheInt32 = eval_node(tree, node.right_child as usize, features);
    go_left.if_then_else(&left_score, &right_score)
}

impl Evaluator for FheEvaluator {
    type Input = [FheInt32];
    type Output = EncryptedScore;

    /// Run XGBoost inference over an encrypted feature vector.
    ///
    /// Trees are evaluated in parallel using the evaluator's dedicated Rayon
    /// thread pool — each tree is independent, so bootstrapped comparisons
    /// across trees run concurrently.  Tree scores are then accumulated
    /// sequentially into a single [`EncryptedScore`] together with the scaled
    /// `base_score` (accumulation is cheap: only homomorphic additions, no PBS).
    ///
    /// The server key is installed automatically on the calling thread the
    /// first time `predict` runs there, and on every worker thread at
    /// evaluator construction. Callers don't need to manage TFHE thread-local
    /// state themselves.
    fn predict(
        &self,
        weirwood_tree: &WeirwoodTree,
        encrypted_features: &[FheInt32],
    ) -> EncryptedScore {
        // The calling thread can run stolen tasks under `install()` and also
        // performs the trivial encryption and additions below, so it must
        // have this evaluator's server key installed too. Worker threads
        // already got it from `start_handler` in `new()`.
        self.ensure_key_installed();

        let tree_scores: Vec<FheInt32> = self.thread_pool.install(|| {
            weirwood_tree
                .trees
                .par_iter()
                .map(|tree| eval_node(tree, 0, encrypted_features))
                .collect()
        });

        let mut total: FheInt32 =
            FheInt32::encrypt_trivial(encode_fixed_point(weirwood_tree.base_score));
        for score in tree_scores {
            total += score;
        }
        total
    }
}
