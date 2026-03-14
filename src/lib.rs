//! `weirwood` — privacy-preserving XGBoost inference via Fully Homomorphic Encryption.
//!
//! Load a trained XGBoost model and evaluate it either in plaintext (for testing)
//! or encrypted under FHE so the server learns nothing about the input.
//!
//! # Quickstart
//!
//! ```no_run
//! use weirwood::{model::WeirwoodTree, eval::{Evaluator, PlaintextEvaluator}};
//!
//! let weirwood_tree = WeirwoodTree::from_json_file("model.json")?;
//! let features = vec![1.0_f32, 0.5, 3.2, 0.1];
//! let score = PlaintextEvaluator.predict(&weirwood_tree, &features);
//! # Ok::<(), weirwood::Error>(())
//! ```

pub mod error;
pub mod eval;
pub mod fhe;
pub mod model;
pub(crate) mod ubj;

pub use error::Error;
