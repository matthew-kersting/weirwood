//! `weirwood` — privacy-preserving XGBoost inference via Fully Homomorphic Encryption.
//!
//! Load a trained XGBoost model and evaluate it either in plaintext (for testing)
//! or encrypted under FHE so the server learns nothing about the input.
//!
//! # Quickstart — plaintext
//!
//! ```no_run
//! use weirwood::{model::WeirwoodTree, eval::PlaintextEvaluator};
//!
//! // `from_file` dispatches by extension: `.ubj` → Universal Binary JSON,
//! // anything else → JSON.
//! let model = WeirwoodTree::from_file("model.ubj")?;
//! let proba = PlaintextEvaluator.predict_proba(&model, &[1.0_f32, 0.5, 3.2, 0.1])?;
//! # Ok::<(), weirwood::Error>(())
//! ```
//!
//! # Quickstart — encrypted gRPC inference
//!
//! With the `transport` Cargo feature enabled. The server holds the XGBoost
//! model and reports its shape during session setup, so the client never
//! has to load the model file:
//!
//! ```no_run
//! # #[cfg(feature = "transport")]
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use weirwood::transport::WeirwoodClient;
//!
//! let mut client = WeirwoodClient::connect("http://127.0.0.1:9999").await?;
//! let proba = client.predict_proba(&[1.0_f32, 0.5, 3.2, 0.1]).await?;
//! # Ok(()) }
//! ```

pub mod error;
pub mod eval;
pub mod model;
pub(crate) mod ubj;

#[cfg(feature = "transport")]
pub mod transport;

/// Re-exported from [`eval::fhe`] for convenience.
pub use eval::fhe;

pub use error::Error;
