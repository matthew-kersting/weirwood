use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse model JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("model format error: {0}")]
    Format(String),

    #[error("FHE error: {0}")]
    Fhe(String),
}
