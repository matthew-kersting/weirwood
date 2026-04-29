//! High-level convenience client that bundles every step of a
//! privacy-preserving inference round-trip:
//!
//!   1. Open a gRPC connection to a `weirwood` inference server.
//!   2. Generate a fresh FHE keypair locally.
//!   3. Upload the [`ServerKey`](tfhe::ServerKey) via `InitSession` and
//!      remember the session id and the [`ModelInfo`](super::ModelInfo) the
//!      server reports.
//!   4. On each predict call: encrypt features → `Predict` RPC → decrypt
//!      result → apply the model's activation function.
//!
//! Because the server reports the model's feature count and objective at
//! session-init time, the client never has to load the XGBoost model itself.
//!
//! The protocol-level types ([`InferenceServiceClient`](super::InferenceServiceClient),
//! the `Predict` / `InitSession` request and response messages) remain
//! available at `weirwood::transport::*` for callers that need finer control.

use tonic::transport::Channel;
use tonic::{Request, Status};

use crate::Error;
use crate::eval::fhe::ClientContext;
use crate::eval::sigmoid;
use crate::model::Objective;

use super::rpc::inference_service_client::InferenceServiceClient;
use super::rpc::{InitSessionRequest, PredictRequest};
use super::{
    MAX_GRPC_MESSAGE_BYTES, deserialize_score, serialize_feature, serialize_server_context,
};

/// High-level FHE inference client.
///
/// Holds a connected [`InferenceServiceClient`], a [`ClientContext`] (which
/// owns the private key and never leaves the process), the active
/// `session_id`, and the model metadata the server reported at session
/// setup.
///
/// Created via [`WeirwoodClient::connect`]. The inner `ClientContext` is not
/// `Clone`, so wrap in `Arc<Mutex<WeirwoodClient>>` if multiple tasks need
/// to share the same key.
pub struct WeirwoodClient {
    grpc: InferenceServiceClient<Channel>,
    fhe: ClientContext,
    session_id: String,
    num_features: usize,
    objective: Objective,
}

impl WeirwoodClient {
    /// Connect to a remote `weirwood` inference server, generate a fresh FHE
    /// keypair, and register the session with the server.
    ///
    /// `dst` is anything `tonic::transport::Endpoint::from_shared` accepts,
    /// e.g. `"http://127.0.0.1:9999"`.
    ///
    /// This is the slow setup step: key generation alone takes 1-3 s and
    /// uploading the ~180 MB serialized `ServerKey` typically dominates over
    /// localhost. Subsequent [`predict_proba`](Self::predict_proba) calls
    /// reuse the same session.
    pub async fn connect(dst: impl Into<String>) -> Result<Self, Error> {
        let endpoint = tonic::transport::Endpoint::from_shared(dst.into())
            .map_err(|e| Error::Transport(format!("invalid server endpoint: {e}")))?;
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Error::Transport(format!("failed to connect to inference server: {e}")))?;

        let mut grpc = InferenceServiceClient::new(channel)
            .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);

        let fhe = ClientContext::generate()?;
        let server_key_bytes = serialize_server_context(&fhe.server_context())?;

        let req = Request::new(InitSessionRequest {
            server_key: server_key_bytes,
        });
        let resp = grpc
            .init_session(req)
            .await
            .map_err(status_to_error)?
            .into_inner();

        let info = resp.model_info.ok_or_else(|| {
            Error::Transport(
                "InitSessionResponse missing model_info — server is too old or out of spec"
                    .to_string(),
            )
        })?;
        let objective = Objective::from_str(&info.objective_name, info.num_class as usize);

        Ok(Self {
            grpc,
            fhe,
            session_id: resp.session_id,
            num_features: info.num_features as usize,
            objective,
        })
    }

    /// Run a single inference end-to-end and return the activated probability
    /// (binary classification) or raw score (regression).
    ///
    /// `multi:softmax` is not supported here because the FHE evaluator
    /// currently returns a single ensemble sum, not per-class scores; calling
    /// this on a multi-class model returns `Err(Error::Format)`. Use
    /// [`predict_raw`](Self::predict_raw) if you need to apply your own
    /// activation.
    pub async fn predict_proba(&mut self, features: &[f32]) -> Result<f32, Error> {
        let raw = self.predict_raw(features).await?;
        match &self.objective {
            Objective::BinaryLogistic => Ok(sigmoid(raw)),
            Objective::RegSquaredError | Objective::Other(_) => Ok(raw),
            Objective::MultiSoftmax { num_class } => Err(Error::Format(format!(
                "WeirwoodClient::predict_proba does not support multi:softmax \
                 (num_class={num_class}); the FHE evaluator returns a single \
                 ensemble sum, not per-class scores"
            ))),
        }
    }

    /// Run inference and return the raw (pre-activation) decrypted score.
    /// Useful when the caller wants to apply its own activation.
    ///
    /// Returns `Err(Error::Format)` if `features.len()` doesn't match the
    /// number of features the server reported at session setup, and
    /// `Err(Error::Transport)` if the server returns an empty response.
    pub async fn predict_raw(&mut self, features: &[f32]) -> Result<f32, Error> {
        if features.len() != self.num_features {
            return Err(Error::Format(format!(
                "model expects {} features, got {}",
                self.num_features,
                features.len()
            )));
        }

        let encrypted = self.fhe.encrypt(features);
        let mut feature_bytes = Vec::with_capacity(encrypted.len());
        for feat in &encrypted {
            feature_bytes.push(serialize_feature(feat)?);
        }

        let req = Request::new(PredictRequest {
            session_id: self.session_id.clone(),
            features: feature_bytes,
        });
        let resp = self
            .grpc
            .predict(req)
            .await
            .map_err(status_to_error)?
            .into_inner();

        let first = resp.encrypted_scores.first().ok_or_else(|| {
            Error::Transport("server returned empty encrypted_scores".to_string())
        })?;
        let encrypted_score = deserialize_score(first)?;
        Ok(self.fhe.decrypt_score(&encrypted_score))
    }

    /// Server-assigned session id (handy for logging).
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Number of features the server's model expects per `Predict` call.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Objective the server's model was trained for (drives the activation
    /// applied by [`predict_proba`](Self::predict_proba)).
    pub fn objective(&self) -> &Objective {
        &self.objective
    }
}

fn status_to_error(status: Status) -> Error {
    Error::Transport(format!("gRPC error: {status}"))
}
