//! High-level convenience client that bundles every step of a
//! privacy-preserving inference round-trip:
//!
//!   1. Open a gRPC connection to a `weirwood` inference server.
//!   2. Generate a fresh FHE keypair locally.
//!   3. Upload the [`ServerKey`](tfhe::ServerKey) via `InitSession` and
//!      remember the session id.
//!   4. On each predict call: encrypt features → `Predict` RPC → decrypt
//!      result → apply the model's activation function.
//!
//! The protocol-level types ([`InferenceServiceClient`](super::InferenceServiceClient),
//! the `Predict` / `InitSession` request and response messages) remain
//! available at `weirwood::transport::*` for callers that need finer control.

use tonic::transport::Channel;
use tonic::{Request, Status};

use crate::Error;
use crate::eval::fhe::ClientContext;
use crate::eval::sigmoid;
use crate::model::{Objective, WeirwoodTree};

use super::rpc::inference_service_client::InferenceServiceClient;
use super::rpc::{InitSessionRequest, PredictRequest};
use super::{
    MAX_GRPC_MESSAGE_BYTES, deserialize_score, serialize_feature, serialize_server_context,
};

/// High-level FHE inference client.
///
/// Holds a connected [`InferenceServiceClient`], a [`ClientContext`] (which
/// owns the private key and never leaves the process), and an active
/// `session_id` that ties subsequent `Predict` calls back to the uploaded
/// server key on the remote.
///
/// Created via [`WeirwoodClient::connect`]. The inner `ClientContext` is not
/// `Clone`, so wrap in `Arc<Mutex<WeirwoodClient>>` if multiple tasks need
/// to share the same key.
pub struct WeirwoodClient {
    grpc: InferenceServiceClient<Channel>,
    fhe: ClientContext,
    session_id: String,
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

        Ok(Self {
            grpc,
            fhe,
            session_id: resp.session_id,
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
    pub async fn predict_proba(
        &mut self,
        model: &WeirwoodTree,
        features: &[f32],
    ) -> Result<f32, Error> {
        let raw = self.predict_raw(features).await?;
        match &model.objective {
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
    /// For multi-class models the server will eventually return one element
    /// per class; until then this method takes the first element of the
    /// response and errors if the response is empty.
    pub async fn predict_raw(&mut self, features: &[f32]) -> Result<f32, Error> {
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

    /// Borrow the server-assigned session id (handy for logging).
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

fn status_to_error(status: Status) -> Error {
    Error::Transport(format!("gRPC error: {status}"))
}
