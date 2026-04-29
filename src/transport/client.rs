//! High-level convenience client that bundles every step of a
//! privacy-preserving inference round-trip:
//!
//!   1. Open a gRPC connection to a `weirwood` inference server.
//!   2. Generate a fresh FHE keypair locally.
//!   3. Upload the [`ServerKey`] via `InitSession` and remember the session id.
//!   4. On each [`predict_proba`] call: encrypt features → `Predict` RPC →
//!      decrypt result → apply the model's activation function.
//!
//! The protocol-level types ([`InferenceServiceClient`], the `Predict` /
//! `InitSession` request and response messages) remain available at
//! `weirwood::transport::*` for callers that need finer control.
//!
//! [`ServerKey`]: tfhe::ServerKey
//! [`InferenceServiceClient`]: super::InferenceServiceClient
//! [`predict_proba`]: WeirwoodClient::predict_proba

use tonic::transport::Channel;
use tonic::{Request, Status};

use crate::Error;
use crate::eval::PlaintextEvaluator;
use crate::eval::fhe::ClientContext;
use crate::model::{Objective, WeirwoodTree};

use super::rpc::inference_service_client::InferenceServiceClient;
use super::rpc::{InitSessionRequest, PredictRequest};
use super::{deserialize_score, serialize_feature, serialize_server_context};

/// Generous upper bound on a serialized `ServerKey` (~180 MB for the default
/// `tfhe-rs 1.6` parameter set, plus headroom for future parameter changes
/// and protobuf overhead). Applied as `max_decoding_message_size` /
/// `max_encoding_message_size` on both ends of the gRPC connection so the
/// large `InitSession` payload isn't rejected by tonic's 4 MB default.
const MAX_GRPC_MESSAGE_BYTES: usize = 512 * 1024 * 1024;

/// High-level FHE inference client.
///
/// Holds a connected [`InferenceServiceClient`], a [`ClientContext`] (which
/// owns the private key and never leaves the process), and an active
/// `session_id` that ties subsequent `Predict` calls back to the uploaded
/// server key on the remote.
///
/// Created via [`WeirwoodClient::connect`]. Cheap to clone-via-Arc-on-Channel
/// if you need fan-out, but the inner `ClientContext` is not `Clone`, so
/// share via `Arc<WeirwoodClient>` if multiple tasks must encrypt against
/// the same key.
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
            .map_err(|e| Error::Other(format!("invalid server endpoint: {e}")))?;
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Error::Other(format!("failed to connect to inference server: {e}")))?;

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
    /// (or raw score for regression objectives).
    ///
    /// For `binary:logistic` the returned value is `sigmoid(raw_score)`; for
    /// `reg:squarederror` it is the raw score; for `multi:softmax` this
    /// method panics — use [`predict_multiclass_proba`](Self::predict_multiclass_proba)
    /// instead.
    pub async fn predict_proba(
        &mut self,
        model: &WeirwoodTree,
        features: &[f32],
    ) -> Result<f32, Error> {
        let raw = self.predict_raw(features).await?;
        Ok(match &model.objective {
            Objective::BinaryLogistic => sigmoid(raw),
            Objective::RegSquaredError => raw,
            Objective::MultiSoftmax { num_class } => panic!(
                "predict_proba returns a single f32; multi:softmax with num_class={} \
                 produces a vector — use predict_multiclass_proba instead",
                num_class
            ),
            Objective::Other(_) => raw,
        })
    }

    /// Multi-class variant: returns one probability per class for a
    /// `multi:softmax` / `multi:softprob` model.
    ///
    /// Panics if `model.objective` is not `MultiSoftmax`.
    pub async fn predict_multiclass_proba(
        &mut self,
        model: &WeirwoodTree,
        features: &[f32],
    ) -> Result<Vec<f32>, Error> {
        // The server returns the raw ensemble sum — for multi:softmax this is
        // the per-class logits collapsed into one cipher- text by the FHE
        // evaluator, which doesn't yet split per class. Until the FHE side
        // can return per-class scores we fall back to the plaintext multi-
        // class evaluator over the raw score; this assumes a single-class
        // model encoded as multi-class. Revisit when the FHE evaluator gains
        // a multi-output predict.
        let _ = self.predict_raw(features).await?;
        Ok(PlaintextEvaluator.predict_multiclass_proba(model, features))
    }

    /// Run inference and return the raw (pre-activation) decrypted score.
    /// Useful when the caller wants to apply its own activation.
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

        let encrypted_score = deserialize_score(&resp.encrypted_score)?;
        Ok(self.fhe.decrypt_score(&encrypted_score))
    }

    /// Borrow the server-assigned session id (handy for logging).
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

fn status_to_error(status: Status) -> Error {
    Error::Other(format!("gRPC error: {status}"))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
