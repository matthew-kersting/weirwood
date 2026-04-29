//! Minimal demo of the `InferenceService` gRPC contract.
//!
//! Implements the generated [`InferenceService`] trait, registers a single
//! pre-loaded model, and serves it under `tonic::transport::Server`.
//! Production deployments will want to layer their own TLS, interceptors,
//! tracing, and rate limits on top — this binary stays intentionally short
//! so the protocol itself is easy to read.
//!
//! Usage:
//!   cargo run --release --example server --features transport -- \
//!     --model tests/fixtures/trained_binary.ubj [--port 9999]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use weirwood::{
    eval::Evaluator as _,
    fhe::FheEvaluator,
    model::WeirwoodTree,
    transport::{
        InferenceService, InferenceServiceServer, InitSessionRequest, InitSessionResponse,
        PredictRequest, PredictResponse, deserialize_feature, deserialize_server_context,
        serialize_score,
    },
};

/// Generous upper bound on a serialized `ServerKey` (~180 MB for the default
/// `tfhe-rs 1.6` parameter set, plus headroom). Mirrors the limit applied on
/// the client side.
const MAX_GRPC_MESSAGE_BYTES: usize = 512 * 1024 * 1024;

type SessionMap = Arc<Mutex<HashMap<String, FheEvaluator>>>;

struct WeirwoodInference {
    model: Arc<WeirwoodTree>,
    sessions: SessionMap,
}

#[tonic::async_trait]
impl InferenceService for WeirwoodInference {
    async fn init_session(
        &self,
        request: Request<InitSessionRequest>,
    ) -> Result<Response<InitSessionResponse>, Status> {
        let req = request.into_inner();

        let server_ctx = deserialize_server_context(&req.server_key).map_err(|e| {
            Status::invalid_argument(format!("failed to deserialize server key: {e}"))
        })?;

        // FheEvaluator construction installs the key on its worker threads;
        // it does not block on FHE work, but key cloning is a few hundred ms,
        // so we run it in `spawn_blocking` to avoid stalling the runtime.
        let evaluator = tokio::task::spawn_blocking(move || FheEvaluator::new(server_ctx))
            .await
            .map_err(|e| Status::internal(format!("evaluator construction panicked: {e}")))?;

        let session_id = Uuid::new_v4().to_string();
        self.sessions
            .lock()
            .await
            .insert(session_id.clone(), evaluator);

        println!(
            "[init] Session created (server key {} MB)",
            req.server_key.len() / 1_000_000
        );

        Ok(Response::new(InitSessionResponse { session_id }))
    }

    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();

        let mut features = Vec::with_capacity(req.features.len());
        for feature_bytes in &req.features {
            features.push(
                deserialize_feature(feature_bytes)
                    .map_err(|e| Status::invalid_argument(format!("bad feature bytes: {e}")))?,
            );
        }

        let model = Arc::clone(&self.model);
        let sessions = Arc::clone(&self.sessions);
        let session_id = req.session_id.clone();

        // Move the multi-second FHE evaluation onto a blocking thread so the
        // tonic runtime stays responsive. We hold the session lock only long
        // enough to look up the evaluator, then release it before doing the
        // actual work — concurrent predicts on different sessions can run
        // truly in parallel.
        let encrypted_score = tokio::task::spawn_blocking(move || {
            let mut guard = sessions.blocking_lock();
            let evaluator = guard
                .get_mut(&session_id)
                .ok_or_else(|| Status::not_found(format!("session {session_id} not found")))?;
            Ok::<_, Status>(evaluator.predict(&model, &features))
        })
        .await
        .map_err(|e| Status::internal(format!("predict task panicked: {e}")))??;

        let score_bytes = serialize_score(&encrypted_score)
            .map_err(|e| Status::internal(format!("failed to serialize score: {e}")))?;

        println!("[predict] Session completed");

        Ok(Response::new(PredictResponse {
            encrypted_score: score_bytes,
        }))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut model_path = "tests/fixtures/trained_binary.ubj".to_string();
    let mut port: u16 = 9999;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().expect("--model requires a path"),
            "--port" => {
                port = args
                    .next()
                    .expect("--port requires a number")
                    .parse()
                    .expect("invalid port")
            }
            other => eprintln!("unknown argument: {other}"),
        }
    }

    println!("Loading model from {model_path}…");
    let model = if model_path.ends_with(".ubj") {
        WeirwoodTree::from_ubj_file(&model_path)?
    } else {
        WeirwoodTree::from_json_file(&model_path)?
    };
    let warnings = model.validate_for_fhe();
    for w in &warnings {
        eprintln!("warning: {w}");
    }
    if !warnings.is_empty() {
        return Err(format!(
            "model has {} FHE-safety warnings; refusing to start",
            warnings.len()
        )
        .into());
    }
    println!(
        "Model: {} trees, {} features",
        model.trees.len(),
        model.num_features
    );

    let svc = WeirwoodInference {
        model: Arc::new(model),
        sessions: Arc::new(Mutex::new(HashMap::new())),
    };

    let addr: SocketAddr = format!("127.0.0.1:{port}").parse()?;
    println!("Inference server listening on {addr}");

    Server::builder()
        .add_service(
            InferenceServiceServer::new(svc)
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES),
        )
        .serve(addr)
        .await?;

    Ok(())
}
