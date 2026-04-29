//! Minimal demo of the `InferenceService` gRPC contract.
//!
//! Implements the generated [`InferenceService`] trait, registers a single
//! pre-loaded model, and serves it under `tonic::transport::Server`.
//! Production deployments will want to layer their own TLS, interceptors,
//! tracing, rate limits, and authentication on top — this binary stays
//! intentionally short so the protocol itself is easy to read.
//!
//! Each session pins ~100–200 MB of `ServerKey` in memory, so the example
//! runs a periodic sweep that evicts sessions idle for more than
//! `SESSION_IDLE_TIMEOUT_SECS` seconds. Override with `--idle-timeout-secs`.
//!
//! Usage:
//!   cargo run --release --example server --features transport -- \
//!     --model tests/fixtures/trained_binary.ubj [--port 9999] \
//!     [--idle-timeout-secs 600]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        MAX_GRPC_MESSAGE_BYTES, ModelInfo, PredictRequest, PredictResponse, deserialize_feature,
        deserialize_server_context, serialize_score,
    },
};

/// Default idle window before a session is evicted. Each session pins a
/// ~100–200 MB `ServerKey`, so an unbounded map will OOM a long-running server.
const SESSION_IDLE_TIMEOUT_SECS: u64 = 600;

/// How often the eviction sweep runs.
const SESSION_SWEEP_INTERVAL: Duration = Duration::from_secs(60);

struct Session {
    evaluator: FheEvaluator,
    last_used: Instant,
}

type SessionMap = Arc<Mutex<HashMap<String, Session>>>;

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
        // it doesn't run FHE work but it does clone a ~100 MB key, so push it
        // off the runtime to avoid stalling the reactor.
        let model = Arc::clone(&self.model);
        let evaluator =
            tokio::task::spawn_blocking(move || FheEvaluator::try_new(&model, server_ctx))
                .await
                .map_err(|e| Status::internal(format!("evaluator construction panicked: {e}")))?
                .map_err(|e| Status::failed_precondition(format!("model rejected for FHE: {e}")))?;

        let session_id = Uuid::new_v4().to_string();
        self.sessions.lock().await.insert(
            session_id.clone(),
            Session {
                evaluator,
                last_used: Instant::now(),
            },
        );

        println!(
            "[init] Session created (server key {} MB)",
            req.server_key.len() / 1_000_000
        );

        Ok(Response::new(InitSessionResponse {
            session_id,
            model_info: Some(ModelInfo::from_model(&self.model)),
        }))
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
        // tonic runtime stays responsive. The session lock is held only long
        // enough to look up the evaluator and bump `last_used`; concurrent
        // predicts on different sessions can run truly in parallel.
        let encrypted_score = tokio::task::spawn_blocking(move || {
            let mut guard = sessions.blocking_lock();
            let session = guard.get_mut(&session_id).ok_or(())?;
            session.last_used = Instant::now();
            Ok::<_, ()>(session.evaluator.predict(&model, &features))
        })
        .await
        .map_err(|e| Status::internal(format!("predict task panicked: {e}")))?
        .map_err(|()| Status::not_found(format!("session {} not found", req.session_id)))?;

        let score_bytes = serialize_score(&encrypted_score)
            .map_err(|e| Status::internal(format!("failed to serialize score: {e}")))?;

        println!("[predict] Session completed");

        // The proto carries `repeated bytes encrypted_scores` to leave room for
        // future multi:softmax support without a wire break. Binary and
        // regression objectives always emit a single-element vector.
        Ok(Response::new(PredictResponse {
            encrypted_scores: vec![score_bytes],
        }))
    }
}

/// Periodically drop sessions whose `last_used` is older than `idle_timeout`.
async fn run_session_sweeper(sessions: SessionMap, idle_timeout: Duration) {
    let mut ticker = tokio::time::interval(SESSION_SWEEP_INTERVAL);
    loop {
        ticker.tick().await;
        let now = Instant::now();
        let mut guard = sessions.lock().await;
        let before = guard.len();
        guard.retain(|_, s| now.duration_since(s.last_used) < idle_timeout);
        let evicted = before - guard.len();
        if evicted > 0 {
            println!("[sweep] Evicted {evicted} idle session(s)");
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut model_path = "tests/fixtures/trained_binary.ubj".to_string();
    let mut port: u16 = 9999;
    let mut idle_timeout = Duration::from_secs(SESSION_IDLE_TIMEOUT_SECS);

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
            "--idle-timeout-secs" => {
                let secs: u64 = args
                    .next()
                    .expect("--idle-timeout-secs requires a number")
                    .parse()
                    .expect("invalid idle timeout");
                idle_timeout = Duration::from_secs(secs);
            }
            other => eprintln!("unknown argument: {other}"),
        }
    }

    println!("Loading model from {model_path}…");
    let model = WeirwoodTree::from_file(&model_path)?;
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

    let sessions: SessionMap = Arc::new(Mutex::new(HashMap::new()));
    tokio::spawn(run_session_sweeper(Arc::clone(&sessions), idle_timeout));

    let svc = WeirwoodInference {
        model: Arc::new(model),
        sessions,
    };

    let addr: SocketAddr = format!("127.0.0.1:{port}").parse()?;
    println!(
        "Inference server listening on {addr} (session idle timeout {} s)",
        idle_timeout.as_secs()
    );

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
