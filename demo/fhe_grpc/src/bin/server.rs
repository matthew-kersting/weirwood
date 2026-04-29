//! Minimal FHE inference gRPC server.
//!
//! Loads the pretrained Breast Cancer Wisconsin model once, then serves
//! `InitSession` (upload a `ServerKey`, learn the model's shape) and
//! `Predict` (run encrypted inference) over gRPC. The model never leaves
//! the server.
//!
//! Run with:
//!   cargo run --release --bin server
//!
//! Then start the client in another terminal:
//!   cargo run --release --bin client

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tonic::transport::Server;
use tonic::{Request, Response, Status};

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

const MODEL_PATH: &str = "model.ubj";
const ADDR: &str = "127.0.0.1:9999";

struct InferenceServer {
    model: Arc<WeirwoodTree>,
    info: ModelInfo,
    sessions: Arc<Mutex<HashMap<u64, FheEvaluator>>>,
    next_session_id: AtomicU64,
}

#[tonic::async_trait]
impl InferenceService for InferenceServer {
    async fn init_session(
        &self,
        request: Request<InitSessionRequest>,
    ) -> Result<Response<InitSessionResponse>, Status> {
        let req = request.into_inner();
        let server_ctx = deserialize_server_context(&req.server_key)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let evaluator = FheEvaluator::try_new(&self.model, server_ctx)
            .map_err(|e| Status::failed_precondition(e.to_string()))?;

        let id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        self.sessions.lock().unwrap().insert(id, evaluator);

        println!(
            "[init] session ({} MB server key)",
            req.server_key.len() / 1_000_000
        );
        Ok(Response::new(InitSessionResponse {
            session_id: id.to_string(),
            model_info: Some(self.info.clone()),
        }))
    }

    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let req = request.into_inner();
        let id: u64 = req
            .session_id
            .parse()
            .map_err(|_| Status::invalid_argument("invalid session id"))?;

        let features: Vec<_> = req
            .features
            .iter()
            .map(|bytes| deserialize_feature(bytes))
            .collect::<Result<_, _>>()
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        println!(
            "[predict] session {id}, {} features (~4 min on CPU)…",
            features.len()
        );

        let mut sessions = self.sessions.lock().unwrap();
        let evaluator = sessions
            .get_mut(&id)
            .ok_or_else(|| Status::not_found("session not found"))?;
        let encrypted_score = evaluator.predict(&self.model, &features);

        let bytes =
            serialize_score(&encrypted_score).map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(PredictResponse {
            encrypted_scores: vec![bytes],
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from {MODEL_PATH}…");
    let model = WeirwoodTree::from_file(MODEL_PATH)?;
    println!(
        "  {} trees, {} features, objective {:?}",
        model.trees.len(),
        model.num_features,
        model.objective
    );

    let info = ModelInfo::from_model(&model);
    let svc = InferenceServer {
        model: Arc::new(model),
        info,
        sessions: Arc::new(Mutex::new(HashMap::new())),
        next_session_id: AtomicU64::new(0),
    };

    println!("Listening on {ADDR}");
    Server::builder()
        .add_service(
            InferenceServiceServer::new(svc)
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES),
        )
        .serve(ADDR.parse()?)
        .await?;
    Ok(())
}
