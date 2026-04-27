//! Simple TCP-based inference server for privacy-preserving XGBoost inference.
//!
//! Demonstrates the transport serialization layer by accepting clients over TCP,
//! receiving encrypted inputs, and returning encrypted scores. Uses prost-encoded
//! messages (same format as gRPC protocol buffers) for a wire-compatible transport.
//!
//! Usage:
//!   cargo run --release --example server --features transport -- \
//!     --model tests/fixtures/trained_binary.ubj [--port 9999]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use prost::Message as _;
use uuid::Uuid;

use weirwood::{
    eval::Evaluator as _,
    fhe::FheEvaluator,
    model::WeirwoodTree,
    transport::{
        deserialize_feature, deserialize_server_context,
        rpc::{InitSessionRequest, InitSessionResponse, PredictRequest, PredictResponse},
        serialize_score,
    },
};

type SessionMap = Arc<Mutex<HashMap<String, FheEvaluator>>>;

fn handle_client(mut stream: TcpStream, model: Arc<WeirwoodTree>, sessions: SessionMap) {
    let peer_addr = stream.peer_addr().ok();
    println!("[connection] Client connected from {:?}", peer_addr);

    // Read message type (1 byte): 0 = InitSession, 1 = Predict
    let mut msg_type = [0u8; 1];
    if stream.read_exact(&mut msg_type).is_err() {
        eprintln!("[error] Failed to read message type");
        return;
    }

    match msg_type[0] {
        // InitSession request
        0 => {
            let mut len_bytes = [0u8; 4];
            if stream.read_exact(&mut len_bytes).is_err() {
                eprintln!("[error] Failed to read message length");
                return;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            let mut buf = vec![0u8; len];
            if stream.read_exact(&mut buf).is_err() {
                eprintln!("[error] Failed to read message");
                return;
            }

            match InitSessionRequest::decode(buf.as_slice()) {
                Ok(req) => {
                    let session_id = Uuid::new_v4().to_string();

                    match weirwood::transport::deserialize_server_context(&req.server_key) {
                        Ok(server_ctx) => {
                            println!(
                                "[init] Creating evaluator (server key {} MB)",
                                req.server_key.len() / 1_000_000
                            );
                            let evaluator = FheEvaluator::new(server_ctx);

                            let mut sess = sessions.lock().unwrap();
                            sess.insert(session_id.clone(), evaluator);
                            drop(sess);

                            let resp = InitSessionResponse {
                                session_id: session_id.clone(),
                            };

                            let mut buf = Vec::new();
                            if let Err(e) = resp.encode(&mut buf) {
                                eprintln!("[error] Failed to encode response: {}", e);
                                return;
                            }

                            let _ = stream.write_all(&(buf.len() as u32).to_le_bytes());
                            let _ = stream.write_all(&buf);

                            println!("[init] Session {} created", session_id);
                        }
                        Err(e) => {
                            eprintln!("[error] Failed to deserialize server key: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[error] Failed to decode InitSessionRequest: {}", e);
                }
            }
        }

        // Predict request
        1 => {
            let mut len_bytes = [0u8; 4];
            if stream.read_exact(&mut len_bytes).is_err() {
                eprintln!("[error] Failed to read message length");
                return;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            let mut buf = vec![0u8; len];
            if stream.read_exact(&mut buf).is_err() {
                eprintln!("[error] Failed to read message");
                return;
            }

            match PredictRequest::decode(buf.as_slice()) {
                Ok(req) => {
                    // Deserialize features first (outside the lock)
                    let mut features = Vec::new();
                    for feature_bytes in req.features {
                        match deserialize_feature(&feature_bytes) {
                            Ok(feature) => features.push(feature),
                            Err(e) => {
                                eprintln!("[error] Failed to deserialize feature: {}", e);
                                return;
                            }
                        }
                    }

                    // Get evaluator from sessions
                    let encrypted_score = {
                        let mut sess = sessions.lock().unwrap();
                        match sess.get_mut(&req.session_id) {
                            Some(evaluator) => evaluator.predict(&model, &features),
                            None => {
                                eprintln!("[error] Session {} not found", req.session_id);
                                return;
                            }
                        }
                    };

                    // Serialize result
                    match serialize_score(&encrypted_score) {
                        Ok(score_bytes) => {
                            let resp = PredictResponse {
                                encrypted_score: score_bytes.into(),
                            };

                            let mut buf = Vec::new();
                            if let Err(e) = resp.encode(&mut buf) {
                                eprintln!("[error] Failed to encode response: {}", e);
                                return;
                            }

                            let _ = stream.write_all(&(buf.len() as u32).to_le_bytes());
                            let _ = stream.write_all(&buf);

                            println!(
                                "[predict] Session {} completed ({} features)",
                                req.session_id,
                                features.len()
                            );
                        }
                        Err(e) => {
                            eprintln!("[error] Failed to serialize score: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[error] Failed to decode PredictRequest: {}", e);
                }
            }
        }

        _ => {
            eprintln!("[error] Unknown message type: {}", msg_type[0]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut model_path = "tests/fixtures/trained_binary.ubj".to_string();
    let mut port = 9999u16;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_path = args.next().expect("--model requires a path");
            }
            "--port" => {
                port = args
                    .next()
                    .expect("--port requires a number")
                    .parse()
                    .expect("invalid port");
            }
            _ => eprintln!("unknown argument: {}", arg),
        }
    }

    println!("Loading model from {}…", model_path);
    let model = if model_path.ends_with(".ubj") {
        WeirwoodTree::from_ubj_file(&model_path)?
    } else {
        WeirwoodTree::from_json_file(&model_path)?
    };

    println!(
        "Model: {} trees, {} features",
        model.trees.len(),
        model.num_features
    );

    let model = Arc::new(model);
    let sessions: SessionMap = Arc::new(Mutex::new(HashMap::new()));

    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))?;
    println!("Server listening on 127.0.0.1:{}", port);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let model = model.clone();
                let sessions = sessions.clone();
                thread::spawn(move || handle_client(stream, model, sessions));
            }
            Err(e) => eprintln!("[error] Failed to accept connection: {}", e),
        }
    }

    Ok(())
}
