//! Simple TCP-based inference server for privacy-preserving XGBoost inference.
//!
//! Demonstrates the transport serialization layer by accepting clients over TCP,
//! receiving encrypted inputs, and returning encrypted scores. Uses prost-encoded
//! messages (Protocol Buffer wire format) framed with a 1-byte message type and a
//! 4-byte little-endian length prefix.
//!
//! Each TCP connection may carry multiple framed messages: typically one
//! `InitSession` followed by N `Predict` requests. The handler loops until the
//! peer closes the stream.
//!
//! Usage:
//!   cargo run --release --example server --features transport -- \
//!     --model tests/fixtures/trained_binary.ubj [--port 9999]

use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
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

/// Read one framed message: 4-byte little-endian length, then `len` bytes.
/// Returns `Ok(None)` on clean EOF before any byte of the frame is read.
fn read_frame(stream: &mut TcpStream) -> std::io::Result<Option<Vec<u8>>> {
    let mut len_bytes = [0u8; 4];
    match stream.read_exact(&mut len_bytes) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(Some(buf))
}

fn write_frame(stream: &mut TcpStream, buf: &[u8]) -> std::io::Result<()> {
    stream.write_all(&(buf.len() as u32).to_le_bytes())?;
    stream.write_all(buf)
}

fn handle_init(stream: &mut TcpStream, sessions: &SessionMap) -> std::io::Result<()> {
    let buf = match read_frame(stream)? {
        Some(b) => b,
        None => return Ok(()),
    };

    let req = match InitSessionRequest::decode(buf.as_slice()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[error] Failed to decode InitSessionRequest: {}", e);
            return Ok(());
        }
    };

    let server_ctx = match deserialize_server_context(&req.server_key) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("[error] Failed to deserialize server key: {}", e);
            return Ok(());
        }
    };

    println!(
        "[init] Creating evaluator (server key {} MB)",
        req.server_key.len() / 1_000_000
    );
    let evaluator = FheEvaluator::new(server_ctx);

    let session_id = Uuid::new_v4().to_string();
    sessions
        .lock()
        .unwrap()
        .insert(session_id.clone(), evaluator);

    let resp = InitSessionResponse {
        session_id: session_id.clone(),
    };
    let mut out = Vec::new();
    if let Err(e) = resp.encode(&mut out) {
        eprintln!("[error] Failed to encode response: {}", e);
        return Ok(());
    }
    write_frame(stream, &out)?;

    println!("[init] Session {} created", session_id);
    Ok(())
}

fn handle_predict(
    stream: &mut TcpStream,
    model: &WeirwoodTree,
    sessions: &SessionMap,
) -> std::io::Result<()> {
    let buf = match read_frame(stream)? {
        Some(b) => b,
        None => return Ok(()),
    };

    let req = match PredictRequest::decode(buf.as_slice()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[error] Failed to decode PredictRequest: {}", e);
            return Ok(());
        }
    };

    let mut features = Vec::with_capacity(req.features.len());
    for feature_bytes in &req.features {
        match deserialize_feature(feature_bytes) {
            Ok(f) => features.push(f),
            Err(e) => {
                eprintln!("[error] Failed to deserialize feature: {}", e);
                return Ok(());
            }
        }
    }

    let encrypted_score = {
        let mut sess = sessions.lock().unwrap();
        match sess.get_mut(&req.session_id) {
            Some(evaluator) => evaluator.predict(model, &features),
            None => {
                eprintln!("[error] Session {} not found", req.session_id);
                return Ok(());
            }
        }
    };

    let score_bytes = match serialize_score(&encrypted_score) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[error] Failed to serialize score: {}", e);
            return Ok(());
        }
    };

    let resp = PredictResponse {
        encrypted_score: score_bytes.into(),
    };
    let mut out = Vec::new();
    if let Err(e) = resp.encode(&mut out) {
        eprintln!("[error] Failed to encode response: {}", e);
        return Ok(());
    }
    write_frame(stream, &out)?;

    println!(
        "[predict] Session {} completed ({} features)",
        req.session_id,
        features.len()
    );
    Ok(())
}

fn handle_client(mut stream: TcpStream, model: Arc<WeirwoodTree>, sessions: SessionMap) {
    let peer_addr = stream.peer_addr().ok();
    println!("[connection] Client connected from {:?}", peer_addr);

    loop {
        let mut msg_type = [0u8; 1];
        match stream.read_exact(&mut msg_type) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                println!("[connection] Client {:?} disconnected", peer_addr);
                return;
            }
            Err(e) => {
                eprintln!("[error] Failed to read message type: {}", e);
                return;
            }
        }

        let result = match msg_type[0] {
            0 => handle_init(&mut stream, &sessions),
            1 => handle_predict(&mut stream, &model, &sessions),
            other => {
                eprintln!("[error] Unknown message type: {}", other);
                return;
            }
        };

        if let Err(e) = result {
            eprintln!("[error] Connection error: {}", e);
            return;
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
