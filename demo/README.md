# weirwood demos

Two minimal Cargo projects that use the pretrained Breast Cancer Wisconsin
model (`tests/fixtures/trained_binary.ubj`) and real samples from its held-out
test split.

Both demos depend on the local `weirwood` crate via path. Each FHE inference
takes ~4 minutes on CPU — always run in `--release`.

## `fhe_local/` — single-file in-process demo

Loads model → generates FHE keys → encrypts → evaluates → decrypts, all in
one process.

```sh
cd demo/fhe_local
cargo run --release
```

## `fhe_grpc/` — client + server over gRPC

Two binaries (`server` + `client`) sharing a `Cargo.toml`. The server holds
the model and reports its shape (number of features, objective) at session
setup; the client never loads the XGBoost file. The client encrypts features
and sends them over gRPC, and the server never sees plaintext data.

```sh
cd demo/fhe_grpc
# Terminal 1
cargo run --release --bin server
# Terminal 2 (after the server prints "Listening…")
cargo run --release --bin client
```
