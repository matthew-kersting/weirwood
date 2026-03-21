# weirwood

Privacy-preserving XGBoost inference via Fully Homomorphic Encryption, written in Rust.

Load a trained XGBoost model, encrypt a feature vector on the client, and evaluate the entire boosted tree ensemble on ciphertext. The server computes the prediction without ever seeing the input data.

**Status:** Model loading, plaintext inference, and FHE inference are all working. The FHE evaluator supports regression (`reg:squarederror`) and produces results matching plaintext within fixed-point rounding error (±0.01 with `SCALE=100`).

## How it works

XGBoost builds an ensemble of regression trees. At inference time, each tree routes the input from root to leaf by evaluating comparisons of the form `feature[i] <= threshold`. The prediction is the sum of leaf values across all trees, passed through an activation (sigmoid for classification, identity for regression).

Under FHE, the client encrypts its feature vector before sending it to the server. The server evaluates the full ensemble on ciphertext using TFHE's programmable bootstrapping — each split comparison is computed as an exact lookup table evaluation, no approximation required. The encrypted result is sent back and decrypted by the client. The server learns nothing.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
weirwood = "0.2"
```

### Plaintext inference

Useful for verifying model loading and as a correctness reference.

`predict_proba` runs inference and applies the appropriate activation for the
model's objective (sigmoid for `binary:logistic`, identity for
`reg:squarederror`). Use `predict` (requires importing the `Evaluator` trait)
if you want the raw pre-activation score instead.

```rust
use weirwood::{model::WeirwoodTree, eval::PlaintextEvaluator};

fn main() -> Result<(), weirwood::Error> {
    // Load from JSON (text) or UBJ (binary) — both produce the same WeirwoodTree.
    let weirwood_tree = WeirwoodTree::from_json_file("model.json")?;
    // or: let weirwood_tree = WeirwoodTree::from_ubj_file("model.ubj")?;

    let features = vec![1.0_f32, 0.5, 3.2, 0.1];

    // Returns probability for binary:logistic, raw score for regression.
    let score = PlaintextEvaluator.predict_proba(&weirwood_tree, &features);
    println!("prediction: {score:.4}");

    Ok(())
}
```

To get the raw pre-activation score:

```rust
use weirwood::{model::WeirwoodTree, eval::{Evaluator, PlaintextEvaluator}};

let raw_score = PlaintextEvaluator.predict(&weirwood_tree, &features);
```

Save the model from Python with:

```python
booster.save_model("model.json")   # JSON (text)
booster.save_model("model.ubj")    # UBJ (binary, smaller on disk)
```

### Encrypted inference

The library models the two-party protocol through distinct types:

- **`ClientContext`** — holds both keys; used for key generation, encryption, and decryption. Never leaves the client.
- **`ServerContext`** — holds only the server key; handed to the inference server. Contains no private key material.
- **`FheEvaluator`** — takes a `ServerContext`; the type system prevents it from holding or using a private key.

```rust
use weirwood::{
    eval::Evaluator as _,
    fhe::{ClientContext, FheEvaluator},
    model::WeirwoodTree,
};

// --- Client ---
let client = ClientContext::generate()?;        // generate keypair (~1–3 s)
let server_ctx = client.server_context();       // extract server key only

let model = WeirwoodTree::from_json_file("model.json")?;
let features = vec![1.0_f32, 0.5, 3.2, 0.1];
let ciphertext = client.encrypt(&features);

// --- "Send server_ctx and ciphertext to the inference server" ---

// --- Server ---
server_ctx.set_active();                        // install server key on thread
let evaluator = FheEvaluator::new(server_ctx);
let encrypted_score = evaluator.predict(&model, &ciphertext);

// --- "Send encrypted_score back to the client" ---

// --- Client ---
let score = client.decrypt_score(&encrypted_score);
println!("prediction: {score:.4}");
```

In a single-process deployment (as in the examples) both parties run in the same process — the `server_ctx` is passed locally instead of over a network.

## Project layout

```
src/
  lib.rs         public API and re-exports
  error.rs       Error enum
  model.rs       XGBoost IR types (WeirwoodTree, Tree, Node) + JSON/UBJ loader
  eval.rs        Evaluator trait + PlaintextEvaluator
  fhe/
    mod.rs        re-exports
    client.rs     ClientContext — key generation, encrypt, decrypt
    server.rs     ServerContext — server key only, set_active
    evaluator.rs  FheEvaluator — encrypted tree evaluation

examples/
  plaintext_inference.rs    end-to-end plaintext demo
  fhe_stump_inference.rs    end-to-end FHE demo (two-party flow)
  bench_plaintext.rs        plaintext throughput benchmark
  bench_fhe_stump.rs        FHE latency benchmark

benchmarks/
  run_benchmark.sh          plaintext benchmark + README update
  run_benchmark_stump.sh    FHE stump benchmark + README update
  bench_python.py           Python/XGBoost baseline (plaintext)
  bench_python_stump.py     Python/XGBoost stump baseline
```

## Supported model formats

| Format | Status |
|--------|--------|
| XGBoost JSON (`.json`) | Supported |
| Universal Binary JSON (`.ubj`) | Supported |

## Supported objectives

| Objective | Plaintext | FHE |
|-----------|-----------|-----|
| `reg:squarederror` | Yes | Yes |
| `binary:logistic` | Yes | Partial (raw score; sigmoid applied post-decrypt) |
| `multi:softmax` | Partial | Planned |

## Building

```sh
cargo build   # tfhe-rs is a required dependency — expect a longer first compile
cargo test
```

## Benchmarks

Plaintext inference throughput measured on the committed `trained_binary.ubj`
fixture (100 trees, depth 3, 2 features), 100,000 iterations each.
Run `./benchmarks/run_benchmark.sh` to regenerate on your machine.

<!-- BENCHMARK_TABLE_START -->
_Last run: 2026-03-18 · model: `tests/fixtures/trained_binary.ubj` · 100,000 iterations_

| Backend                    | Total (ms)   | Per call (ns) | Throughput (inf/sec) |
|----------------------------|-------------|---------------|---------------------|
| weirwood (Rust, plaintext) |       0.722 |           7.2 |           138416159 |
| XGBoost (Python)  |    7799.412 |       77994.1 |               12821 |
<!-- BENCHMARK_TABLE_END -->

## FHE Stump Benchmark

End-to-end FHE inference on the single decision stump (`stump_regression.json`,
depth 1, 1 tree, 1 feature).  This is the simplest XGBoost model supported by
weirwood's FHE evaluator.  Because bootstrapping is expensive, FHE latency is
measured as a single-call wall-clock time rather than a throughput figure;
plaintext backends use 10,000 iterations for a stable per-call number.

Run `./benchmarks/run_benchmark_stump.sh` to regenerate on your machine
(expect ~5–15 min of CPU time).

<!-- FHE_STUMP_TABLE_START -->
_Last run: 2026-03-18 · model: `tests/fixtures/stump_regression.json` · stump (depth 1, 1 tree)_

> **Note:** FHE latency is the average of 5 bootstrapping runs;
> plaintext throughput uses 10,000 iterations.
> Key generation and encryption are one-time client costs.

| Backend                        | Per call          | Throughput (inf/s) | Notes                          |
|--------------------------------|-------------------|--------------------|--------------------------------|
| weirwood (Rust, plaintext)     |      3.3 ns      |          301750151 |                                |
| XGBoost (Python, plaintext)    |  61879.2 ns      |              16161 |                                |
| weirwood (Rust, **FHE**)       |      520 ms      |               1.93 | avg 5 runs, 1 PBS op each      |

FHE phase breakdown: keygen 958 ms · encrypt 0.813 ms · inference 0.52 s (avg 5) · decrypt 0.030 ms · |Δ plaintext| = 0.0000
<!-- FHE_STUMP_TABLE_END -->

## Performance notes

A typical XGBoost model with 100 trees at depth 5 requires roughly 31,000 bootstrapping operations. On CPU with `tfhe-rs`, each TFHE comparison takes about 10–20 ms, putting naive single-threaded inference around 5 minutes. GPU acceleration (targeting ~1 ms per comparison via `tfhe-rs`'s CUDA backend) is the primary optimization target for v0.3.

## License

Licensed under the [MIT License](LICENSE-MIT).
