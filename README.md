# weirwood

Privacy-preserving XGBoost inference via Fully Homomorphic Encryption, written in Rust.

Load a trained XGBoost model, encrypt a feature vector on the client, and evaluate the entire boosted tree ensemble on ciphertext. The server computes the prediction without ever seeing the input data.

**Status:** Model loading, plaintext inference, and FHE inference are all working. The FHE evaluator supports multi-tree ensembles of arbitrary depth, validated on a 100-tree `binary:logistic` model with 177 internal nodes. Results match plaintext within fixed-point rounding error (`N × 0.5/SCALE` accumulated over N trees; ±0.50 worst-case for 100 trees with `SCALE=100`, observed ≈ 0.017 on the benchmark fixture). Sigmoid and softmax activations are applied client-side on the decrypted raw score.

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
// decrypt_score returns the raw pre-activation ensemble score.
// Apply sigmoid / identity client-side depending on the model objective.
let raw_score = client.decrypt_score(&encrypted_score);
println!("prediction: {raw_score:.4}"); // for regression (identity activation)
// for binary:logistic: let proba = 1.0 / (1.0 + (-raw_score).exp());
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
  fhe_stump_inference.rs    end-to-end FHE demo on single stump
  fhe_full_inference.rs     end-to-end FHE demo on full ensemble (client-side activation)
  bench_plaintext.rs        plaintext throughput benchmark
  bench_fhe_stump.rs        FHE latency benchmark (stump)
  bench_fhe_full.rs         FHE latency benchmark (100-tree ensemble, 177 PBS ops)

benchmarks/
  run_benchmark.sh          full benchmark (plaintext + FHE) + README update
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
| `binary:logistic` | Yes | Yes (sigmoid applied client-side post-decrypt) |
| `multi:softmax` | Partial | Planned |

## Building

```sh
cargo build   # tfhe-rs is a required dependency — expect a longer first compile
cargo test
```

## Benchmarks

Inference benchmarks on the committed `trained_binary.ubj` fixture (100 trees,
max_depth=8, 177 internal nodes, 2 features, `binary:logistic`).
Run `./benchmarks/run_benchmark.sh` to regenerate on your machine.

<!-- BENCHMARK_TABLE_START -->
_Last run: 2026-04-06 · model: `tests/fixtures/trained_binary.ubj` · plaintext: 100,000 iterations · FHE: avg 10 runs_

| Backend                        | Per call        | Throughput (inf/s) | Notes                              |
|--------------------------------|-----------------|--------------------|------------------------------------|
| weirwood (Rust, plaintext)     |     252.7 ns    |            3957400 |                                    |
| XGBoost (Python, plaintext)    |   79344.4 ns    |              12603 |                                    |
| weirwood (Rust, **FHE**)       |   1.5 min       |             0.0109 | avg 10 runs, 177 PBS ops        |

FHE phase breakdown: keygen 793 ms · encrypt 1.702 ms · inference 91.80 s (avg 10) · decrypt 0.029 ms · |Δ plaintext| = 0.0166
<!-- BENCHMARK_TABLE_END -->

## FHE Stump Benchmark

End-to-end FHE inference on the single decision stump (`stump_regression.json`,
depth 1, 1 tree, 1 PBS op per inference).  FHE latency is the average of 10
runs (~410 ms each); plaintext uses 10,000 iterations for a stable per-call
figure.

Run `./benchmarks/run_benchmark_stump.sh` to regenerate on your machine
(expect ~30 s total).

<!-- FHE_STUMP_TABLE_START -->
_Last run: 2026-04-06 · model: `tests/fixtures/stump_regression.json` · stump (depth 1, 1 tree)_

> **Note:** FHE latency is the average of 10 bootstrapping runs;
> plaintext throughput uses 10,000 iterations.
> Key generation and encryption are one-time client costs.

| Backend                        | Per call          | Throughput (inf/s) | Notes                          |
|--------------------------------|-------------------|--------------------|--------------------------------|
| weirwood (Rust, plaintext)     |      4.7 ns      |          213533770 |                                |
| XGBoost (Python, plaintext)    |  48327.5 ns      |              20692 |                                |
| weirwood (Rust, **FHE**)       |      410 ms      |               2.44 | avg 10 runs, 1 PBS op each     |

FHE phase breakdown: keygen 752 ms · encrypt 0.791 ms · inference 0.41 s (avg 10) · decrypt 0.020 ms · |Δ plaintext| = 0.0000
<!-- FHE_STUMP_TABLE_END -->

## Performance notes

Each tree node comparison requires one TFHE programmable-bootstrapping operation. On CPU with `tfhe-rs`, each PBS call takes ~410–520 ms single-threaded (measured: 410 ms on the stump, 92 s / 177 ops = 520 ms on the full model), so inference latency scales linearly with the number of internal nodes evaluated. All nodes are visited obliviously regardless of the actual path taken. The committed 100-tree fixture has 177 internal nodes, giving ~92 s per inference.

The two primary optimization targets for v0.3:
- **Rayon parallelization** — nodes across trees are independent; `tfhe-rs` exposes thread-safe bootstrapping. Parallelizing across trees would reduce latency proportionally to available cores.
- **GPU acceleration** — `tfhe-rs`'s CUDA backend targets ~1 ms per PBS op, which would reduce the 177-node model from ~90 s to under 1 s.

## License

Licensed under the [MIT License](LICENSE-MIT).
