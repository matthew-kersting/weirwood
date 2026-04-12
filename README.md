# weirwood

Privacy-preserving XGBoost inference via Fully Homomorphic Encryption, written in Rust.

Load a trained XGBoost model, encrypt a feature vector on the client, and evaluate the entire boosted tree ensemble on ciphertext. The server computes the prediction without ever seeing the input data.

**Status:** Model loading, plaintext inference, and FHE inference are all working. The FHE evaluator supports multi-tree ensembles of arbitrary depth with Rayon tree-level parallelism, validated on a 100-tree `binary:logistic` model with 525 internal nodes (~64 s per inference on CPU, avg 10 runs). Results match plaintext within fixed-point rounding error (`N × 0.5/SCALE` accumulated over N trees; ±0.05 worst-case for 100 trees with `SCALE=1000`, observed ≈ 0.0166 on the benchmark fixture). Sigmoid and softmax activations are applied client-side on the decrypted raw score.

## How it works

XGBoost builds an ensemble of regression trees. At inference time, each tree routes the input from root to leaf by evaluating comparisons of the form `feature[i] <= threshold`. The prediction is the sum of leaf values across all trees, passed through an activation (sigmoid for classification, identity for regression).

Under FHE, the client encrypts its feature vector before sending it to the server. The server evaluates the full ensemble on ciphertext using TFHE's programmable bootstrapping — each split comparison is computed as an exact lookup table evaluation, no approximation required. The encrypted result is sent back and decrypted by the client. The server learns nothing.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
weirwood = "0.3"
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
// FheEvaluator::new installs the server key on its Rayon worker threads.
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
  lib.rs            public API and re-exports
  error.rs          Error enum
  model.rs          XGBoost IR types (WeirwoodTree, Tree, Node) + JSON/UBJ loader
  ubj.rs            Universal Binary JSON parser
  eval/
    mod.rs          Evaluator trait + PlaintextEvaluator
    fhe/
      mod.rs        re-exports + unit tests
      client.rs     ClientContext — key generation, encrypt, decrypt; EncryptedInput; SCALE
      server.rs     ServerContext — server key only, set_active
      evaluator.rs  FheEvaluator — encrypted tree evaluation

examples/
  plaintext_inference.rs    end-to-end plaintext demo
  fhe_stump_inference.rs    end-to-end FHE demo on single stump
  fhe_full_inference.rs     end-to-end FHE demo on full ensemble (client-side activation)
  bench_plaintext.rs        plaintext throughput benchmark
  bench_fhe_stump.rs        FHE latency benchmark (stump)
  bench_fhe_full.rs         FHE latency benchmark (100-tree ensemble, 525 PBS ops)

tests/
  integration.rs            end-to-end plaintext + FHE correctness tests
  fixtures/                 trained_binary.{json,ubj}, stump_regression.json, two_trees_binary.json

benchmarks/
  train_model.py            train Breast Cancer Wisconsin XGBoost model; print test vectors
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

Inference benchmarks on the committed `trained_binary.ubj` fixture (Breast Cancer
Wisconsin, 100 trees, max_depth=8, 525 internal nodes, 30 features, `binary:logistic`,
StandardScaler-normalized).  Run `./benchmarks/run_benchmark.sh` to regenerate on your machine.

<!-- BENCHMARK_TABLE_START -->
_Last run: 2026-04-12 · model: `tests/fixtures/trained_binary.ubj` · plaintext: 100,000 iterations · FHE: avg 10 runs_

| Backend                        | Per call        | Throughput (inf/s) | Notes                              |
|--------------------------------|-----------------|--------------------|------------------------------------|
| weirwood (Rust, plaintext)     |     388.0 ns    |            2577011 |                                    |
| XGBoost (Python, plaintext)    |  103862.0 ns    |               9628 |                                    |
| weirwood (Rust, **FHE**)       |   3.9 min       |             0.0042 | avg 10 runs, 525 PBS ops        |

FHE phase breakdown: keygen 843 ms · encrypt 47.583 ms · inference 236.98 s (avg 10) · decrypt 0.031 ms · |Δ plaintext| = 0.0068
<!-- BENCHMARK_TABLE_END -->

## FHE Stump Benchmark

End-to-end FHE inference on the single decision stump (`stump_regression.json`,
depth 1, 1 tree, 1 PBS op per inference).  FHE latency is the average of 10
runs (~1.1 s each); plaintext uses 10,000 iterations for a stable per-call
figure.

Run `./benchmarks/run_benchmark_stump.sh` to regenerate on your machine
(expect ~30 s total).

<!-- FHE_STUMP_TABLE_START -->
_Last run: 2026-04-12 · model: `tests/fixtures/stump_regression.json` · stump (depth 1, 1 tree)_

> **Note:** FHE latency is the average of 10 bootstrapping runs;
> plaintext throughput uses 10,000 iterations.
> Key generation and encryption are one-time client costs.

| Backend                        | Per call          | Throughput (inf/s) | Notes                          |
|--------------------------------|-------------------|--------------------|--------------------------------|
| weirwood (Rust, plaintext)     |      2.5 ns      |          396982930 |                                |
| XGBoost (Python, plaintext)    |  68684.2 ns      |              14559 |                                |
| weirwood (Rust, **FHE**)       |      610 ms      |               1.64 | avg 10 runs, 1 PBS op each     |

FHE phase breakdown: keygen 841 ms · encrypt 1.648 ms · inference 0.61 s (avg 10) · decrypt 0.030 ms · |Δ plaintext| = 0.0000
<!-- FHE_STUMP_TABLE_END -->

## Performance notes

Each tree node comparison requires one TFHE programmable-bootstrapping operation on `FheInt32` encrypted inputs. On CPU with `tfhe-rs`, each PBS call takes ~0.6 s (measured on the stump). The 100-tree Breast Cancer ensemble (525 PBS ops) runs with Rayon tree-level parallelism; see the benchmark table for current timings. All nodes are visited obliviously regardless of the actual path taken.

`FheEvaluator` uses a dedicated Rayon thread pool so trees are evaluated in parallel; the speedup scales with available cores but is limited by memory-bandwidth saturation from NTT polynomial operations.  The primary remaining optimization target:
- **GPU acceleration** — `tfhe-rs`'s CUDA backend targets ~1 ms per PBS op, which would reduce the 525-node model to under 1 s.

## License

Licensed under the [MIT License](LICENSE-MIT).
