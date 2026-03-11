# weirwood

Privacy-preserving XGBoost inference via Fully Homomorphic Encryption, written in Rust.

Load a trained XGBoost model, encrypt a feature vector on the client, and evaluate the entire boosted tree ensemble on ciphertext. The server computes the prediction without ever seeing the input data.

**Status:** early development. Model loading and plaintext inference work today. The FHE evaluator is scaffolded behind the `tfhe-backend` feature flag and is the active area of development.

## How it works

XGBoost builds an ensemble of regression trees. At inference time, each tree routes the input from root to leaf by evaluating comparisons of the form `feature[i] <= threshold`. The prediction is the sum of leaf values across all trees, passed through an activation (sigmoid for classification, identity for regression).

Under FHE, the client encrypts its feature vector before sending it to the server. The server evaluates the full ensemble on ciphertext using TFHE's programmable bootstrapping — each split comparison is computed as an exact lookup table evaluation, no approximation required. The encrypted result is sent back and decrypted by the client. The server learns nothing.

The comparison-heavy tree traversal uses TFHE (via [tfhe-rs](https://github.com/zama-ai/tfhe-rs)); the final activation functions (sigmoid, softmax) are handled in CKKS where approximate real arithmetic is the right tool.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
weirwood = "0.1"

# For encrypted inference:
weirwood = { version = "0.1", features = ["tfhe-backend"] }
```

### Plaintext inference

Useful for verifying model loading and as a correctness reference.

```rust
use weirwood::{model::Ensemble, eval::{Evaluator, PlaintextEvaluator}};

fn main() -> Result<(), weirwood::Error> {
    let ensemble = Ensemble::from_json_file("model.json")?;

    let features = vec![1.0_f32, 0.5, 3.2, 0.1];
    let score = PlaintextEvaluator.predict_proba(&ensemble, &features);
    println!("prediction: {score:.4}");

    Ok(())
}
```

Save the model from Python with:

```python
booster.save_model("model.json")
```

### Encrypted inference (in progress)

```rust
#[cfg(feature = "tfhe-backend")]
{
    use weirwood::fhe::{FheContext, FheEvaluator};

    let ctx = FheContext::generate()?;
    let evaluator = FheEvaluator::new(ctx);
    // encrypted predict coming in a future release
}
```

## Project layout

```
src/
  lib.rs       public API and re-exports
  error.rs     WeirwoodError enum
  model.rs     XGBoost IR types (Ensemble, Tree, Node) + JSON loader
  eval.rs      Evaluator trait + PlaintextEvaluator
  fhe.rs       FheContext + FheEvaluator stub (tfhe-backend feature)
```

## Supported model formats

| Format | Status |
|--------|--------|
| XGBoost JSON (`.json`) | Supported |
| Universal Binary JSON (`.ubj`) | Planned |

## Supported objectives

| Objective | Plaintext | FHE |
|-----------|-----------|-----|
| `binary:logistic` | Yes | Planned |
| `reg:squarederror` | Yes | Planned |
| `multi:softmax` | Partial | Planned |

## Building

```sh
# library only
cargo build

# with FHE backend (pulls in tfhe-rs — expect a long compile)
cargo build --features tfhe-backend

# run tests
cargo test
```

## Performance notes

A typical XGBoost model with 100 trees at depth 5 requires roughly 31,000 bootstrapping operations. On CPU with `tfhe-rs`, each TFHE comparison takes about 10–20 ms, putting naive single-threaded inference around 5 minutes. `weirwood` parallelizes across nodes at the same tree depth using Rayon. GPU acceleration (targeting ~1 ms per comparison via `tfhe-rs`'s CUDA backend) is the primary optimization target for v0.2.

## License

Licensed under the [MIT License](LICENSE-MIT).
