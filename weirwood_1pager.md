# Privacy-Preserving XGBoost Inference via Fully Homomorphic Encryption in Rust

## Overview

`weirwood` is a Rust crate for running XGBoost inference directly on encrypted data using Fully Homomorphic Encryption (FHE).The client encrypts their feature vector before sending it to the server. The server then performs inference on the encrypted data and returns the result in an encrypted form, ensuring that the server doesn't learn anything about the data.

The motivation is straightforward: gradient boosted tree models are widely deployed for prediction tasks including sensitive areas such as malware detection. Generally, deployment patterns require either sending plaintext features to a server or shipping the model to the client. FHE eliminates this tradeoff and keeps data secure without overwHelming the client. The downside of FHE is that it tends to be very slow.

## Background

### XGBoost Inference

XGBoost [1] builds an ensemble of *K* regression trees. At inference time, the raw prediction for input **x** ∈ ℝᵈ is `F(x) = Σ_{k=1}^{K} f_k(x)`. Each tree `f_k` routes **x** down from root to leaf by evaluating splits of the form `xᵢ ≤ t`. A leaf's contribution equals its weight `v_l` times the product of indicator functions for every split on the path to that leaf: `f_k(x) = Σ_{l ∈ Leaves(k)} v_l · Π_{(n,b) ∈ path(l)} 𝟙[g_{n,b}(x)]`. For binary classification, `F(x)` is passed through sigmoid; for regression it's used directly. Sigmoid and softmax activations are one of the harder aspects to implement in FHE because PHE works better for polynomial functions.


### Fully Homomorphic Encryption

FHE [2] is a cryptographic primitive that allows arbitrary computation over encrypted data, it is the successor to HE. The security of all modern FHE schemes rests on the hardness of Ring Learning With Errors (RLWE): given a polynomial ring `Rq = Zq[X] / (Xⁿ + 1)` and samples `(a, b = a·s + e)` where `e` is small, it is hard to distinguish these from random pairs in `Rq²`. This problem is generally also considered to be hard for quantum computers, making FHE a quantum resistant function.

The fundamental cost of homomorphic evaluation is noise growth. Every multiplication operation increases the error term, and once it exceeds a decryption threshold, we lose the ciphertext. Bootstrapping [2] resets this noise by homomorphically evaluating the decryption circuit, but it is very computationally expensive. The goal of FHE schemes in practice is to minimize how frequently bootstrapping is required.

`weirwood` uses two schemes selected for different parts of the computation:

- **TFHE** [3] Fully Homomorphic Encryption Scheme over the Torus (TFHE) for tree split comparisons (`xᵢ ≤ t`). TFHE operates over the real torus `T = ℝ/ℤ` and supports *programmable bootstrapping*, meaning that the bootstrapping step that refreshes noise can simultaneously evaluate an arbitrary lookup table. This means comparisons are exact at the cost of one bootstrapping call per node, roughly 10–20 ms on CPU.
- **CKKS** [4] CKKS is named by the last initials of the 4 others of the paper. This is used for the sigmoid and softmax activations. CKKS is designed for approximate real arithmetic and supports single instruction, multiple data (SIMD) packing of thousands of values into a single ciphertext polynomial via chinese remainder theorem slot encoding, which makes it efficient for the activation layer once the tree scores are accumulated.

## Project Structure

The name references the weirwood trees of *A Song of Ice and Fire* which are an ancient species of trees revered by those who follow the Old Gods, the religion of the First Men. An ensemble of boosted trees felt like as good a fit as any and the crate name had not yet been taken.

The repository is hosted at [github.com/matthew-kersting/weirwood](https://github.com/matthew-kersting/weirwood) and published to [crates.io/crates/weirwood](https://crates.io/crates/weirwood) under an MIT license. CI is handled by GitHub Actions: pull requests trigger build and test and version tags additionally publish to crates.io. The project is structured as a library with usage examples in `examples/` and integration tests with fixtures in `tests/`.

## Implementation

The implementation will have three major phases. Phase 1 is mostly complete and phase 2 is partially started.

**Phase 1: Model loading and IR.** Parse a trained XGBoost model from its Universal Binary JSON (UBJ) serialization format [7] into an internal intermediate representation in rust data structures. UBJ is XGBoost's default binary format since v1.6; it encodes the tree arrays (`split_indices`, `split_conditions`, `base_weights`, child pointers) in a compact binary layout. The `WeirwoodTree` type holds this intermediate representation, a `PlaintextEvaluator` runs standard f32 inference over it as a baseline reference.

**Phase 2: FHE circuit evaluation.** Translate each tree in the IR into an FHE circuit. Key generation, fixed-point feature encryption, and decryption of the result are complete. The active work is circuit translation: each internal node becomes a TFHE look-up-table-bootstrapped comparison using the technique from Boura et al. [6], which encodes the Heaviside step function directly into the bootstrapping key. Leaf values are accumulated homomorphically. Because nodes at the same depth of different trees are independent, these comparisons can be parallelized — the `tfhe-rs` crate exposes thread-safe bootstrapping that `weirwood` will exploit with Rayon. After all tree scores are summed, the sigmoid or softmax activation is computed in CKKS over the accumulated scores.

**Phase 3: Transport and API.** Define wire types for encrypted inference requests and responses, serializable over gRPC or REST. The client generates keys, encrypts features, sends the ciphertext, and receives an encrypted prediction it can decrypt locally. The server holds only the evaluation key (no decryption capability).

## References

- **[1] Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."**
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)


- **[2] Gentry, C. (2009). "Fully Homomorphic Encryption Using Ideal Lattices."**
*Proceedings of the 41st Annual ACM Symposium on Theory of Computing (STOC).* DOI: [10.1145/1536414.1536440](https://doi.org/10.1145/1536414.1536440)


- **[3] Chillotti, I., Gama, N., Georgieva, M., and Izabachène, M. (2020). "TFHE: Fast Fully Homomorphic Encryption over the Torus."**
*Journal of Cryptology, 33, 34–91.* DOI: [10.1007/s00145-019-09319-x](https://doi.org/10.1007/s00145-019-09319-x)


- **[4] Cheon, J.-H., Kim, A., Kim, M., and Song, Y. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers."**
*Advances in Cryptology — ASIACRYPT 2017.* DOI: [10.1007/978-3-319-70694-8_15](https://doi.org/10.1007/978-3-319-70694-8_15)


- **[5] Ducas, L. and Micciancio, D. (2015). "FHEW: Bootstrapping Homomorphic Encryption in Less Than a Second."**
*Advances in Cryptology — EUROCRYPT 2015.* DOI: [10.1007/978-3-662-46800-5_24](https://doi.org/10.1007/978-3-662-46800-5_24)

- **[6] Boura, C., Gama, N., Georgieva, M., and Jetchev, D. (2021). "TFHE Deep Dive."**
*IACR ePrint Archive, Report 2021/315.* Available: [https://ia.cr/2021/315](https://ia.cr/2021/315)

- **[7] XGBoost Contributors (2022). "UBJ — Universal Binary JSON Model Serialization."**
*XGBoost Documentation.* Available: [https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)# Privacy-Preserving XGBoost Inference via Fully Homomorphic Encryption in Rust