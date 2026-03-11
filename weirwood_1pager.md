# `weirwood`
### Privacy-Preserving XGBoost Inference via Fully Homomorphic Encryption in Rust

---

## Overview

`weirwood` is a Rust crate for running XGBoost inference directly on encrypted data using Fully Homomorphic Encryption (FHE). The client encrypts their feature vector before sending it to the server; the server evaluates the entire boosted tree ensemble on the ciphertext and returns an encrypted result. Neither side learns anything about the other's data — the server never sees the input, and the client never sees the model weights.

The motivation is straightforward: gradient boosted tree models are widely deployed for sensitive prediction tasks (medical risk scoring, fraud detection, credit decisions), but current deployment patterns require either sending plaintext features to a server or shipping the model to the client. FHE eliminates this tradeoff. The challenge is making it fast enough to be practical.

XGBoost is a better FHE target than neural networks for a few reasons. Tree depth is a fixed hyperparameter — usually between 3 and 8 — which directly bounds the multiplicative depth of the FHE circuit. Each internal node tests exactly one feature against a threshold, so the circuit structure is sparse and predictable. And the final prediction is just a sum of leaf scores across all trees, which is cheap in any FHE scheme. Neural networks by contrast require deep compositions of non-linear activations, which are expensive to approximate homomorphically.

---

## Background

### XGBoost Inference

XGBoost [1] builds an ensemble of *K* regression trees. At inference time, the raw prediction for input **x** ∈ ℝᵈ is:

```
F(x) = Σ_{k=1}^{K} f_k(x)
```

Each tree `f_k` routes **x** down from root to leaf by evaluating splits of the form `xᵢ ≤ t`. A leaf's contribution equals its weight `v_l` times the product of indicator functions for every split on the path to that leaf:

```
f_k(x) = Σ_{l ∈ Leaves(k)} v_l · Π_{(n,b) ∈ path(l)} 𝟙[g_{n,b}(x)]
```

For binary classification, `F(x)` is passed through sigmoid; for regression it's used directly. The sigmoid and softmax activations are the hard part for FHE since they're non-polynomial.

### Fully Homomorphic Encryption

FHE [2] is a cryptographic primitive that allows arbitrary computation over encrypted data. The security of all modern FHE schemes rests on the hardness of Ring Learning With Errors (RLWE): given a polynomial ring `Rq = Zq[X] / (Xⁿ + 1)` and samples `(a, b = a·s + e)` where `e` is small, it is hard to distinguish these from random pairs in `Rq²`. This problem is believed hard for quantum computers, which is relevant given the sensitivity of the target applications.

The fundamental cost of homomorphic evaluation is noise growth — each multiplication increases the error term, and once it exceeds a decryption threshold, the ciphertext is lost. Bootstrapping [2] resets this noise by homomorphically evaluating the decryption circuit, but it's expensive. Practical schemes are designed to minimize how often bootstrapping is required.

`weirwood` uses two schemes selected for different parts of the computation:

- **TFHE** [3] for tree split comparisons (`xᵢ ≤ t`). TFHE operates over the real torus `T = ℝ/ℤ` and supports *programmable bootstrapping* — the bootstrapping step that refreshes noise can simultaneously evaluate an arbitrary lookup table. This means comparisons are exact (no approximation error) at the cost of one bootstrapping call per node, roughly 10–20 ms on CPU.

- **CKKS** [4] for the sigmoid and softmax activations. CKKS is designed for approximate real arithmetic and supports SIMD packing of thousands of values into a single ciphertext polynomial via CRT slot encoding, which makes it efficient for the activation layer once the tree scores are accumulated.

The bootstrapping technique underlying TFHE traces to FHEW [5], which first demonstrated sub-second bootstrapping via *blind rotation* — a homomorphic algorithm that rotates a test polynomial by a secret-key-encoded shift, evaluating a lookup table in `O(n log n)`. TFHE refined this into the programmable bootstrapping used here.

---

## Proposed Approach

The implementation proceeds in three phases:

**Phase 1 — Model loading and IR.** Parse a trained XGBoost model from its Universal Binary JSON (UBJ) serialization format [7] into an internal intermediate representation. UBJ is XGBoost's default binary format since v1.6; it encodes the tree arrays (`split_indices`, `split_conditions`, `base_weights`, child pointers) in a compact binary layout. The goal is zero-copy extraction of the numerically significant arrays — no JSON round-trip, no per-node heap allocation. A `ModelLoader` struct handles this, outputting a flat IR that the evaluator can consume without knowing anything about FHE.

**Phase 2 — FHE circuit evaluation.** Translate each tree in the IR into an FHE circuit. Each internal node becomes a TFHE LUT-bootstrapped comparison using the technique from Boura et al. [6], which encodes the Heaviside step function directly into the bootstrapping key. Leaf values are accumulated homomorphically. Because nodes at the same depth level of different trees are independent, these comparisons can be parallelized — the `tfhe-rs` crate exposes thread-safe bootstrapping that `weirwood::eval` exploits with Rayon. After all tree scores are summed, the sigmoid or softmax activation is computed in CKKS over the accumulated plaintext-equivalent scores.

**Phase 3 — Transport and API.** Define wire types for encrypted inference requests and responses, serializable over gRPC or REST. The client generates keys, encrypts features, sends the ciphertext, and receives an encrypted prediction it can decrypt locally. The server holds only the evaluation key (no decryption capability).

The main performance question is whether parallelism across tree nodes brings wall-clock time into a range that's practical for batch inference. For a typical ensemble (K=100 trees, depth 5), the naive estimate is ~31,000 bootstrapping calls at ~10 ms each — about 5 minutes single-threaded. With full parallelism across nodes at the same depth level, this drops significantly, and TFHE-rs GPU offload (targeting ~1 ms per comparison) is the v0.2 performance target.

A secondary research question is the two-scheme handoff: accumulating TFHE ciphertext outputs into CKKS for the activation layer requires a scheme-switching step that adds both implementation complexity and latency. An alternative is approximating sigmoid via a degree-7 or degree-15 Chebyshev polynomial directly in TFHE using multiple bootstrapping calls. Whether this is faster or slower than scheme-switching is an open question this project will benchmark.

---

## References

**[1] Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System."**
*Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

Defines the regularized boosted tree objective and the inference algorithm that `weirwood` implements under encryption. The leaf path product formulation in Section 2 is the direct source of the circuit structure.

---

**[2] Gentry, C. (2009). "Fully Homomorphic Encryption Using Ideal Lattices."**
*Proceedings of the 41st Annual ACM Symposium on Theory of Computing (STOC).* DOI: [10.1145/1536414.1536440](https://doi.org/10.1145/1536414.1536440)

The original FHE construction and the introduction of bootstrapping. Every modern scheme is a descendant of this blueprint, including TFHE and CKKS.

---

**[3] Chillotti, I., Gama, N., Georgieva, M., and Izabachène, M. (2020). "TFHE: Fast Fully Homomorphic Encryption over the Torus."**
*Journal of Cryptology, 33, 34–91.* DOI: [10.1007/s00145-019-09319-x](https://doi.org/10.1007/s00145-019-09319-x)

Introduces programmable bootstrapping over the real torus. The key result for this project is that the bootstrapping step can evaluate any lookup table, making exact comparison `xᵢ ≤ t` possible without polynomial approximation. The `tfhe-rs` library is the production Rust implementation of this scheme.

---

**[4] Cheon, J.-H., Kim, A., Kim, M., and Song, Y. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers."**
*Advances in Cryptology — ASIACRYPT 2017.* DOI: [10.1007/978-3-319-70694-8_15](https://doi.org/10.1007/978-3-319-70694-8_15)

Introduces CKKS and its SIMD slot packing via CRT decomposition. Used in `weirwood` for evaluating sigmoid and softmax after the tree traversal phase produces accumulated leaf scores.

---

**[5] Ducas, L. and Micciancio, D. (2015). "FHEW: Bootstrapping Homomorphic Encryption in Less Than a Second."**
*Advances in Cryptology — EUROCRYPT 2015.* DOI: [10.1007/978-3-662-46800-5_24](https://doi.org/10.1007/978-3-662-46800-5_24)

Introduced blind rotation and sub-second bootstrapping. TFHE's programmable bootstrapping is a direct extension of this technique, so FHEW's asymptotic complexity bounds also bound `weirwood`'s per-node cost.

---

**[6] Boura, C., Gama, N., Georgieva, M., and Jetchev, D. (2021). "TFHE Deep Dive."**
*IACR ePrint Archive, Report 2021/315.* Available: [https://ia.cr/2021/315](https://ia.cr/2021/315)

Detailed technical exposition of the full TFHE bootstrapping pipeline, including how to encode arbitrary functions (including Heaviside) as lookup tables inside the bootstrapping key. Also provides concrete parameter recommendations for 128-bit security that inform `weirwood`'s default key generation.

---

**[7] XGBoost Contributors (2022). "UBJ — Universal Binary JSON Model Serialization."**
*XGBoost Documentation.* Available: [https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)

Documents the binary model format `weirwood` parses. UBJ's compact encoding of the numeric tree arrays (`split_conditions`, `base_weights`, etc.) is what makes zero-copy model loading feasible.
