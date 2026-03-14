"""XGBoost Python plaintext inference benchmark.

Loads the same model and runs inference on the same feature vector as the
Rust benchmark so the numbers are directly comparable.

Called by run_benchmark.sh, or run standalone:
    python3 benchmarks/bench_python.py
    python3 benchmarks/bench_python.py tests/fixtures/trained_binary.json
"""

import time

import numpy as np
import xgboost as xgb

WARMUP = 1000
ITERATIONS = 100000
MODEL_PATH = "tests/fixtures/trained_binary.ubj"

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

features = np.array([[0.7, 0.3]], dtype=np.float32)
d_matrix = xgb.DMatrix(features)

# Warm up
for _ in range(WARMUP):
    booster.predict(d_matrix)

start = time.perf_counter()
for _ in range(ITERATIONS):
    booster.predict(d_matrix)
elapsed = time.perf_counter() - start

per_ns = (elapsed / ITERATIONS) * 1e9
throughput = ITERATIONS / elapsed

print(f"XGBoost Python plaintext inference ({MODEL_PATH})")
print(f"  iterations : {ITERATIONS}")
print(f"  total      : {elapsed * 1000:.3f} ms")
print(f"  per call   : {per_ns:.1f} ns")
print(f"  throughput : {throughput:.0f} inferences/sec")
