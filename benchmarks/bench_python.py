"""XGBoost Python plaintext inference benchmark.

Loads the same model and runs inference on the same feature vector as the
Rust benchmark so the numbers are directly comparable.

Called by run_benchmark.sh, or run standalone:
    python3 benchmarks/bench_python.py
    python3 benchmarks/bench_python.py tests/fixtures/trained_binary.json
"""

import sys
import time

import numpy as np
import xgboost as xgb

WARMUP = 1_000
ITERATIONS = 100_000
DEFAULT_MODEL = "tests/fixtures/trained_binary.ubj"

model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

booster = xgb.Booster()
booster.load_model(model_path)

features = np.array([[0.7, 0.3]], dtype=np.float32)
dmat = xgb.DMatrix(features)

# Warm up
for _ in range(WARMUP):
    booster.predict(dmat)

start = time.perf_counter()
for _ in range(ITERATIONS):
    booster.predict(dmat)
elapsed = time.perf_counter() - start

per_ns = (elapsed / ITERATIONS) * 1e9
throughput = ITERATIONS / elapsed

print(f"XGBoost Python plaintext inference ({model_path})")
print(f"  iterations : {ITERATIONS}")
print(f"  total      : {elapsed * 1000:.3f} ms")
print(f"  per call   : {per_ns:.1f} ns")
print(f"  throughput : {throughput:.0f} inferences/sec")
