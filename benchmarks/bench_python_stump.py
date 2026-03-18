"""XGBoost Python plaintext inference benchmark for the stump regression fixture.

Runs inference with the same feature vector used by bench_fhe_stump.rs so the
numbers are directly comparable in the benchmark table.

Called by run_benchmark_stump.sh, or run standalone:
    python3 benchmarks/bench_python_stump.py
    python3 benchmarks/bench_python_stump.py tests/fixtures/stump_regression.json
"""

import sys
import time

import numpy as np
import xgboost as xgb

WARMUP = 1000
ITERATIONS = 10_000
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/stump_regression.json"

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

# Same feature vector used by the Rust FHE benchmark (left-branch probe).
features = np.array([[0.0]], dtype=np.float32)
d_matrix = xgb.DMatrix(features)

for _ in range(WARMUP):
    booster.predict(d_matrix)

start = time.perf_counter()
for _ in range(ITERATIONS):
    booster.predict(d_matrix)
elapsed = time.perf_counter() - start

per_ns = (elapsed / ITERATIONS) * 1e9
throughput = ITERATIONS / elapsed

print(f"XGBoost Python plaintext stump inference ({MODEL_PATH})")
print(f"  iterations : {ITERATIONS}")
print(f"  total      : {elapsed * 1000:.3f} ms")
print(f"  per call   : {per_ns:.1f} ns")
print(f"  throughput : {throughput:.0f} inferences/sec")
print(f"BENCH_PY_NS={per_ns:.1f}")
print(f"BENCH_PY_THRU={throughput:.0f}")
