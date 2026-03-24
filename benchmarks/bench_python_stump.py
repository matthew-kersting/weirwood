"""XGBoost Python plaintext inference benchmark for the stump regression fixture.

Trains a depth-1 stump matching the weirwood fixture (feature[0] <= 1.5,
leaves ±0.5, base_score 1.0) and benchmarks inference on the same feature
vector used by bench_fhe_stump.rs.

The stump fixture (stump_regression.json) is weirwood's custom JSON format and
cannot be loaded directly by XGBoost's Booster.load_model.  Training here
ensures the Python benchmark exercises the same XGBoost inference code path.

Called by run_benchmark_stump.sh, or run standalone:
    python3 benchmarks/bench_python_stump.py
"""

import time

import numpy as np
import xgboost as xgb

WARMUP = 1000
ITERATIONS = 10_000

# Train a depth-1 stump: two samples straddle the threshold at 1.5.
# Labels are chosen so XGBoost learns leaves near ±0.5.
X_train = np.array([[0.0], [2.0]], dtype=np.float32)
y_train = np.array([0.5, 1.5], dtype=np.float32)

booster = xgb.train(
    {"max_depth": 1, "n_estimators": 1, "eta": 1.0, "base_score": 1.0,
     "objective": "reg:squarederror", "verbosity": 0},
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=1,
)

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

print(f"XGBoost Python plaintext stump inference (trained depth-1 stump)")
print(f"  iterations : {ITERATIONS}")
print(f"  total      : {elapsed * 1000:.3f} ms")
print(f"  per call   : {per_ns:.1f} ns")
print(f"  throughput : {throughput:.0f} inferences/sec")
print(f"BENCH_PY_NS={per_ns:.1f}")
print(f"BENCH_PY_THRU={throughput:.0f}")
