#!/usr/bin/env bash
# run_benchmark.sh — benchmark weirwood (Rust plaintext + FHE) vs XGBoost
# (Python plaintext) on trained_binary.ubj, then update the Benchmarks table
# in README.md.
#
# Usage:
#   ./benchmarks/run_benchmark.sh
#   ./benchmarks/run_benchmark.sh tests/fixtures/trained_binary.ubj
#
# Requirements:
#   - Rust toolchain (cargo)
#   - Python 3 with packages from benchmarks/requirements.txt
#   - ~5–10 min of CPU time (FHE bootstrapping is expensive)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL="${1:-tests/fixtures/trained_binary.ubj}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Python benchmark
# ---------------------------------------------------------------------------
# Use WEIRWOOD_PYTHON to point at a venv's interpreter, e.g.:
#   WEIRWOOD_PYTHON=/path/to/venv/bin/python3 ./benchmarks/run_benchmark.sh
PYTHON="${WEIRWOOD_PYTHON:-python3}"

echo "=== Checking Python dependencies (using $PYTHON) ==="
"$PYTHON" -c "import xgboost, numpy" \
    || { echo "ERROR: xgboost/numpy not available. Set WEIRWOOD_PYTHON to a venv interpreter."; exit 1; }

echo ""
echo "=== Python (XGBoost, plaintext) ==="
PY_OUTPUT=$("$PYTHON" benchmarks/bench_python.py "$MODEL")
echo "$PY_OUTPUT"

PY_TOTAL=$(echo "$PY_OUTPUT" | awk '/total/      {print $3}')
PY_PER=$(  echo "$PY_OUTPUT" | awk '/per call/   {print $4}')
PY_THRU=$( echo "$PY_OUTPUT" | awk '/throughput/ {print $3}')

# ---------------------------------------------------------------------------
# Rust plaintext benchmark
# ---------------------------------------------------------------------------
echo ""
echo "=== Building weirwood benchmarks (release) ==="
cargo build --release --example bench_plaintext --example bench_fhe_full 2>&1 | tail -3

echo ""
echo "=== Rust (weirwood, plaintext) ==="
RUST_OUTPUT=$(./target/release/examples/bench_plaintext "$MODEL")
echo "$RUST_OUTPUT"

RUST_TOTAL=$(echo "$RUST_OUTPUT" | awk '/total/      {print $3}')
RUST_PER=$(  echo "$RUST_OUTPUT" | awk '/per call/   {print $4}')
RUST_THRU=$( echo "$RUST_OUTPUT" | awk '/throughput/ {print $3}')

# ---------------------------------------------------------------------------
# Rust FHE full-ensemble benchmark (expect ~5–10 min)
# ---------------------------------------------------------------------------
echo ""
echo "=== Rust (weirwood, FHE full ensemble — expect ~5–10 min) ==="
FHE_OUTPUT=$(./target/release/examples/bench_fhe_full "$MODEL")
echo "$FHE_OUTPUT"

FHE_KEYGEN_MS=$( echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_KEYGEN_MS=/  {print $2}')
FHE_ENC_MS=$(    echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_ENC_MS=/     {print $2}')
FHE_INFER_S=$(   echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_INFER_S=/    {print $2}')
FHE_DEC_MS=$(    echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_DEC_MS=/     {print $2}')
FHE_THRU=$(      echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_THRU=/       {print $2}')
DELTA=$(         echo "$FHE_OUTPUT" | awk -F= '/^BENCH_DELTA=/           {print $2}')
PBS_OPS=$(       echo "$FHE_OUTPUT" | awk -F= '/^BENCH_PBS_OPS=/         {print $2}')

# ---------------------------------------------------------------------------
# Build markdown table and inject into README.md
# ---------------------------------------------------------------------------
RUN_DATE=$(date -u '+%Y-%m-%d')

export PY_TOTAL PY_PER PY_THRU RUST_TOTAL RUST_PER RUST_THRU \
       FHE_KEYGEN_MS FHE_ENC_MS FHE_INFER_S FHE_DEC_MS FHE_THRU \
       DELTA PBS_OPS RUN_DATE MODEL

"$PYTHON" - <<'PYEOF'
import os, re

readme_path = "README.md"

py_total      = os.environ["PY_TOTAL"]
py_per        = os.environ["PY_PER"]
py_thru       = os.environ["PY_THRU"]
rust_total    = os.environ["RUST_TOTAL"]
rust_per      = os.environ["RUST_PER"]
rust_thru     = os.environ["RUST_THRU"]
fhe_keygen_ms = os.environ["FHE_KEYGEN_MS"]
fhe_enc_ms    = os.environ["FHE_ENC_MS"]
fhe_infer_s   = os.environ["FHE_INFER_S"]
fhe_dec_ms    = os.environ["FHE_DEC_MS"]
fhe_thru      = os.environ["FHE_THRU"]
delta         = os.environ["DELTA"]
pbs_ops       = os.environ["PBS_OPS"]
run_date      = os.environ["RUN_DATE"]
model         = os.environ["MODEL"]

fhe_infer_s_f = float(fhe_infer_s)
if fhe_infer_s_f >= 60:
    fhe_per_display = f"{fhe_infer_s_f / 60:.1f} min"
else:
    fhe_per_display = f"{fhe_infer_s_f:.1f} s"

table = (
    f"<!-- BENCHMARK_TABLE_START -->\n"
    f"_Last run: {run_date} · model: `{model}` · plaintext: 100,000 iterations · FHE: avg 10 runs_\n\n"
    f"| Backend                        | Per call        | Throughput (inf/s) | Notes                              |\n"
    f"|--------------------------------|-----------------|--------------------|------------------------------------|\n"
    f"| weirwood (Rust, plaintext)     | {rust_per:>9} ns    | {rust_thru:>18} |                                    |\n"
    f"| XGBoost (Python, plaintext)    | {py_per:>9} ns    | {py_thru:>18} |                                    |\n"
    f"| weirwood (Rust, **FHE**)       | {fhe_per_display:>9}       | {fhe_thru:>18} | avg 10 runs, {pbs_ops} PBS ops        |\n\n"
    f"FHE phase breakdown: keygen {fhe_keygen_ms} ms · encrypt {fhe_enc_ms} ms · "
    f"inference {fhe_infer_s} s (avg 10) · decrypt {fhe_dec_ms} ms · |Δ plaintext| = {delta}\n"
    f"<!-- BENCHMARK_TABLE_END -->"
)

with open(readme_path) as f:
    content = f.read()

content = re.sub(
    r"<!-- BENCHMARK_TABLE_START -->.*?<!-- BENCHMARK_TABLE_END -->",
    table,
    content,
    flags=re.DOTALL,
)

with open(readme_path, "w") as f:
    f.write(content)

print(f"\nREADME.md updated with results from {run_date}.")
PYEOF

echo ""
echo "Benchmark complete."
