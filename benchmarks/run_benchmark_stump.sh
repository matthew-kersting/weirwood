#!/usr/bin/env bash
# run_benchmark_stump.sh — benchmark FHE vs plaintext (Rust) vs plaintext
# (Python) on the single-stump regression fixture, then update README.md.
#
# The stump is the simplest possible XGBoost model (depth 1, one tree), and
# the first model for which weirwood supports end-to-end FHE evaluation.
# This benchmark emphasises latency rather than throughput because each FHE
# inference requires one full TFHE bootstrapping chain (~seconds on CPU).
#
# Usage:
#   ./benchmarks/run_benchmark_stump.sh
#   ./benchmarks/run_benchmark_stump.sh tests/fixtures/stump_regression.json
#
# Requirements:
#   - Rust toolchain (cargo)
#   - Python 3 with xgboost and numpy (see benchmarks/requirements.txt)
#   - ~5–15 min of CPU time (FHE bootstrapping is expensive)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL="${1:-tests/fixtures/stump_regression.json}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Python plaintext benchmark
# ---------------------------------------------------------------------------
# Use WEIRWOOD_PYTHON to point at a venv's interpreter, e.g.:
#   WEIRWOOD_PYTHON=/path/to/venv/bin/python3 ./benchmarks/run_benchmark_stump.sh
PYTHON="${WEIRWOOD_PYTHON:-python3}"

echo "=== Checking Python dependencies (using $PYTHON) ==="
"$PYTHON" -c "import xgboost, numpy" \
    || { echo "ERROR: xgboost/numpy not available. Set WEIRWOOD_PYTHON to a venv interpreter."; exit 1; }

echo ""
echo "=== Python (XGBoost, plaintext stump) ==="
PY_OUTPUT=$("$PYTHON" benchmarks/bench_python_stump.py)
echo "$PY_OUTPUT"

PY_NS=$(   echo "$PY_OUTPUT" | awk -F= '/^BENCH_PY_NS=/   {print $2}')
PY_THRU=$( echo "$PY_OUTPUT" | awk -F= '/^BENCH_PY_THRU=/ {print $2}')

# ---------------------------------------------------------------------------
# Rust FHE benchmark (release build required — expect several minutes)
# ---------------------------------------------------------------------------
echo ""
echo "=== Building weirwood FHE benchmark (release) ==="
cargo build --release --example bench_fhe_stump 2>&1 | tail -3

echo ""
echo "=== Rust FHE stump benchmark (this will take several minutes) ==="
FHE_OUTPUT=$(./target/release/examples/bench_fhe_stump "$MODEL")
echo "$FHE_OUTPUT"

PLAIN_NS=$(      echo "$FHE_OUTPUT" | awk -F= '/^BENCH_PLAIN_NS=/       {print $2}')
PLAIN_THRU=$(    echo "$FHE_OUTPUT" | awk -F= '/^BENCH_PLAIN_THRU=/     {print $2}')
FHE_KEYGEN_MS=$( echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_KEYGEN_MS=/  {print $2}')
FHE_ENC_MS=$(    echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_ENC_MS=/     {print $2}')
FHE_INFER_S=$(   echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_INFER_S=/    {print $2}')
FHE_DEC_MS=$(    echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_DEC_MS=/     {print $2}')
FHE_THRU=$(      echo "$FHE_OUTPUT" | awk -F= '/^BENCH_FHE_THRU=/       {print $2}')
DELTA=$(         echo "$FHE_OUTPUT" | awk -F= '/^BENCH_DELTA=/           {print $2}')

# ---------------------------------------------------------------------------
# Inject FHE stump table into README.md
# ---------------------------------------------------------------------------
RUN_DATE=$(date -u '+%Y-%m-%d')

export PLAIN_NS PLAIN_THRU PY_NS PY_THRU \
       FHE_KEYGEN_MS FHE_ENC_MS FHE_INFER_S FHE_DEC_MS FHE_THRU \
       DELTA RUN_DATE MODEL

"$PYTHON" - <<'PYEOF'
import os, re

readme_path = "README.md"

plain_ns      = os.environ["PLAIN_NS"]
plain_thru    = os.environ["PLAIN_THRU"]
py_ns         = os.environ["PY_NS"]
py_thru       = os.environ["PY_THRU"]
fhe_keygen_ms = os.environ["FHE_KEYGEN_MS"]
fhe_enc_ms    = os.environ["FHE_ENC_MS"]
fhe_infer_s   = os.environ["FHE_INFER_S"]
fhe_dec_ms    = os.environ["FHE_DEC_MS"]
fhe_thru      = os.environ["FHE_THRU"]
delta         = os.environ["DELTA"]
run_date      = os.environ["RUN_DATE"]
model         = os.environ["MODEL"]

# FHE inference converts to ms for the per-call column
fhe_infer_ms = f"{float(fhe_infer_s) * 1000:.0f}"

table = (
    f"<!-- FHE_STUMP_TABLE_START -->\n"
    f"_Last run: {run_date} · model: `{model}` · stump (depth 1, 1 tree)_\n\n"
    f"> **Note:** FHE latency is the average of 5 bootstrapping runs;\n"
    f"> plaintext throughput uses 10,000 iterations.\n"
    f"> Key generation and encryption are one-time client costs.\n\n"
    f"| Backend                        | Per call          | Throughput (inf/s) | Notes                          |\n"
    f"|--------------------------------|-------------------|--------------------|--------------------------------|\n"
    f"| weirwood (Rust, plaintext)     | {plain_ns:>8} ns      | {plain_thru:>18} |                                |\n"
    f"| XGBoost (Python, plaintext)    | {py_ns:>8} ns      | {py_thru:>18} |                                |\n"
    f"| weirwood (Rust, **FHE**)       | {fhe_infer_ms:>8} ms      | {fhe_thru:>18} | avg 5 runs, 1 PBS op each      |\n\n"
    f"FHE phase breakdown: keygen {fhe_keygen_ms} ms · encrypt {fhe_enc_ms} ms · "
    f"inference {fhe_infer_s} s (avg 5) · decrypt {fhe_dec_ms} ms · |Δ plaintext| = {delta}\n"
    f"<!-- FHE_STUMP_TABLE_END -->"
)

with open(readme_path) as f:
    content = f.read()

if "<!-- FHE_STUMP_TABLE_START -->" in content:
    content = re.sub(
        r"<!-- FHE_STUMP_TABLE_START -->.*?<!-- FHE_STUMP_TABLE_END -->",
        table,
        content,
        flags=re.DOTALL,
    )
else:
    # Append a new section before the Performance notes heading
    content = content.replace(
        "## Performance notes",
        "## FHE Stump Benchmark\n\n"
        + table
        + "\n\n## Performance notes",
    )

with open(readme_path, "w") as f:
    f.write(content)

print(f"\nREADME.md updated with FHE stump results from {run_date}.")
PYEOF

echo ""
echo "Benchmark complete."
