#!/usr/bin/env bash
# run_benchmark.sh — compare weirwood (Rust) vs XGBoost (Python) plaintext
# inference throughput, then update the Benchmarks table in README.md.
#
# Usage:
#   ./benchmarks/run_benchmark.sh
#   ./benchmarks/run_benchmark.sh tests/fixtures/trained_binary.json
#
# Requirements:
#   - Rust toolchain (cargo)
#   - Python 3 with packages from benchmarks/requirements.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL="${1:-tests/fixtures/trained_binary.ubj}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Python benchmark
# ---------------------------------------------------------------------------
echo "=== Installing Python dependencies ==="
python3 -m pip install -q -r benchmarks/requirements.txt

echo ""
echo "=== Python (XGBoost) ==="
PY_OUTPUT=$(python3 benchmarks/bench_python.py "$MODEL")
echo "$PY_OUTPUT"

PY_TOTAL=$(echo "$PY_OUTPUT" | awk '/total/      {print $3}')
PY_PER=$(  echo "$PY_OUTPUT" | awk '/per call/   {print $4}')
PY_THRU=$( echo "$PY_OUTPUT" | awk '/throughput/ {print $3}')

# ---------------------------------------------------------------------------
# Rust benchmark
# ---------------------------------------------------------------------------
echo ""
echo "=== Building weirwood benchmark (release) ==="
cargo build --release --example bench_plaintext 2>&1 | tail -3

echo ""
echo "=== Rust (weirwood) ==="
RUST_OUTPUT=$(./target/release/examples/bench_plaintext "$MODEL")
echo "$RUST_OUTPUT"

RUST_TOTAL=$(echo "$RUST_OUTPUT" | awk '/total/      {print $3}')
RUST_PER=$(  echo "$RUST_OUTPUT" | awk '/per call/   {print $4}')
RUST_THRU=$( echo "$RUST_OUTPUT" | awk '/throughput/ {print $3}')

# ---------------------------------------------------------------------------
# Build markdown table and inject into README.md
# ---------------------------------------------------------------------------
RUN_DATE=$(date -u '+%Y-%m-%d')

export PY_TOTAL PY_PER PY_THRU RUST_TOTAL RUST_PER RUST_THRU RUN_DATE MODEL

python3 - <<'PYEOF'
import os, re

readme_path = "README.md"

py_total  = os.environ["PY_TOTAL"]
py_per    = os.environ["PY_PER"]
py_thru   = os.environ["PY_THRU"]
rust_total = os.environ["RUST_TOTAL"]
rust_per   = os.environ["RUST_PER"]
rust_thru  = os.environ["RUST_THRU"]
run_date   = os.environ["RUN_DATE"]
model      = os.environ["MODEL"]

table = (
    f"<!-- BENCHMARK_TABLE_START -->\n"
    f"_Last run: {run_date} · model: `{model}` · 100,000 iterations_\n\n"
    f"| Backend                    | Total (ms)   | Per call (ns) | Throughput (inf/sec) |\n"
    f"|----------------------------|-------------|---------------|---------------------|\n"
    f"| weirwood (Rust, plaintext) | {rust_total:>11} | {rust_per:>13} | {rust_thru:>19} |\n"
    f"| XGBoost (Python)  | {py_total:>11} | {py_per:>13} | {py_thru:>19} |\n"
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
