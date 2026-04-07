"""Train and save a binary classification XGBoost model.

Produces tests/fixtures/trained_binary.json and trained_binary.ubj.
Features are two independent Gaussian features; the label is determined
by feature[0] > 0.5. This keeps the fixture simple and reproducible.

Usage:
    python3 benchmarks/train_model.py
    python3 benchmarks/train_model.py --n_estimators 100 --max_depth 8
"""

import argparse
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_samples = 2000
    X = rng.standard_normal((n_samples, 2)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    acc = (model.predict(X_test) == y_test).mean()
    print(f"Trained: n_estimators={args.n_estimators} max_depth={args.max_depth} "
          f"test_acc={acc:.3f}")

    booster = model.get_booster()

    # Count internal nodes
    trees_dump = booster.get_dump()
    total_trees = len(trees_dump)

    # Save both formats
    booster.save_model("tests/fixtures/trained_binary.json")
    booster.save_model("tests/fixtures/trained_binary.ubj")

    # Verify structure via weirwood's own JSON parser expectation
    with open("tests/fixtures/trained_binary.json") as f:
        raw = json.load(f)
    trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    total_internal = sum(
        sum(1 for x in t["left_children"] if x != -1) for t in trees
    )
    max_internal = max(
        sum(1 for x in t["left_children"] if x != -1) for t in trees
    )
    print(f"Saved:   trees={len(trees)} total_internal_nodes={total_internal} "
          f"max_internal_per_tree={max_internal}")

    # Print reference outputs for test vectors
    test_vecs = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [0.3, 0.7], [0.7, 0.3], [-1.0, 0.0], [2.0, 0.0],
    ], dtype=np.float32)
    probas = booster.predict(xgb.DMatrix(test_vecs))
    print("\nReference outputs (feature[0], feature[1] -> P(class=1)):")
    for vec, p in zip(test_vecs, probas):
        print(f"  [{vec[0]:5.1f}, {vec[1]:5.1f}] -> {p:.8f}")

if __name__ == "__main__":
    main()
