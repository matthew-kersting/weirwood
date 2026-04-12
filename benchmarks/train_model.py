"""Train and save a binary classification XGBoost model on Breast Cancer Wisconsin.

Uses sklearn's built-in breast_cancer dataset (569 samples, 30 features,
binary: malignant=0 / benign=1). Features are StandardScaler-normalized so
they fit within weirwood's fixed-point encoding (SCALE=100, i16, range ±3.27).

Produces:
  tests/fixtures/trained_binary.json
  tests/fixtures/trained_binary.ubj

After running this script, copy the printed Rust constants into
tests/integration.rs to keep the expected-output tests in sync.

Usage:
    python3 benchmarks/train_model.py
    python3 benchmarks/train_model.py --n_estimators 100 --max_depth 8
"""

import argparse
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load and normalize the dataset.  StandardScaler maps each feature to
    # mean=0 / std=1, keeping ~99.7% of values within [-3, 3] — well inside
    # the ±3.27 range imposed by weirwood's SCALE=100 / i16 encoding.
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

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
    print(f"Dataset:  breast_cancer  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Trained:  n_estimators={args.n_estimators}  max_depth={args.max_depth}  "
          f"test_acc={acc:.3f}")

    booster = model.get_booster()

    booster.save_model("tests/fixtures/trained_binary.json")
    booster.save_model("tests/fixtures/trained_binary.ubj")

    with open("tests/fixtures/trained_binary.json") as f:
        raw = json.load(f)
    trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    total_internal = sum(
        sum(1 for x in t["left_children"] if x != -1) for t in trees
    )
    max_internal = max(
        sum(1 for x in t["left_children"] if x != -1) for t in trees
    )
    print(f"Saved:    trees={len(trees)}  total_internal_nodes={total_internal}  "
          f"max_internal_per_tree={max_internal}")

    # Pick 7 representative test samples: the first 5 from the test set plus
    # the first malignant and first benign sample in the full dataset (normalized).
    n_features = X_test.shape[1]
    test_vecs = X_test[:7].copy()
    probas = booster.predict(xgb.DMatrix(test_vecs))
    true_labels = y_test[:7].astype(int)

    print(f"\nReference outputs (normalized features, P(benign=1)):")
    for i, (vec, p, label) in enumerate(zip(test_vecs, probas, true_labels)):
        print(f"  sample {i}: true={label}  pred={p:.8f}")

    # Print Rust constants ready to paste into tests/integration.rs
    print(f"\n// ---- paste into tests/integration.rs ----")
    print(f"const TRAINED_TEST_VECTORS: &[[f32; {n_features}]] = &[")
    for vec in test_vecs:
        vals = ", ".join(f"{v:.6f}_f32" for v in vec)
        print(f"    [{vals}],")
    print("];")
    print()
    proba_vals = ", ".join(f"{p:.8f}" for p in probas)
    print(f"const TRAINED_EXPECTED_PROBA: &[f32] = &[{proba_vals}];")
    print("// -------------------------------------------")


if __name__ == "__main__":
    main()
