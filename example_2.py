"""End-to-end demo of MaltaRock: download -> features -> well-wise split -> train -> evaluate.

Examples:
    python script.py                                   # quick run (40 wells, Random Forest)
    python script.py --wells 400 --trees 100           # closer to the paper setting
    python script.py --optuna 3                        # + Optuna tuning on training wells only
    python script.py --model LSTM                      # sequence model (needs torch)
"""

from __future__ import annotations

import argparse
import logging
import time

import MaltaRock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="./data", help="dataset directory")
    parser.add_argument("--revision", default=None, help="dataset commit to pin (default: cached / main)")
    parser.add_argument("--wells", type=int, default=40, help="total_wells (first N wells, sorted)")
    parser.add_argument("--window", type=int, default=5, help="rolling window; 0 disables rolling stats")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="RandomForest",
                        choices=["RandomForest", "DecisionTree", "KNN", "LogisticRegression", "XGBoost", "LSTM", "BILSTM"])
    parser.add_argument("--trees", type=int, default=50, help="n_estimators for RandomForest/XGBoost")
    parser.add_argument("--optuna", type=int, default=0, help="number of Optuna trials (0 = no tuning)")
    parser.add_argument("--output", default="results", help="where results.json / well_split.json go")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("MaltaRock").setLevel(logging.INFO)

    def step(title: str) -> float:
        print(f"\n=== {title} ===")
        return time.time()

    # 1. Download (no network if the cache is complete)
    t = step("1. download_dataset")
    path = MaltaRock.download_dataset(args.data, revision=args.revision)
    manifest = MaltaRock.read_manifest(args.data)
    print(f"{path} | revision {manifest.revision} | {len(manifest.files)} files "
          f"| {sum(f.size for f in manifest.files) / 1e6:.1f} MB | {time.time() - t:.2f}s")

    # 2. Features
    t = step("2. fetch_data")
    features = MaltaRock.fetch_data(
        path=args.data,
        revision=args.revision,
        total_wells=args.wells,
        apply_scaling=True,
        apply_presence=True,
        apply_rolling_stats=args.window > 0,
        window=max(args.window, 1),
        labels=MaltaRock.CORE15_LABELS,
    )
    print(f"shape {features.shape} | {features['well'].nunique()} wells | {time.time() - t:.1f}s")
    print("columns:", list(features.columns))
    print("class counts:\n" + features["rock"].value_counts().to_string())

    # 3. Well-wise split
    t = step("3. split_train_test")
    split = MaltaRock.split_train_test(features, test_size=args.test_size, random_state=args.seed)
    X_train, y_train, X_test, y_test = split
    overlap = set(split.train_wells) & set(split.test_wells)
    print(f"train: {len(split.train_wells)} wells / {len(X_train)} rows | "
          f"test: {len(split.test_wells)} wells / {len(X_test)} rows | shared wells: {len(overlap)}")
    print("X columns:", X_train.shape[1], "| metadata in X:", {"well", "depth", "rock"} & set(X_train.columns) or "none")
    print("test wells:", split.test_wells[:10], "..." if len(split.test_wells) > 10 else "")
    assert not overlap

    # 4. Model
    params: dict = {}
    if args.model in ("RandomForest", "XGBoost"):
        params["n_estimators"] = args.trees
    if args.model in ("RandomForest", "KNN", "XGBoost"):
        params["n_jobs"] = -1
    if args.model in ("LSTM", "BILSTM"):
        params.update(epochs=5, patience=2, num_workers=0, batch_size=1024, device="cpu")
    model = MaltaRock.get_model_class(args.model)(**params)

    if args.optuna:
        t = step(f"4a. optimize ({args.optuna} trials, GroupKFold on training wells)")
        study = model.optimize(X_train, y_train, n_trials=args.optuna, n_splits=3, random_state=args.seed)
        print(f"best CV f1_macro {study.best_value:.4f} | params {study.best_params} | {time.time() - t:.1f}s")

    t = step(f"4. train {args.model}")
    model.train(X_train, y_train)
    print(f"trained in {time.time() - t:.1f}s")

    # 5. Evaluate on held-out wells
    t = step("5. evaluate (test wells)")
    results = model.evaluate(X_test, y_test)
    print(f"accuracy    {results['accuracy']:.4f}")
    print(f"f1_macro    {results['f1_macro']:.4f}")
    print(f"f1_weighted {results['f1_weighted']:.4f}")
    print(f"absent classes: {results['absent_classes']} | unseen classes: {results['unseen_classes']}")
    print(f"{'class':<14}{'prec':>7}{'rec':>7}{'f1':>7}{'support':>10}")
    for label, s in results["per_class"].items():
        print(f"{label:<14}{s['precision']:>7.3f}{s['recall']:>7.3f}{s['f1']:>7.3f}{s['support']:>10}")

    # 6. Save results and the well assignment (reusable via load_well_assignment)
    from pathlib import Path

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    MaltaRock.save_results(results, out / f"{args.model}_results.json")
    split.save_assignment(out / "well_split.json", test_size=args.test_size, random_state=args.seed)
    print(f"\nsaved {out / f'{args.model}_results.json'} and {out / 'well_split.json'}")


if __name__ == "__main__":
    main()
