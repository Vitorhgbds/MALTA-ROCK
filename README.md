# MALTA-ROCK

**Rock-type Optimization and Classification Kit.** This is a Python library for reproducible lithology classification on public Brazilian ANP well logs. It works with:

- [MALTA-WELD](../MALTA-WELD), which collects the well logs and lithology descriptions and builds the dataset.
- [MALTA-WLOGS](https://huggingface.co/datasets/MALTA-Lab/MALTA-WLOGS), the dataset MALTA-WELD publishes on Hugging Face.
- MALTA-ROCK (this repository), which downloads MALTA-WLOGS, prepares the features, splits the data by well, and trains and evaluates the models from the paper.

## Installation

Python ≥ 3.13.

```bash
pip install -e .              # core: data pipeline + scikit-learn models
pip install -e ".[xgboost]"   # + XGBoost
pip install -e ".[deep]"      # + LSTM / BILSTM (PyTorch)
pip install -e ".[all]"       # everything
# with uv: uv sync --all-extras --group dev
```

`import MaltaRock` is lightweight. Model classes are imported only when you first access them, so `torch` and `xgboost` are needed only for the models that use them.

## Quick start

```python
import MaltaRock

MaltaRock.download_dataset("./data")          # no-op if already cached

features = MaltaRock.fetch_data(
    path="./data",
    total_wells=400,
    apply_scaling=True,          # physical-range outlier filter (no rescaling!)
    apply_presence=True,
    apply_rolling_stats=True,
    window=5,
    labels=MaltaRock.CORE15_LABELS,   # the paper's 15 classes
)

X_train, y_train, X_test, y_test = MaltaRock.split_train_test(
    features, test_size=0.2, random_state=42,
)

model = MaltaRock.RandomForest(random_state=42, n_estimators=10,
                               min_samples_split=6, min_samples_leaf=8, max_features=0.5)
model.train(X_train, y_train)
results = model.evaluate(X_test, y_test)
print(results["accuracy"], results["f1_macro"], results["f1_weighted"])
```

## Dataset source and local cache

**Source.** The data comes from the public dataset repository `MALTA-Lab/MALTA-WLOGS`. No Hugging Face login is needed. Its dataset card declares four configurations. MALTA-ROCK downloads **only** `BronzeFeatures`, whose `data_files` are `Features/*.parquet`; it never fetches `LogsFeatures`, `AGPSummary` or `AGPLithology`. The file list comes from the card at the requested revision, and the files are fetched with `HfApi.dataset_info` and `hf_hub_download`.

**Layout.**

```
./data/MALTA-WLOGS/
├── manifest.json                 # written last, atomically
└── Features/train-00000.parquet … train-NNNNN.parquet
```

**Caching rules.**

- `manifest.json` records:
  - the repository id and configuration name
  - the `data_files` pattern
  - the requested revision and the resolved commit
  - the size and SHA-256 of every file
- **When the cache is reused:** the cache counts as complete only if three things hold:
  - the manifest is valid
  - it matches the requested revision
  - every listed file exists with the recorded size

  A complete cache is reused **without any network access**. Calling `download_dataset` or `fetch_data` again does not download anything.
- **When files are re-downloaded:** missing, truncated, size-mismatched or checksum-mismatched files are fetched again. So is everything when a different revision is requested or `force=True` is passed.
  - Before downloading, the old manifest is deleted, and the new one is written only after every file has been verified against the SHA-256 published on the Hub. An interrupted download is therefore never treated as complete.
  - Files that already match are kept, so a failed download can be resumed.
  - Files left over from another revision are removed.
- `is_dataset_complete(path, revision, verify_checksums=True)` re-hashes every file without contacting the Hub.
- `fetch_data(..., download=False)` never downloads anything. If the data is missing, it raises `DatasetNotFoundError`.

**Pinning a revision.** The `main` branch can change over time. To get exactly the same data later, pin the dataset commit:

```python
MaltaRock.download_dataset("./data", revision="dd7db33c4803780d629ced50ebb61ddd41575a64")
features = MaltaRock.fetch_data("./data", revision="dd7db33c4803780d629ced50ebb61ddd41575a64", ...)
print(features.attrs["malta_rock"]["revision"])   # provenance of the loaded table
```

With `revision=None`, any complete cache is reused as is. Pass `force=True` or a specific revision to update it.

### Bronze-feature schema (verified on the Hub)

This was verified at commit `dd7db33c4803780d629ced50ebb61ddd41575a64`. The 113 Parquet shards are the SQLite `Features` table that MALTA-WELD exported in 500 000-row chunks (`MALTA-WELD/weld/huggingface.py`):

| column | Arrow type | meaning |
|---|---|---|
| `well` | string | well identifier (grouping key) |
| `depth` | double | depth of the sample |
| `GR`, `SP`, `DT`, `CALI`, `ILD`, `RHOB`, `NPHI`, `MSFL`, `DRHO`, `SFLU` | double (shards where a curve is entirely missing store it as the Arrow `null` type, and the loader casts it to float64) | conventional logs |
| `top`, `bottom` | double | limits of the lithology interval (not used) |
| `rock` | string | lithology class (target) |

The loader reads `well`, `depth`, the ten logs and `rock`. It raises `SchemaError` if a required column is missing or has an incompatible type.

## Feature preparation (`fetch_data`)

`fetch_data` reproduces `DefaultFeatures` from the reference implementation (`GeoAI-ANP-DataExtractor/slide/models`), which built the table with one SQL query followed by pandas steps:

1. **Well subset** (`total_wells=N`). Keeps the first N wells in ascending well-id order. The SQL `SELECT DISTINCT well … LIMIT N` ran through the `(well, depth)` index, so this is the order the original experiments used.
   - With `labels`, a well qualifies only if it has at least one row of a requested class.
2. **Label filter.** Keeps rows whose `rock` is in `labels`, case-insensitively (ASCII only, like SQLite `lower()`). With `labels=None`, it keeps every row with a non-null class.
3. **Outlier handling** (`apply_scaling`). Values outside fixed physical ranges become NaN, for example GR outside [0, 200] or RHOB outside [1.95, 2.95]. The full list is `MaltaRock.data.schema.VALID_RANGES`.
4. Rows are sorted by `well` and then `depth`.
5. **Presence indicators** (`apply_presence`). Adds a `<LOG>_present` column that is 1 when the value exists and 0 otherwise.
6. **Rolling statistics** (`apply_rolling_stats`, `window`). Adds `<LOG>_rolling_mean` and `<LOG>_rolling_std`.
   - Each is computed over a trailing window of `window` rows, using whatever observations are available (`min_periods=1`).
   - Windows are computed **separately for each well, in depth order**, so they never cross a well boundary.

The result is a DataFrame with these columns: `well, depth, <10 logs>, [presence], [rolling], rock`. Every step is deterministic and uses only values from the same row or the same well. **No statistic is learned from the data before the split.** Missing-value imputation (median) and standardisation are fitted **inside each model, on the training wells only**.

## Well-wise split (`split_train_test`)

```python
split = MaltaRock.split_train_test(features, test_size=0.2, random_state=42)
X_train, y_train, X_test, y_test = split           # unpacks like a tuple
split.train_wells, split.test_wells
MaltaRock.get_groups(X_train)                      # well id per row (for GroupKFold etc.)
split.save_assignment("well_split.json")
```

Leakage guarantees:

- The split is made over **unique wells**, never over individual rows. Every well, with all of its depth samples, lands in exactly one partition.
- The random assignment sorts the wells and then calls `sklearn.model_selection.train_test_split(wells, test_size, random_state)`. This is exactly what the reference did, so a given seed gives **the same wells as the original experiments**. The result is deterministic for a fixed `random_state`.
- `X` contains only predictive columns. `well`, `depth` and `rock` are never in `X`; the `(well, depth)` index carries the group information instead, aligned row by row with `y`. Rows stay sorted by depth within each well.
- **Explicit assignments.** You can give the wells yourself:

  ```python
  MaltaRock.split_train_test(features, train_wells=[...], test_wells=[...])
  MaltaRock.split_train_test(features, test_wells=[...])           # all others → train
  MaltaRock.split_train_test(features, **MaltaRock.load_well_assignment("well_split.json"))
  ```

  Unknown wells, overlapping lists and empty partitions raise `SplitError`.

## Models

All models share the same interface: `train(X, y)`, `predict(X)`, `evaluate(X, y)`, `save(path)` / `load(path)`, `get_params()` / `set_params()` and `optimize(...)`. The legacy `ModelDataset(train, train_target, test, test_target)` container is still accepted by `train` and `evaluate`.

| class | estimator | extra dependency |
|---|---|---|
| `RandomForest` | `RandomForestClassifier` | – |
| `DecisionTree` | `DecisionTreeClassifier(class_weight="balanced")` | – |
| `KNN` | `KNeighborsClassifier` | – |
| `LogisticRegression` | `LogisticRegression` | – |
| `XGBoost` | `xgboost.XGBClassifier(tree_method="hist")` | `[xgboost]` |
| `LSTM` | LSTM over centred windows, one per well | `[deep]` |
| `BILSTM` | bidirectional LSTM over trailing windows, one per well | `[deep]` |

- **Tabular models.** These apply `SimpleImputer(median)`, then `StandardScaler`, then the estimator, with the imputer and scaler fitted on training rows only. A parameter left as `None` falls back to the scikit-learn default.
- **Sequence models.** These need the `(well, depth)` index, or an explicit `groups=` argument, and build their windows separately for each well (`MaltaRock.build_sequences`). Rows without a complete window inside their own well are not scored. `evaluate` reports how many rows were skipped as `n_rows_without_window`.

`evaluate` returns:

- `accuracy`, `f1_macro` and `f1_weighted`, computed with the same definitions as the reference (sklearn defaults).
- `per_class` (precision, recall, F1 and support for every known class) and `confusion_matrix` with its `labels`.
- `classification_report`.
- `absent_classes` (known classes with zero support in the partition) and `unseen_classes` (test classes never seen in training).
- `f1_macro_all_classes`, which averages over all known classes and scores absent classes as 0.

Use `MaltaRock.save_results(results, "results.json")` to export the results.

### Hyperparameter optimisation (Optuna)

```python
study = model.optimize(X_train, y_train, n_trials=15, metric="f1_macro", n_splits=3, random_state=42)
model.train(X_train, y_train)          # final fit with the best parameters
results = model.evaluate(X_test, y_test)
```

- **Validation stays inside the training wells.** Each trial is scored with `GroupKFold` over the **training wells only**, so the test wells are not touched until the final `evaluate`.
- **Search spaces.** By default, the reference search spaces are loaded from `models/configs/hyperparam-tuning/*.yaml`.
- **Sampler.** The sampler is a seeded TPE.
- **Persisting a study.** Pass `storage="sqlite:///optuna.db"` and a `study_name` to save a study and resume it later.

**Command line.**

```bash
malta-rock -d ./data download [--force] [--revision <commit>]
malta-rock -d ./data run --model-config src/MaltaRock/models/configs/hyperparam-tuning/forest.yaml \
    --num-trials 15 --cv-folds 3 --output results [--split-file well_split.json]
```

## Methodological notes and corrections

These differences from the reference code are intentional:

1. **Optuna selection leaked test information.** The reference `run.py` trained each trial on the training wells and scored it on the **test** wells. MaltaRock scores trials by grouped CV on the training wells. Expect slightly different selected hyperparameters and test scores.
2. **The LSTM/BiLSTM reference had four problems:**
   - It **stopped early on the test set**. MaltaRock now uses a validation subset of the training wells (`validation_size=0.1`).
   - It built windows over the concatenated table, so **windows crossed wells**. MaltaRock now builds them per well.
   - `LSTM.predict` did not apply the scaler. MaltaRock now does.
   - It hard-coded `lr=1e-3`. MaltaRock honours `learning_rate`, whose default is still `1e-3`. The YAML configs that set `learning_rate: 0.0001` previously trained with `1e-3`.
3. **`depth` as a feature.** The reference fed `depth` to the models as a feature. MaltaRock treats it as metadata and leaves it out of `X` by default. Pass `split_train_test(..., include_depth=True)`, or set `include_depth: true` in the config `policy`, to reproduce the original feature set.
4. **`rolling_window` was ignored.** The reference configs set `policy.rolling_window`, but `DefaultFeatures` read `window`, so the "rolling-10" experiments actually used `window=5`. `malta-rock run` honours `rolling_window`. Set it to 5 to reproduce the published rolling-10 numbers.
5. Logging verbosity of the estimators (`verbose`) defaults to 0.

## Development and offline tests

```bash
uv pip install -e ".[all]" pytest      # or: uv sync --all-extras --group dev
pytest                                   # fully offline, synthetic data
```

The tests use a fake Hugging Face Hub and small synthetic Parquet shards. They cover:

- file discovery
- cache reuse and refresh
- partial downloads
- schema errors
- feature engineering against the reference implementation
- well-wise splitting
- the models
- an Optuna run with 2 trials

They need neither network access nor the published dataset.

```
src/MaltaRock/
├── data/          download.py · loader.py · features.py · split.py · sequences.py · schema.py
├── models/        base.py · forest.py · tree.py · knn.py · logistic.py · xgboost.py · sequence.py · lstm.py · bilstm.py · configs/
├── evaluation/    metrics.py
├── optimization/  Optuna with grouped CV
├── run.py · cli.py
tests/
```

Downloaded data (`data/`), SQLite/Optuna databases (`*.db`), pickled models and `results/` are git-ignored.

**Upgrading from the old layout.** The package was previously importable as `src`; it is now `MaltaRock`. For example, `from src.models.forest import RandomForest` becomes `from MaltaRock import RandomForest`.
