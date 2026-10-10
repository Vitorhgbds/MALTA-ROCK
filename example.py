import MaltaRock
import time

DATA_PATH = "./data"
TOTAL_WELLS = 40
APPLY_SCALING = True
APPLY_PRESENCE = True
APPLY_ROLLING_STATS = True
WINDOW_SIZE = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42



def download_dataset():
    t = time.time()
    
    path = MaltaRock.download_dataset(DATA_PATH)
    manifest: MaltaRock.Manifest = MaltaRock.read_manifest(DATA_PATH)

    print(f"{path} | revision {manifest.revision} | {len(manifest.files)} files "
            f"| {sum(f.size for f in manifest.files) / 1e6:.1f} MB | {time.time() - t:.2f}s")


def fetch_features():
    t = time.time()

    features = MaltaRock.fetch_data(
        path=DATA_PATH,
        total_wells=TOTAL_WELLS,
        apply_scaling=APPLY_SCALING,
        apply_presence=APPLY_PRESENCE,
        apply_rolling_stats=APPLY_ROLLING_STATS,
        window=WINDOW_SIZE,
        labels=MaltaRock.CORE15_LABELS,
    )

    print(f"shape {features.shape} | {features['well'].nunique()} wells | {time.time() - t:.1f}s")
    print("columns:", list(features.columns))
    print("class counts:\n" + features["rock"].value_counts().to_string())

    return features


def split_train_test(features):

    split = MaltaRock.split_train_test(features, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, y_train, X_test, y_test = split
    
    print(f"train: {len(split.train_wells)} wells / {len(X_train)} rows | "
            f"test: {len(split.test_wells)} wells / {len(X_test)} rows")
    print("X columns:", X_train.shape[1], "| metadata in X:", {"well", "depth", "rock"} & set(X_train.columns) or "none")
    print("test wells:", split.test_wells[:10], "..." if len(split.test_wells) > 10 else "")

    return split


def train_model(split):

    X_train, y_train, X_test, y_test = split

    model = MaltaRock.RandomForest()
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)

    print(f"accuracy    {results['accuracy']:.4f}")
    print(f"f1_macro    {results['f1_macro']:.4f}")
    print(f"f1_weighted {results['f1_weighted']:.4f}")
    print(f"absent classes: {results['absent_classes']} | unseen classes: {results['unseen_classes']}")
    print(f"{'class':<14}{'prec':>7}{'rec':>7}{'f1':>7}{'support':>10}")

    return model


if __name__ == "__main__":
    download_dataset()
    features = fetch_features()
    split = split_train_test(features)
    model = train_model(split)