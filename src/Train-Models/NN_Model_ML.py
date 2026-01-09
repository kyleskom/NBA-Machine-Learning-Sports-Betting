import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models"

DEFAULT_DATASET = "dataset_2012-24_new"
TARGET_COLUMN = "Home-Team-Win"
DATE_COLUMN = "Date"
DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    "OU",
    "OU-Cover",
]
NUM_CLASSES = 2


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[TARGET_COLUMN].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.to_numpy(dtype=float)
    X = tf.keras.utils.normalize(X, axis=1)
    return X, y


def split_time_series(X, y, val_size=0.1, test_size=0.1):
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def sample_params(rng):
    depth = int(rng.integers(2, 7))
    width = int(rng.choice([64, 96, 128, 160, 192, 256]))
    units = [max(32, int(width / (2 ** i))) for i in range(depth)]
    return {
        "units": units,
        "dropout": float(rng.uniform(0.0, 0.4)),
        "batch_norm": bool(rng.integers(0, 2)),
        "l2": float(10 ** rng.uniform(np.log10(1e-6), np.log10(1e-4))),
        "lr": float(10 ** rng.uniform(np.log10(5e-5), np.log10(5e-3))),
        "batch_size": int(rng.choice([64, 128, 256])),
        "epochs": int(rng.integers(40, 121)),
    }


def build_model(input_dim, params):
    reg = tf.keras.regularizers.l2(params["l2"]) if params["l2"] > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in params["units"]:
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if params["batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        if params["dropout"] > 0:
            x = tf.keras.layers.Dropout(params["dropout"])(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def format_param(value, precision=3):
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Train NN moneyline model (random search).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--trials", type=int, default=50, help="Number of parameter trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation size.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test size.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.optimizer.set_experimental_options({"remapping": False})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y = prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(
        X, y, val_size=args.val_size, test_size=args.test_size
    )
    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None}
    temp_path = MODEL_DIR / "best_ml_temp.keras"

    for trial in range(1, args.trials + 1):
        tf.keras.backend.clear_session()
        params = sample_params(rng)
        model = build_model(X_train.shape[1], params)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=params["lr"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        val_loss = float(np.min(history.history["val_loss"]))
        if not np.isfinite(val_loss):
            print(f"Trial {trial}/{args.trials}: val loss invalid, skipping")
            continue
        print(f"Trial {trial}/{args.trials}: val loss {val_loss:.4f}")

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            model.save(temp_path)

    if best["params"] is None or not temp_path.exists():
        print("No valid model trained.")
        return

    best_model = tf.keras.models.load_model(temp_path, compile=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=best["params"]["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Best val loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

    params = best["params"]
    model_name = (
        f"Trained-Model-ML-{accuracy * 100:.1f}"
        f"_d{len(params['units'])}"
        f"_u{params['units'][0]}"
        f"_dr{format_param(params['dropout'])}"
        f"_bn{int(params['batch_norm'])}"
        f"_l2{format_param(params['l2'])}"
        f"_lr{format_param(params['lr'])}"
        f"_bs{params['batch_size']}.keras"
    )
    model_path = MODEL_DIR / model_name
    best_model.save(model_path)
    temp_path.unlink(missing_ok=True)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
