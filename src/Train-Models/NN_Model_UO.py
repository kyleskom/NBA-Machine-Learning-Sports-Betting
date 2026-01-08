import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "NN_Models"
LOG_DIR = BASE_DIR / "Logs"

DEFAULT_DATASET = "dataset_2012-26_new"
TARGET_COLUMN = "OU-Cover"
DATE_COLUMN = "Date"
DROP_COLUMNS = [
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "TEAM_ID",
    "Date",
    "TEAM_NAME.1",
    "TEAM_ID.1",
    "Date.1",
    "OU-Cover",
]


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[TARGET_COLUMN].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore").astype(float).to_numpy()
    return X, y


def split_time_series(X, y, val_size=0.1, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(192, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description="Train NN totals model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y = prepare_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

    model = build_model(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    run_id = str(time.time())
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR / run_id)),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / f"Trained-Model-OU-{run_id}"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
