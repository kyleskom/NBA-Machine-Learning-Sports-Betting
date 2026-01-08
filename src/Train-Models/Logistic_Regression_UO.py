import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"

DEFAULT_DATASET = "dataset_2012-26_new"
TARGET_COLUMN = "OU-Cover"
DATE_COLUMN = "Date"
DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "TEAM_ID",
    "Date",
    "index.1",
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
    X = data.drop(columns=DROP_COLUMNS, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y


def split_time_series(X, y, val_size=0.1, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    X_train, y_train = X.iloc[:val_start], y[:val_start]
    X_test, y_test = X.iloc[test_start:], y[test_start:]
    return X_train, X_test, y_train, y_test


def split_train_test(X, y, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    return X.iloc[:test_start], y[:test_start], X.iloc[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = int(n * (1 - calib_size))
    return X.iloc[:calib_start], y[:calib_start], X.iloc[calib_start:], y[calib_start:]


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression totals model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--trials", type=int, default=50, help="Number of hyperparameter trials.")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward CV splits.")
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        choices=["sigmoid", "isotonic", "none"],
        help="Calibration method for probabilities.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y = prepare_data(df)
    X_train_val, y_train_val, X_test, y_test = split_train_test(X, y)

    rng = np.random.default_rng(args.seed)
    best = {
        "val_loss": float("inf"),
        "params": None,
    }

    for trial in range(1, args.trials + 1):
        C = float(10 ** rng.uniform(np.log10(0.01), np.log10(10.0)))
        penalty = rng.choice(["l2", "l1"])
        solver = "saga" if penalty == "l1" else "lbfgs"

        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                C=C,
                penalty=penalty,
                solver=solver,
                multi_class="multinomial",
            ),
        )

        tscv = TimeSeriesSplit(n_splits=args.splits)
        losses = []
        for train_idx, val_idx in tscv.split(X_train_val):
            base_model.fit(X_train_val.iloc[train_idx], y_train_val[train_idx])
            val_proba = base_model.predict_proba(X_train_val.iloc[val_idx])
            loss = log_loss(y_train_val[val_idx], val_proba, labels=[0, 1, 2])
            losses.append(loss)
        val_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f}")

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = {"C": C, "penalty": penalty, "solver": solver}

    if best["params"] is None:
        print("No valid model trained.")
        return

    X_train, y_train, X_calib, y_calib = split_train_calib(X_train_val, y_train_val)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=best["params"]["C"],
            penalty=best["params"]["penalty"],
            solver=best["params"]["solver"],
            multi_class="multinomial",
        ),
    )
    model.fit(X_train, y_train)

    if args.calibration == "none":
        calibrated = model
    else:
        calibrated = CalibratedClassifierCV(model, method=args.calibration, cv="prefit")
        calibrated.fit(X_calib, y_calib)

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba, labels=[0, 1, 2])

    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {loss:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
