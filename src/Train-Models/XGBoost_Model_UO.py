import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

DEFAULT_DATASET = "dataset_2012-26"
TARGET_COLUMN = "OU-Cover"
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
    "OU-Cover",
]
NUM_CLASSES = 3


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


def split_train_test(X, y, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = int(n * (1 - calib_size))
    return X[:calib_start], y[:calib_start], X[calib_start:], y[calib_start:]



def compute_sample_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def sample_params(rng, seed):
    eta = 10 ** rng.uniform(np.log10(0.003), np.log10(0.3))
    params = {
        "max_depth": int(rng.integers(2, 13)),
        "eta": float(eta),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.5, 1.0)),
        "colsample_bynode": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": int(rng.integers(1, 21)),
        "gamma": float(rng.uniform(0.0, 10.0)),
        "max_delta_step": int(rng.integers(0, 11)),
        "max_bin": int(rng.integers(128, 1025)),
        "lambda": float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0))),
        "alpha": float(10 ** rng.uniform(np.log10(0.01), np.log10(5.0))),
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": ["mlogloss", "merror"],
        "seed": seed,
        "tree_method": "hist",
    }
    num_boost_round = int(rng.integers(300, 2501))
    return params, num_boost_round


def train_model(X_train, y_train, X_val, y_val, params, num_boost_round):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=compute_sample_weights(y_train, num_classes=NUM_CLASSES))
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    return model


def format_param(value, precision=3):
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")


class BoosterWrapper:
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def walk_forward_cv_loss(X, y, params, num_boost_round, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for train_idx, val_idx in tscv.split(X):
        model = train_model(X[train_idx], y[train_idx], X[val_idx], y[val_idx], params, num_boost_round)
        val_probs = model.predict(xgb.DMatrix(X[val_idx]))
        loss = log_loss(y[val_idx], val_probs, labels=list(range(NUM_CLASSES)))
        losses.append(loss)
    return float(np.mean(losses)) if losses else None


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost totals model with walk-forward CV.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--trials", type=int, default=100, help="Number of parameter trials.")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward CV splits.")
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        choices=["sigmoid", "isotonic", "none"],
        help="Calibration method for probabilities.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
        "num_boost_round": None,
    }

    for trial in range(1, args.trials + 1):
        params, num_boost_round = sample_params(rng, seed=args.seed + trial)
        val_loss = walk_forward_cv_loss(X_train_val, y_train_val, params, num_boost_round, args.splits)
        if val_loss is None:
            continue

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            best["num_boost_round"] = num_boost_round

        print(f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f}")

    if best["params"] is None:
        print("No valid parameter set found.")
        return

    X_train, y_train, X_calib, y_calib = split_train_calib(X_train_val, y_train_val)
    best_model = train_model(
        X_train,
        y_train,
        X_calib,
        y_calib,
        best["params"],
        best["num_boost_round"],
    )

    calibrator = None
    if args.calibration == "none":
        probabilities = best_model.predict(xgb.DMatrix(X_test))
    else:
        calibrator = CalibratedClassifierCV(
            BoosterWrapper(best_model, NUM_CLASSES),
            method=args.calibration,
            cv="prefit",
        )
        calibrator.fit(X_calib, y_calib)
        probabilities = calibrator.predict_proba(X_test)

    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, probabilities, labels=list(range(NUM_CLASSES)))

    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {test_loss:.4f}")

    params = best["params"]
    model_name = (
        f"XGBoost_{accuracy * 100:.1f}%_UO"
        f"_md{params['max_depth']}"
        f"_eta{format_param(params['eta'])}"
        f"_sub{format_param(params['subsample'])}"
        f"_col{format_param(params['colsample_bytree'])}"
        f"_cbl{format_param(params['colsample_bylevel'])}"
        f"_cbn{format_param(params['colsample_bynode'])}"
        f"_mcw{params['min_child_weight']}"
        f"_g{format_param(params['gamma'])}"
        f"_mds{params['max_delta_step']}"
        f"_mb{params['max_bin']}"
        f"_l{format_param(params['lambda'])}"
        f"_a{format_param(params['alpha'])}"
        f"_nb{best['num_boost_round']}.json"
    )
    model_path = MODEL_DIR / model_name
    best_model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    if calibrator is not None:
        calibration_path = MODEL_DIR / f"{model_path.stem}_calibration.pkl"
        joblib.dump(calibrator, calibration_path)
        print(f"Saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
