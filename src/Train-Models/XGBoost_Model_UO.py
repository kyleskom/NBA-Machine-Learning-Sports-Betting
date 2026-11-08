import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))
from src.Utils import ModelFeatures  # noqa: E402
from src.Predict.model_selection import record_candidate  # noqa: E402
from src.Evaluation.experiment import (  # noqa: E402
    chronological_cv_indices, chronological_holdout_indices, classification_diagnostics,
    dataset_fingerprint, moneyline_market_and_backtest, write_experiment_report,
)

DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

DEFAULT_DATASET = "dataset_2012-26_team"
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
DEFAULT_FINAL_START = "2025-10-01"


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def prepare_data(df):
    data = ModelFeatures.team_only_frame(df).copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN, kind="stable")
    y = data[TARGET_COLUMN].astype(int).to_numpy()
    feature_frame = data.drop(columns=DROP_COLUMNS, errors="ignore")
    feature_columns = list(feature_frame.columns)
    X = feature_frame.astype(float).to_numpy()
    return X, y, feature_columns


def split_train_test(X, y, dates, test_size=0.1, final_start=None):
    train_idx, test_idx, split = chronological_holdout_indices(
        dates, final_fraction=test_size, final_start=final_start
    )
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], train_idx, test_idx, split


def split_train_calib(X, y, dates, calib_size=0.1):
    fit_idx, calib_idx, _ = chronological_holdout_indices(dates, final_fraction=calib_size)
    return X[fit_idx], y[fit_idx], X[calib_idx], y[calib_idx], fit_idx, calib_idx



def compute_sample_weights(y, num_classes, push_weight=0.25, dates=None, recency_half_life_days=0):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {
        cls: (total / (num_classes * count)) if count else 1.0
        for cls, count in enumerate(counts)
    }
    # A push is only about 1% of historical totals.  It should remain a valid
    # output class without dominating the over/under betting objective.
    if num_classes == 3:
        class_weights[2] = push_weight
    weights = np.array([class_weights[label] for label in y])
    if dates is not None and recency_half_life_days and recency_half_life_days > 0:
        days_old = (pd.Timestamp(dates.max()) - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
        weights = weights * np.power(0.5, days_old / recency_half_life_days)
    return weights


def sample_params(rng, seed, wide=False):
    eta_range = (0.002, 0.20) if wide else (0.015, 0.04)
    eta = 10 ** rng.uniform(np.log10(eta_range[0]), np.log10(eta_range[1]))
    ranges = {
        "depth": (2, 11) if wide else (3, 6),
        "subsample": (0.40, 1.0) if wide else (0.70, 0.85),
        "tree": (0.35, 1.0) if wide else (0.60, 0.75),
        "level": (0.35, 1.0) if wide else (0.60, 0.80),
        "node": (0.35, 1.0) if wide else (0.60, 0.80),
        "child": (1, 31) if wide else (10, 19),
        "gamma": (0.0, 15.0) if wide else (2.0, 5.0),
        "delta": (0, 13) if wide else (2, 6),
        "bin": (128, 1025) if wide else (256, 769),
        "lambda": (0.03, 20.0) if wide else (1.0, 4.0),
        "alpha": (0.001, 10.0) if wide else (0.2, 1.0),
        "rounds": (200, 3001) if wide else (500, 1201),
    }
    params = {
        "max_depth": int(rng.integers(*ranges["depth"])),
        "eta": float(eta),
        "subsample": float(rng.uniform(*ranges["subsample"])),
        "colsample_bytree": float(rng.uniform(*ranges["tree"])),
        "colsample_bylevel": float(rng.uniform(*ranges["level"])),
        "colsample_bynode": float(rng.uniform(*ranges["node"])),
        "min_child_weight": int(rng.integers(*ranges["child"])),
        "gamma": float(rng.uniform(*ranges["gamma"])),
        "max_delta_step": int(rng.integers(*ranges["delta"])),
        "max_bin": int(rng.integers(*ranges["bin"])),
        "lambda": float(10 ** rng.uniform(np.log10(ranges["lambda"][0]), np.log10(ranges["lambda"][1]))),
        "alpha": float(10 ** rng.uniform(np.log10(ranges["alpha"][0]), np.log10(ranges["alpha"][1]))),
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": "mlogloss",
        "seed": seed,
        "tree_method": "hist",
    }
    num_boost_round = int(rng.integers(*ranges["rounds"]))
    return params, num_boost_round


def fixed_params(raw, seed):
    source = Path(raw)
    overrides = json.loads(source.read_text() if source.exists() else raw)
    if not isinstance(overrides, dict):
        raise ValueError("--fixed-params must be a JSON object or a path to one")
    rounds = int(overrides.pop("num_boost_round", 800))
    allowed = set(sample_params(np.random.default_rng(seed), seed)[0])
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown fixed XGBoost parameters: {sorted(unknown)}")
    params, _ = sample_params(np.random.default_rng(seed), seed)
    params.update(overrides)
    params["seed"] = seed
    return params, rounds


def train_model(X_train, y_train, train_dates, X_val, y_val, params, num_boost_round, push_weight=0.25, recency_half_life_days=0):
    dtrain = xgb.DMatrix(X_train, label=y_train,
                         weight=compute_sample_weights(y_train, num_classes=NUM_CLASSES, push_weight=push_weight,
                                      dates=train_dates, recency_half_life_days=recency_half_life_days))
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    return model[:model.best_iteration + 1]


def format_param(value, precision=3):
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")


from src.Utils.Calibration import BoosterWrapper


def walk_forward_cv_loss(X, y, dates, params, num_boost_round, n_splits, push_weight, recency_half_life_days):
    losses = []
    for train_idx, val_idx in chronological_cv_indices(dates, n_splits):
        model = train_model(X[train_idx], y[train_idx], dates.iloc[train_idx], X[val_idx], y[val_idx],
                            params, num_boost_round, push_weight, recency_half_life_days)
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
    parser.add_argument("--fixed-params", help="JSON object or JSON file with one exact parameter set.")
    parser.add_argument("--final-start", default=DEFAULT_FINAL_START, help="First game date in the sealed final test period (YYYY-MM-DD).")
    parser.add_argument("--final-fraction", type=float, default=0.10, help="Newest whole game dates held for final testing when --final-start is omitted.")
    parser.add_argument("--push-weight", type=float, default=0.25,
                        help="Training weight for the rare push class (default: 0.25).")
    parser.add_argument("--recency-half-life-days", type=float, default=730,
                        help="Age in days at which a training game's weight halves; use 0 to disable.")
    parser.add_argument("--calibration-fraction", type=float, default=0.10,
                        help="Latest pre-final fraction reserved for probability calibration.")
    parser.add_argument("--wide-search", action="store_true",
                        help="Search a broad regularization and tree-complexity range.")
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

    df = df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
    X, y, feature_columns = prepare_data(df)
    dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce").reset_index(drop=True)
    fingerprint = dataset_fingerprint(df, args.dataset, TARGET_COLUMN, feature_columns)
    X_train_val, y_train_val, X_test, y_test, train_val_idx, test_idx, final_split = split_train_test(
        X, y, dates, test_size=args.final_fraction, final_start=args.final_start
    )
    train_val_dates = dates.iloc[train_val_idx].reset_index(drop=True)
    final_frame = df.iloc[test_idx].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    best = {
        "val_loss": float("inf"),
        "params": None,
        "num_boost_round": None,
    }

    trials = 1 if args.fixed_params else args.trials
    for trial in range(1, trials + 1):
        params, num_boost_round = (fixed_params(args.fixed_params, args.seed + trial)
                                   if args.fixed_params else sample_params(rng, seed=args.seed + trial,
                                                                            wide=args.wide_search))
        val_loss = walk_forward_cv_loss(X_train_val, y_train_val, train_val_dates, params, num_boost_round, args.splits,
                                         args.push_weight, args.recency_half_life_days)
        if val_loss is None:
            continue

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            best["num_boost_round"] = num_boost_round

        print(f"Trial {trial}/{trials}: val log loss {val_loss:.4f}")

    if best["params"] is None:
        print("No valid parameter set found.")
        return

    X_train, y_train, X_calib, y_calib, train_idx, calib_idx = split_train_calib(
        X_train_val, y_train_val, train_val_dates, args.calibration_fraction
    )
    train_dates = train_val_dates.iloc[train_idx].reset_index(drop=True)
    X_fit, y_fit, X_stop, y_stop, fit_idx, _ = split_train_calib(X_train, y_train, train_dates, 0.10)
    fit_dates = train_dates.iloc[fit_idx].reset_index(drop=True)
    best_model = train_model(
        X_fit, y_fit, fit_dates, X_stop, y_stop, best["params"], best["num_boost_round"],
        args.push_weight, args.recency_half_life_days,
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

    diagnostics = classification_diagnostics(y_test, probabilities, range(NUM_CLASSES))
    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final log loss: {test_loss:.4f}")
    if "brier_score" in diagnostics:
        print(f"Final Brier score: {diagnostics['brier_score']:.4f}")
        print(f"Final calibration error: {diagnostics['expected_calibration_error']:.4f}")

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
    feature_path = ModelFeatures.save_feature_columns(model_path, feature_columns)
    print(f"Saved feature columns: {feature_path}")

    if calibrator is not None:
        calibration_path = MODEL_DIR / f"{model_path.stem}_calibration.pkl"
        joblib.dump(calibrator, calibration_path)
        print(f"Saved calibration: {calibration_path}")
    report = {
        "kind": "UO", "dataset": fingerprint, "final_split": final_split,
        "calibration": args.calibration, "selection_cv_log_loss": best["val_loss"],
        "training": {"recency_half_life_days": args.recency_half_life_days,
                     "calibration_fraction": args.calibration_fraction, "push_weight": args.push_weight},
        "final_diagnostics": diagnostics,
        "market_and_backtest": {"available": False, "reason": "Totals payout prices and quote timestamps are not stored yet."},
        "sealed_final_note": "Do not use this final period to tune future candidates. Change it only when opening a new evaluation cycle.",
    }
    report_path = write_experiment_report(MODEL_DIR / f"{model_path.stem}_experiment.json", report)
    print(f"Saved experiment report: {report_path}")
    promoted, incumbent = record_candidate("UO", model_path, args.dataset, accuracy, test_loss,
                                            experiment=report)
    if promoted:
        print("Promoted as the selected UO model.")
    else:
        print(f"Not promoted: selected UO model test log loss is {incumbent['test_log_loss']:.4f}.")


if __name__ == "__main__":
    main()
