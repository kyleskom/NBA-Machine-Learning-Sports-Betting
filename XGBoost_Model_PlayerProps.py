"""
Обучение модели для предсказания player props (over/under).
Использует XGBoost для классификации.
"""

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "player_props_dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "PlayerProps_Models"

# Колонки, которые не используются для обучения
DROP_COLUMNS = [
    "player_id",
    "player_name",
    "game_date",
    "actual_points",
    "target",
    "line_diff",
]


def load_dataset():
    """Загрузить датасет из базы"""
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query("SELECT * FROM player_props_training", con)
    
    if df.empty:
        raise ValueError("Датасет пуст. Сначала запустите Create_Player_Props_Dataset.py")
    
    # Сортируем по дате для time series split
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date")
    
    return df


def prepare_data(df):
    """Подготовить данные для обучения"""
    y = df["target"].astype(int).to_numpy()
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    
    # Заполняем пропуски
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    
    return X.astype(float).to_numpy(), y


def split_train_test(X, y, test_size=0.15):
    """Разделить на train/test по времени"""
    n = len(X)
    if n == 0:
        raise ValueError("Пустой датасет")
    
    test_start = int(n * (1 - test_size))
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1):
    """Разделить train на train/calibration"""
    n = len(X)
    calib_start = int(n * (1 - calib_size))
    return X[:calib_start], y[:calib_start], X[calib_start:], y[calib_start:]


def compute_sample_weights(y):
    """Рассчитать веса для балансировки классов"""
    counts = np.bincount(y)
    total = len(y)
    class_weights = {
        cls: (total / (2 * count)) if count else 1.0 for cls, count in enumerate(counts)
    }
    return np.array([class_weights[label] for label in y])


def sample_params(rng, seed):
    """Сэмплировать гиперпараметры для XGBoost"""
    eta = 10 ** rng.uniform(np.log10(0.005), np.log10(0.3))
    
    params = {
        "max_depth": int(rng.integers(3, 10)),
        "eta": float(eta),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
        "min_child_weight": int(rng.integers(1, 15)),
        "gamma": float(rng.uniform(0.0, 5.0)),
        "lambda": float(10 ** rng.uniform(np.log10(0.5), np.log10(5.0))),
        "alpha": float(10 ** rng.uniform(np.log10(0.01), np.log10(2.0))),
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "seed": seed,
        "tree_method": "hist",
    }
    
    num_boost_round = int(rng.integers(200, 1501))
    
    return params, num_boost_round


def train_model(X_train, y_train, X_val, y_val, params, num_boost_round):
    """Обучить одну модель XGBoost"""
    dtrain = xgb.DMatrix(
        X_train, label=y_train, weight=compute_sample_weights(y_train)
    )
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    
    return model


class BoosterWrapper:
    """Обёртка для sklearn calibration"""
    
    def __init__(self, booster):
        self.booster = booster
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        preds = self.booster.predict(xgb.DMatrix(X))
        return np.column_stack([1 - preds, preds])


def walk_forward_cv(X, y, params, num_boost_round, n_splits=5):
    """Walk-forward cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    
    for train_idx, val_idx in tscv.split(X):
        model = train_model(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], params, num_boost_round
        )
        
        val_preds = model.predict(xgb.DMatrix(X[val_idx]))
        loss = log_loss(y[val_idx], val_preds)
        losses.append(loss)
    
    return float(np.mean(losses)) if losses else None


def format_param(value, precision=3):
    """Форматировать параметр для имени файла"""
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Обучение модели player props")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trials", type=int, default=50, help="Число попыток")
    parser.add_argument("--splits", type=int, default=5, help="Число фолдов CV")
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        choices=["sigmoid", "isotonic", "none"],
        help="Метод калибровки вероятностей",
    )
    
    args = parser.parse_args()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные
    print("Загрузка датасета...")
    df = load_dataset()
    print(f"✓ Загружено {len(df)} записей")
    
    X, y = prepare_data(df)
    X_train_val, y_train_val, X_test, y_test = split_train_test(X, y)
    
    print(f"Train/Val: {len(X_train_val)} | Test: {len(X_test)}")
    print(f"Over rate: {y.mean()*100:.1f}%")
    
    # Поиск лучших гиперпараметров
    print(f"\nПоиск оптимальных параметров ({args.trials} попыток)...")
    
    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None, "num_boost_round": None}
    
    for trial in range(1, args.trials + 1):
        params, num_boost_round = sample_params(rng, seed=args.seed + trial)
        val_loss = walk_forward_cv(
            X_train_val, y_train_val, params, num_boost_round, args.splits
        )
        
        if val_loss is None:
            continue
        
        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            best["num_boost_round"] = num_boost_round
        
        print(f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f}")
    
    if best["params"] is None:
        print("Не удалось найти подходящие параметры")
        return
    
    # Обучаем финальную модель
    print("\nОбучение финальной модели...")
    X_train, y_train, X_calib, y_calib = split_train_calib(X_train_val, y_train_val)
    
    best_model = train_model(
        X_train, y_train, X_calib, y_calib, best["params"], best["num_boost_round"]
    )
    
    # Калибровка
    calibrator = None
    if args.calibration == "none":
        test_preds = best_model.predict(xgb.DMatrix(X_test))
        probabilities = np.column_stack([1 - test_preds, test_preds])
    else:
        print(f"Калибровка ({args.calibration})...")
        calibrator = CalibratedClassifierCV(
            BoosterWrapper(best_model), method=args.calibration, cv="prefit"
        )
        calibrator.fit(X_calib, y_calib)
        probabilities = calibrator.predict_proba(X_test)
    
    # Метрики
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    
    print(f"\n{'='*50}")
    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test log loss: {test_loss:.4f}")
    print(f"Test AUC-ROC: {auc:.4f}")
    print(f"{'='*50}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Under", "Over"]))
    
    # Сохраняем модель
    params_str = best["params"]
    model_name = (
        f"PlayerProps_{accuracy * 100:.1f}%"
        f"_md{params_str['max_depth']}"
        f"_eta{format_param(params_str['eta'])}"
        f"_sub{format_param(params_str['subsample'])}"
        f"_nb{best['num_boost_round']}.json"
    )
    
    model_path = MODEL_DIR / model_name
    best_model.save_model(str(model_path))
    print(f"\n✓ Модель сохранена: {model_path}")
    
    if calibrator is not None:
        calib_path = MODEL_DIR / f"{model_path.stem}_calibration.pkl"
        joblib.dump(calibrator, calib_path)
        print(f"✓ Калибратор сохранён: {calib_path}")
    
    # Важность фич
    print("\n=== Топ-15 важных фич ===")
    feature_importance = best_model.get_score(importance_type="gain")
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    
    for i, (feature, importance) in enumerate(sorted_features[:15], 1):
        print(f"{i:2d}. {feature:25s}: {importance:8.2f}")


if __name__ == "__main__":
    main()
