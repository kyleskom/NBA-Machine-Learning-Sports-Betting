import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"

DEFAULT_DATASET = "dataset_2012-26_new"
TARGET_COLUMN = "Home-Team-Win"
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
    "OU",
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


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression moneyline model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_time_series(X, y)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {loss:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
