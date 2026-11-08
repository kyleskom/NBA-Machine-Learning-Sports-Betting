"""Chronological experiment splits, diagnostics, and moneyline backtests.

These helpers deliberately operate on calendar dates rather than row positions so all
matches from a slate stay in the same partition.  A saved report is an audit record,
not a claim that a model is profitable.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


BASE_DIR = Path(__file__).resolve().parents[2]
CANONICAL_DB = BASE_DIR / "Data" / "NBAData.sqlite"


def _normal_dates(values):
    dates = pd.Series(pd.to_datetime(values, errors="coerce")).dt.normalize()
    if dates.isna().any():
        raise ValueError("Every experiment row must have a valid Date.")
    return dates


def chronological_holdout_indices(dates, final_fraction=0.10, final_start=None):
    """Return disjoint earlier/final indices without splitting a game date.

    `final_start` is preferred for a permanently sealed period.  With no explicit
    boundary, the newest whole dates comprising at least `final_fraction` are held
    out.  The returned metadata makes the boundary reviewable in saved reports.
    """
    if not 0 < final_fraction < 1:
        raise ValueError("final_fraction must be between 0 and 1.")
    normalized = _normal_dates(dates)
    unique_dates = np.sort(normalized.unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct game dates are required.")
    if final_start:
        boundary = pd.Timestamp(final_start).normalize()
        final_mask = normalized >= boundary
        if not final_mask.any() or final_mask.all():
            raise ValueError("--final-start must leave rows on both sides of the boundary.")
    else:
        target_rows = math.ceil(len(normalized) * final_fraction)
        counts = normalized.value_counts().sort_index(ascending=False)
        running = 0
        boundary = unique_dates[-1]
        for day, count in counts.items():
            running += int(count)
            boundary = day
            if running >= target_rows:
                break
        final_mask = normalized >= boundary
    train_idx = np.flatnonzero(~final_mask.to_numpy())
    final_idx = np.flatnonzero(final_mask.to_numpy())
    return train_idx, final_idx, {
        "method": "fixed_date" if final_start else "latest_whole_dates_fraction",
        "final_start": pd.Timestamp(boundary).date().isoformat(),
        "train_start": pd.Timestamp(normalized.iloc[train_idx].min()).date().isoformat(),
        "train_end": pd.Timestamp(normalized.iloc[train_idx].max()).date().isoformat(),
        "final_end": pd.Timestamp(normalized.iloc[final_idx].max()).date().isoformat(),
        "train_rows": int(len(train_idx)),
        "final_rows": int(len(final_idx)),
        "train_dates": int(normalized.iloc[train_idx].nunique()),
        "final_dates": int(normalized.iloc[final_idx].nunique()),
    }


def dataset_fingerprint(frame, dataset_name, target_column, feature_columns, date_column="Date"):
    """Stable SHA-256 identity for data/schema used by an experiment.

    The digest intentionally includes every cell in the model features and target.
    It changes when the data, ordering, or feature contract changes.
    """
    fields = [date_column, target_column, *feature_columns]
    available = [field for field in fields if field in frame.columns]
    payload = {
        "dataset": dataset_name,
        "rows": int(len(frame)),
        "columns": available,
        "data": frame.loc[:, available].replace({np.nan: None}).to_dict(orient="split"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    dates = _normal_dates(frame[date_column])
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dataset": dataset_name,
        "rows": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
    }


def reliability_table(y_true, probabilities, bins=10):
    """Return a calibration table for a binary home-win probability."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError("Targets and probabilities must have equal length.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (predicted >= low) & ((predicted < high) if index < bins - 1 else (predicted <= high))
        count = int(mask.sum())
        if count:
            rows.append({
                "lower": round(float(low), 4), "upper": round(float(high), 4), "count": count,
                "mean_prediction": round(float(predicted[mask].mean()), 6),
                "observed_rate": round(float(actual[mask].mean()), 6),
            })
    return rows


def expected_calibration_error(y_true, probabilities, bins=10):
    rows = reliability_table(y_true, probabilities, bins=bins)
    total = max(1, len(y_true))
    return float(sum(row["count"] / total * abs(row["mean_prediction"] - row["observed_rate"])
                     for row in rows))


def classification_diagnostics(y_true, probabilities, classes):
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.argmax(probabilities, axis=1)
    result = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=list(classes))),
    }
    if len(classes) == 2:
        home_probability = probabilities[:, 1]
        result.update({
            "brier_score": float(brier_score_loss(y_true, home_probability)),
            "expected_calibration_error": expected_calibration_error(y_true, home_probability),
            "reliability": reliability_table(y_true, home_probability),
        })
    return result


def american_to_implied(price):
    if price is None or not np.isfinite(price) or price == 0:
        return None
    price = float(price)
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def no_vig_home_probability(home_price, away_price):
    home = american_to_implied(home_price)
    away = american_to_implied(away_price)
    if home is None or away is None or home + away <= 0:
        return None
    return home / (home + away)


def _moneyline_quotes(sportsbook):
    if not CANONICAL_DB.exists():
        return pd.DataFrame(columns=["Date", "Home", "Away", "ML_Home", "ML_Away"])
    query = """SELECT g.game_date AS Date, g.home_team_name AS Home, g.away_team_name AS Away,
        MAX(CASE WHEN q.side='home' THEN q.price END) AS ML_Home,
        MAX(CASE WHEN q.side='away' THEN q.price END) AS ML_Away
        FROM games g JOIN odds_quotes q ON q.game_id=g.game_id
        WHERE q.sportsbook=? AND q.market='moneyline'
        GROUP BY g.game_id"""
    with sqlite3.connect(CANONICAL_DB) as con:
        quotes = pd.read_sql_query(query, con, params=[sportsbook])
    quotes["Date"] = pd.to_datetime(quotes["Date"], errors="coerce").dt.normalize()
    return quotes


def attach_moneyline_market(frame, sportsbook="fanduel"):
    """Join the sole stored quote per game; quote timing remains explicitly unknown."""
    required = {"Date", "TEAM_NAME", "TEAM_NAME.1"}
    if not required.issubset(frame.columns):
        return frame.copy(), {"available": False, "reason": "game identity columns are unavailable"}
    games = frame.copy().reset_index(drop=True)
    games = games.drop(columns=["ML_Home", "ML_Away"], errors="ignore")
    games["Date"] = _normal_dates(games["Date"])
    games = games.rename(columns={"TEAM_NAME": "Home", "TEAM_NAME.1": "Away"})
    games["_quote_order"] = np.arange(len(games))
    merged = games.merge(_moneyline_quotes(sportsbook), on=["Date", "Home", "Away"],
                         how="left", validate="many_to_one", sort=False)
    merged = merged.sort_values("_quote_order").drop(columns="_quote_order").reset_index(drop=True)
    matched = int(merged["ML_Home"].notna().sum() if "ML_Home" in merged else 0)
    return merged, {
        "available": matched > 0,
        "sportsbook": sportsbook,
        "matched_rows": matched,
        "total_rows": int(len(frame)),
        "quote_timing": "unknown (the current schema does not retain opening/closing timestamps)",
    }


def moneyline_ledger(frame, y_true, home_probabilities, min_ev=0.02, min_probability=0.0):
    """Settle at most one fixed-unit wager per game, considering both offered sides.

    The frame must already contain aligned ML_Home/ML_Away quotes. PASS rows are
    retained for auditing coverage. Thresholds are fixed inputs, never optimized here.
    """
    outcomes = np.asarray(y_true)
    probabilities = np.asarray(home_probabilities, dtype=float)
    if outcomes.ndim != 1 or probabilities.ndim != 1 or not len(frame) == len(outcomes) == len(probabilities):
        raise ValueError("Games, outcomes, and probabilities must be aligned one-dimensional arrays.")
    if not np.isin(outcomes, [0, 1]).all():
        raise ValueError("Moneyline outcomes must be binary.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not np.isfinite(min_ev) or min_ev < 0 or not 0 <= min_probability <= 1:
        raise ValueError("Use a nonnegative finite EV threshold and probability threshold in [0, 1].")
    if "Home-Team-Win" in frame and not np.array_equal(frame["Home-Team-Win"].to_numpy(), outcomes):
        raise ValueError("Game identities and outcome arrays are misaligned.")
    records = []
    for position, row in enumerate(frame.to_dict("records")):
        probability, truth = probabilities[position], int(outcomes[position])
        candidates = []
        for side, side_probability, column in [(1, probability, "ML_Home"), (0, 1 - probability, "ML_Away")]:
            price = row.get(column)
            if price is None or not np.isfinite(price) or abs(price) < 100:
                continue
            payout = 100.0 / -price if price < 0 else price / 100.0
            ev = side_probability * payout - (1 - side_probability)
            if side_probability >= min_probability and ev >= min_ev:
                candidates.append((ev, side, price, payout, side_probability))
        record = {key: row[key] for key in ("game_id", "Date", "Home", "Away") if key in row}
        record.update(home_probability=float(probability), outcome=truth,
                      ML_Home=row.get("ML_Home"), ML_Away=row.get("ML_Away"),
                      status="PASS", side=None, price=None, expected_value=None,
                      stake_units=0.0, profit_units=0.0)
        if candidates:
            ev, side, price, payout, side_probability = max(candidates)
            record.update(status="BET", side="home" if side else "away", price=float(price),
                          expected_value=float(ev), stake_units=1.0,
                          profit_units=float(payout if truth == side else -1.0))
        records.append(record)
    return pd.DataFrame(records)


def summarize_moneyline_ledger(ledger):
    bets = ledger.loc[ledger.stake_units > 0]
    profits = bets.profit_units.to_numpy(dtype=float)
    equity = np.r_[0.0, np.cumsum(profits)]
    drawdowns = equity - np.maximum.accumulate(equity)
    return {
        "available": bool(len(bets)), "bets": int(len(bets)),
        "stake_units": float(len(bets)), "profit_units": float(profits.sum()),
        "roi": float(profits.mean()) if len(bets) else None,
        "win_rate": float((profits > 0).mean()) if len(bets) else None,
        "max_drawdown_units": float(drawdowns.min()),
    }


def moneyline_market_and_backtest(frame, y_true, home_probabilities, sportsbook="fanduel",
                                  min_ev=0.02, min_probability=0.50):
    """Compare to de-vig market forecasts and simulate fixed one-unit EV bets.

    It uses only the archived quote; absence of quote timestamps is reported, never
    treated as an executable closing price or used for closing-line value.
    """
    merged, info = attach_moneyline_market(frame, sportsbook)
    if not info["available"]:
        return {"market": info, "backtest": {"available": False, "reason": info["reason"]}}
    probs = np.asarray(home_probabilities, dtype=float)
    outcomes = np.asarray(y_true, dtype=int)
    market_prob = np.array([no_vig_home_probability(h, a) for h, a in zip(merged.ML_Home, merged.ML_Away)], dtype=float)
    valid = np.isfinite(market_prob)
    result = {"market": dict(info)}
    result["market"].update({
        "evaluated_rows": int(valid.sum()),
        "no_vig_log_loss": float(log_loss(outcomes[valid], np.column_stack((1 - market_prob[valid], market_prob[valid])), labels=[0, 1])) if valid.any() else None,
        "no_vig_brier_score": float(brier_score_loss(outcomes[valid], market_prob[valid])) if valid.any() else None,
    })
    ledger = moneyline_ledger(merged, outcomes, probs, min_ev, min_probability)
    result["backtest"] = {
        **summarize_moneyline_ledger(ledger), "min_expected_value": min_ev,
        "min_probability": min_probability,
        "closing_line_value": None,
        "limitation": "Closing-line value is unavailable until odds_quotes retains timestamped quote history.",
    }
    return result


def select_moneyline_policy(frame, y_true, home_probabilities, sportsbook="fanduel",
                            ev_thresholds=(0.02,), probability_thresholds=(0.50,), min_bets=50):
    """Pick an EV/probability rule on a pre-final policy window.

    Selection requires a meaningful sample of archived bets and ranks candidates by
    total unit profit, then ROI, then the smaller drawdown. The returned rule must
    subsequently be evaluated once on the separate final period.
    """
    candidates = []
    for min_ev in ev_thresholds:
        for min_probability in probability_thresholds:
            result = moneyline_market_and_backtest(
                frame, y_true, home_probabilities, sportsbook, min_ev, min_probability
            )
            backtest = result["backtest"]
            if backtest.get("available") and backtest["stake_units"] >= min_bets:
                candidates.append({
                    "min_expected_value": float(min_ev),
                    "min_probability": float(min_probability),
                    "profit_units": backtest["profit_units"], "roi": backtest["roi"],
                    "max_drawdown_units": backtest["max_drawdown_units"],
                    "bets": backtest["stake_units"],
                })
    if not candidates:
        return {
            "available": False,
            "reason": f"No policy produced at least {min_bets} archived bets.",
            "min_bets": min_bets,
        }
    selected = max(candidates, key=lambda row: (row["profit_units"], row["roi"], row["max_drawdown_units"]))
    return {"available": True, "selected": selected, "candidates": candidates, "min_bets": min_bets}


def write_experiment_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def chronological_cv_indices(dates, n_splits):
    """Expanding-window folds that keep each calendar date intact."""
    normalized = _normal_dates(dates)
    unique_dates = np.sort(normalized.unique())
    if len(unique_dates) <= n_splits:
        raise ValueError("Not enough distinct game dates for the requested CV splits.")
    blocks = np.array_split(unique_dates, n_splits + 1)
    folds = []
    for index in range(1, len(blocks)):
        train_days = np.concatenate(blocks[:index])
        validation_days = blocks[index]
        train_idx = np.flatnonzero(normalized.isin(train_days).to_numpy())
        validation_idx = np.flatnonzero(normalized.isin(validation_days).to_numpy())
        if len(train_idx) and len(validation_idx):
            folds.append((train_idx, validation_idx))
    return folds
