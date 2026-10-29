"""Frozen moneyline benchmarks: python -m src.Evaluation.backtest --help.

Runs are retrospective diagnostics. Archived odds lack timestamps; the evaluation
period is a development benchmark, not a sealed final test or an execution replay.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sqlite3
import platform
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .experiment import (
    BASE_DIR, attach_moneyline_market, chronological_holdout_indices,
    classification_diagnostics, moneyline_ledger, no_vig_home_probability,
    summarize_moneyline_ledger, write_experiment_report,
)

KEYS = ['Date', 'TEAM_NAME', 'TEAM_NAME.1']
TARGET = 'Home-Team-Win'
EXCLUDE = set(KEYS + [TARGET, 'game_id', 'index', 'index.1', 'Date.1', 'Score',
                      'OU-Cover', 'OU', 'Win_Margin', 'Points', 'home_score', 'away_score',
                      'ML_Home', 'ML_Away', 'Spread'])
LIMITATION = ('Retrospective development benchmark. Archived quote timing is unknown; '
              'ROI is not evidence of executable returns. Repeated comparisons use this '
              'period for development; confirm a frozen strategy on future unseen games.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     default=str).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_fingerprint(protocol, frame, quotes):
    identity = frame[['game_id', 'Date', TARGET]].reset_index(drop=True).to_json(orient='split')
    prices = quotes[['game_id', 'ML_Home', 'ML_Away']].astype({'ML_Home': float, 'ML_Away': float})
    prices = prices.sort_values('game_id').reset_index(drop=True).to_json(orient='split', double_precision=15)
    return digest([protocol, identity, prices])


def canonical_games(frame):
    """Return one deterministic row per natural game key, preserving X/y identity."""
    frame = frame.copy()
    missing = set(KEYS + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing game identity/outcome columns: {sorted(missing)}')
    if frame[KEYS + [TARGET]].isna().any().any():
        raise ValueError('Game identity and outcome values cannot be missing.')
    frame['Date'] = pd.to_datetime(frame.Date, errors='raise').dt.strftime('%Y-%m-%d')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate game identities are not allowed.')
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError('Settled binary home-win outcomes are required.')
    frame[TARGET] = frame[TARGET].astype(int)
    frame['game_id'] = [digest(list(values)) for values in frame[KEYS].itertuples(index=False, name=None)]
    return frame.sort_values(KEYS, kind='stable').reset_index(drop=True)


def load_dataset(path, table):
    path = Path(path).resolve()
    with sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            raise ValueError(f'Dataset table not found: {table}')
        escaped = table.replace('"', '""')
        return canonical_games(pd.read_sql_query(f'SELECT * FROM "{escaped}"', con))


def align_candidate(frame, reference):
    """Reject changed coverage or outcomes; allow genuinely new/changed features."""
    frame = canonical_games(frame)
    if set(frame.game_id) != set(reference.game_id):
        raise ValueError('Candidate game coverage differs from the baseline. Start a new baseline for different games.')
    frame = frame.set_index('game_id').loc[reference.game_id].reset_index()
    if not np.array_equal(frame[TARGET].to_numpy(), reference[TARGET].to_numpy()):
        raise ValueError('Candidate outcomes differ from the baseline.')
    return frame


def feature_columns(frame, drops):
    columns = [c for c in frame if c not in EXCLUDE]
    for pattern in drops:
        matches = [c for c in columns if fnmatch.fnmatchcase(c, pattern)]
        if not matches:
            raise ValueError(f'Feature pattern matched no remaining columns: {pattern}')
        columns = [c for c in columns if c not in matches]
    if not columns:
        raise ValueError('No model features remain.')
    return columns


def fit_predict(frame, protocol, model_kind, drops=(), params=None):
    """Fit only before the benchmark boundary; optional sigmoid tail calibration."""
    train_mask = frame.Date < protocol['start']
    train, test = frame.loc[train_mask], frame.loc[~train_mask]
    columns = feature_columns(frame, drops)
    numeric = frame[columns].apply(pd.to_numeric, errors='raise').astype(float).replace([np.inf, -np.inf], np.nan)
    X_train, X_test = numeric.loc[train_mask], numeric.loc[~train_mask]
    if X_train.isna().all().any():
        raise ValueError('A feature is entirely missing in training; explicitly drop it.')
    if train[TARGET].nunique() < 2:
        raise ValueError('Training requires both outcome classes.')
    params = params or {}
    if model_kind == 'logistic':
        settings = dict(C=0.1, max_iter=2000, random_state=protocol['seed'])
        settings.update(params)
        estimator = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                  LogisticRegression(**settings))
    else:
        from xgboost import XGBClassifier
        settings = dict(n_estimators=250, max_depth=3, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                        reg_lambda=5, objective='binary:logistic', eval_metric='logloss',
                        random_state=protocol['seed'], n_jobs=2)
        settings.update(params)
        if settings.get('objective') != 'binary:logistic':
            raise ValueError('The moneyline benchmark requires binary:logistic.')
        estimator = XGBClassifier(**settings)
    calibration = protocol['calibration_fraction']
    if calibration:
        fit_idx, calib_idx, _ = chronological_holdout_indices(train.Date, final_fraction=calibration)
        if train.iloc[fit_idx][TARGET].nunique() < 2 or train.iloc[calib_idx][TARGET].nunique() < 2:
            raise ValueError('Fit and calibration date blocks must each contain both classes.')
        estimator.fit(X_train.iloc[fit_idx], train.iloc[fit_idx][TARGET])
        estimator = CalibratedClassifierCV(estimator, method='sigmoid', cv='prefit')
        estimator.fit(X_train.iloc[calib_idx], train.iloc[calib_idx][TARGET])
    else:
        estimator.fit(X_train, train[TARGET])
    probabilities = estimator.predict_proba(X_test)[:, 1]
    metadata = dict(kind=model_kind, parameters=settings, features=columns,
                    feature_data_sha256=digest(numeric.to_json(orient='split', double_precision=15)),
                    train_rows=len(train), test_rows=len(test),
                    calibration_fraction=calibration, calibration='sigmoid' if calibration else 'none')
    return probabilities, metadata, estimator


def diagnostics(outcomes, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    return classification_diagnostics(outcomes, np.column_stack([1 - probabilities, probabilities]), [0, 1])


def evaluate(frame, quotes, probabilities, protocol):
    test = frame.loc[frame.Date >= protocol['start']].copy()
    # All outcomes, predictions and prices travel with immutable game keys.
    if len(probabilities) != len(test):
        raise ValueError('Exactly one probability per evaluation game is required.')
    test['home_probability'] = probabilities
    test = test.merge(quotes, on='game_id', how='left', validate='one_to_one', sort=False)
    test = test.sort_values(KEYS, kind='stable').reset_index(drop=True)
    probabilities = test.home_probability.to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Predictions must be finite probabilities in [0, 1].')
    outcomes = test[TARGET].to_numpy()
    market = np.array([no_vig_home_probability(h, a) if abs(h) >= 100 and abs(a) >= 100 else np.nan
                       for h, a in zip(test.ML_Home, test.ML_Away)], dtype=float)
    valid = np.isfinite(market)
    rate = float(frame.loc[frame.Date < protocol['start'], TARGET].mean())
    benchmarks = {'home_rate': diagnostics(outcomes, np.full(len(test), rate)),
                  'market': diagnostics(outcomes[valid], market[valid]) if valid.any() else None}
    policy = protocol['policy']
    ledger = moneyline_ledger(test.rename(columns={'TEAM_NAME': 'Home', 'TEAM_NAME.1': 'Away'}),
                             outcomes, probabilities, policy['min_ev'], policy['min_probability'])
    ledger['market_home_probability'] = market
    ledger['cumulative_profit_units'] = ledger.profit_units.cumsum()
    per_season = {}
    seasons = pd.to_datetime(ledger.Date).map(lambda d: str(d.year if d.month >= 7 else d.year - 1))
    for season in sorted(seasons.unique()):
        per_season[season] = summarize_moneyline_ledger(ledger.loc[seasons == season])
    summary = dict(diagnostics=diagnostics(outcomes, probabilities), benchmarks=benchmarks,
                   priced_rows=int(valid.sum()), unpriced_rows=int((~valid).sum()),
                   model_on_priced_games=diagnostics(outcomes[valid], probabilities[valid]) if valid.any() else None,
                   betting=summarize_moneyline_ledger(ledger), betting_by_season=per_season,
                   limitation=LIMITATION, closing_line_value=None)
    return summary, ledger


def load_baseline(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_digest(path / name) != expected:
            raise ValueError(f'Baseline file changed: {name}. Create a new baseline.')
    frame = canonical_games(pd.read_csv(path / 'snapshot.csv.gz', float_precision='round_trip'))
    quotes = pd.read_csv(path / 'quotes.csv', float_precision='round_trip')
    if protocol_fingerprint(manifest['protocol'], frame, quotes) != manifest['protocol_id']:
        raise ValueError('Baseline manifest does not match its frozen games/prices/protocol.')
    return frame, quotes, manifest


def compare_reports(baseline, candidate):
    if baseline['protocol_id'] != candidate['protocol_id']:
        raise ValueError('Reports use different games, prices, date splits, or policies.')
    pairs = [('log_loss', baseline['diagnostics']['log_loss'], candidate['diagnostics']['log_loss']),
             ('brier_score', baseline['diagnostics']['brier_score'], candidate['diagnostics']['brier_score']),
             ('accuracy', baseline['diagnostics']['accuracy'], candidate['diagnostics']['accuracy'])]
    pairs += [(key, baseline['betting'][key], candidate['betting'][key])
              for key in ['bets', 'profit_units', 'roi', 'max_drawdown_units']]
    return {key: {'baseline': old, 'candidate': new,
                  'delta': new - old if old is not None and new is not None else None}
            for key, old, new in pairs}


def save_run(output, frame, quotes, manifest, args):
    model = None
    if args.predictions:
        supplied = pd.read_csv(args.predictions, float_precision='round_trip')
        if not {'game_id', 'home_probability'}.issubset(supplied):
            raise ValueError('Prediction CSV requires game_id and home_probability.')
        expected = frame.loc[frame.Date >= manifest['protocol']['start'], ['game_id']]
        if supplied.game_id.duplicated().any() or set(supplied.game_id) != set(expected.game_id):
            raise ValueError('Prediction CSV must contain exactly the evaluation game IDs once each.')
        probabilities = expected.merge(supplied[['game_id', 'home_probability']], on='game_id',
                                       validate='one_to_one', sort=False).home_probability.to_numpy(dtype=float)
        model_info = {'kind': 'external_predictions', 'file_sha256': file_digest(args.predictions),
                      'training_provenance': 'External: caller must ensure training used only earlier data.'}
    else:
        params = json.loads(Path(args.params).read_text()) if args.params else {}
        if not isinstance(params, dict):
            raise ValueError('Model parameters must be a JSON object.')
        probabilities, model_info, model = fit_predict(frame, manifest['protocol'], args.model,
                                                       args.drop_feature, params)
    summary, ledger = evaluate(frame, quotes, probabilities, manifest['protocol'])
    report = dict(name=args.name, created_at=datetime.now(timezone.utc).isoformat(),
                  protocol_id=manifest['protocol_id'], protocol=manifest['protocol'],
                  inputs={'baseline': str(args.baseline.resolve()) if getattr(args, 'baseline', None) else None,
                          'refreshed_features': getattr(args, 'refresh_features', False),
                          'dataset': args.dataset, 'source_database': str(args.dataset_db.resolve())},
                  code_sha256={name: file_digest(Path(__file__).parent / name)
                               for name in ['backtest.py', 'experiment.py']},
                  software={package: version(package) for package in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
                  python=platform.python_version(),
                  model=model_info, **summary)
    if getattr(args, 'baseline', None):
        reference = json.loads((Path(args.baseline) / 'report.json').read_text())
        report['comparison'] = compare_reports(reference, report)
    # Never overwrite a prior run. All validation/fitting happens before creation.
    output.mkdir(parents=True, exist_ok=False)
    ledger.to_csv(output / 'predictions.csv', index=False, float_format='%.17g')
    if getattr(args, 'refresh_features', False):
        frame.to_csv(output / 'snapshot.csv.gz', index=False, float_format='%.17g')
    if model is not None:
        joblib.dump(model, output / 'model.joblib')
    write_experiment_report(output / 'report.json', report)
    lines = [f'# {args.name}', '', LIMITATION, '',
                 f"Games: {summary['diagnostics']['rows']}; priced: {summary['priced_rows']}", '',
                 f"Log loss: {summary['diagnostics']['log_loss']:.6f}",
                 f"Brier score: {summary['diagnostics']['brier_score']:.6f}",
                 f"Bets: {summary['betting']['bets']}; profit: {summary['betting']['profit_units']:.2f} units",
                 f"ROI (fraction): {summary['betting']['roi']}; maximum drawdown: {summary['betting']['max_drawdown_units']:.2f} units", '']
    lines += ['| Forecast | Games | Log loss | Brier score |', '|---|---:|---:|---:|']
    for label, result in [('Model', summary['diagnostics']), ('Home win rate', summary['benchmarks']['home_rate']),
                          ('Model, priced games', summary['model_on_priced_games']),
                          ('No-vig market', summary['benchmarks']['market'])]:
        if result:
            lines.append(f"| {label} | {result['rows']} | {result['log_loss']:.6f} | {result['brier_score']:.6f} |")
    if 'comparison' in report:
        lines += ['', '| Metric | Baseline | Candidate | Change |', '|---|---:|---:|---:|']
        for key, values in report['comparison'].items():
            def fmt(value):
                return 'n/a' if value is None else f'{value:.6f}'
            lines.append(f"| {key} | {fmt(values['baseline'])} | {fmt(values['candidate'])} | {fmt(values['delta'])} |")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['baseline', 'run']:
        p = sub.add_parser(command)
        p.add_argument('--name', required=True, help='Unique saved run name')
        p.add_argument('--output-root', type=Path, default=BASE_DIR / 'Data' / 'Backtests')
        p.add_argument('--model', choices=['logistic', 'xgboost'], default='logistic')
        p.add_argument('--params', type=Path, help='Fixed estimator parameter JSON file; no tuning on benchmark')
        p.add_argument('--drop-feature', action='append', default=[], help='Column or glob; repeat to ablate groups')
        p.add_argument('--predictions', type=Path, help='External game_id/home_probability CSV; training provenance unverified')
        p.add_argument('--dataset-db', type=Path, default=BASE_DIR / 'Data' / 'dataset.sqlite')
        p.add_argument('--dataset', default='dataset_2012-26_player_v2')
        if command == 'baseline':
            p.add_argument('--start', required=True, help='First evaluation date YYYY-MM-DD')
            p.add_argument('--end', help='Last evaluation date; defaults to latest available')
            p.add_argument('--sportsbook', default='fanduel')
            p.add_argument('--min-ev', type=float, default=0.02)
            p.add_argument('--min-probability', type=float, default=0.0, help='Default allows value underdogs')
            p.add_argument('--calibration-fraction', type=float, default=0.1)
            p.add_argument('--seed', type=int, default=42)
        else:
            p.add_argument('--baseline', required=True, type=Path)
            p.add_argument('--refresh-features', action='store_true', help='Read candidate dataset; require identical games/outcomes')
    compare = sub.add_parser('compare')
    compare.add_argument('baseline', type=Path)
    compare.add_argument('candidate', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'compare':
            reports = [json.loads((path / 'report.json').read_text()) for path in [args.baseline, args.candidate]]
            print(json.dumps(compare_reports(*reports), indent=2, allow_nan=False))
            return
        if Path(args.name).name != args.name or args.name in ['', '.', '..']:
            raise ValueError('Run name must be a single directory name.')
        output = args.output_root / args.name
        if output.exists():
            raise ValueError(f'Run already exists: {output}. Choose a new name.')
        if args.command == 'baseline':
            frame = load_dataset(args.dataset_db, args.dataset)
            start = pd.Timestamp(args.start).date().isoformat()
            end = pd.Timestamp(args.end).date().isoformat() if args.end else frame.Date.max()
            frame = frame.loc[frame.Date <= end].reset_index(drop=True)
            if not (frame.Date < start).any() or not (frame.Date >= start).any():
                raise ValueError('Date range must leave training and evaluation games.')
            if not 0 <= args.calibration_fraction < 1:
                raise ValueError('Calibration fraction must be in [0, 1).')
            # Validate policy even if all market quotes are absent.
            moneyline_ledger(pd.DataFrame({'ML_Home': [-110], 'ML_Away': [-110]}), [1], [.5],
                             args.min_ev, args.min_probability)
            test = frame.loc[frame.Date >= start]
            merged, info = attach_moneyline_market(test, args.sportsbook)
            quotes = merged[['game_id', 'ML_Home', 'ML_Away']]
            protocol = dict(version=1, task='moneyline', split='fixed_chronological_holdout',
                            start=start, end=end, train_start=frame.Date.min(), sportsbook=args.sportsbook,
                            policy=dict(min_ev=args.min_ev, min_probability=args.min_probability,
                                        stake_units=1.0, side_selection='highest_qualifying_ev'),
                            calibration_fraction=args.calibration_fraction, seed=args.seed,
                            quote_timing=info.get('quote_timing', 'unknown'))
            manifest = dict(protocol=protocol, protocol_id=protocol_fingerprint(protocol, frame, quotes),
                            dataset=args.dataset, source_database=str(args.dataset_db.resolve()))
            report = save_run(output, frame, quotes, manifest, args)
            frame.to_csv(output / 'snapshot.csv.gz', index=False, float_format='%.17g')
            quotes.to_csv(output / 'quotes.csv', index=False, float_format='%.17g')
            manifest['files'] = {name: file_digest(output / name)
                                 for name in ['snapshot.csv.gz', 'quotes.csv', 'report.json', 'predictions.csv']}
            write_experiment_report(output / 'manifest.json', manifest)
        else:
            frame, quotes, manifest = load_baseline(args.baseline)
            if args.refresh_features:
                candidate = load_dataset(args.dataset_db, args.dataset)
                candidate = candidate.loc[(candidate.Date >= frame.Date.min()) & (candidate.Date <= manifest['protocol']['end'])]
                frame = align_candidate(candidate, frame)
            report = save_run(output, frame, quotes, manifest, args)
        print(f"Saved {output / 'summary.md'}")
        print(f"Log loss {report['diagnostics']['log_loss']:.6f}; bets {report['betting']['bets']}; "
              f"profit {report['betting']['profit_units']:.2f} units")
        print(LIMITATION)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
