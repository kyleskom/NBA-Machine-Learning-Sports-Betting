# NBA Sports Betting Using Machine Learning
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

## Overview
This project predicts NBA game winners and totals (over/under) using team stats and sportsbook odds. It pulls team data from 2007-08 through the current season, builds matchup features, and runs trained models to estimate win probabilities and totals outcomes. It also outputs expected value and optional Kelly Criterion stake sizing.

## Features
- Moneyline and totals predictions (XGBoost and Neural Net models).
- Expected value calculation and optional Kelly Criterion sizing.
- Odds ingest from supported sportsbooks or manual input.
- Data processing pipeline and model training scripts.
- Flask web app for browsing outputs.

## How it works
1. **Collect stats and odds**: `Get_Data` pulls daily team stats from NBA endpoints and stores them in SQLite. `Get_Odds_Data` pulls sportsbook odds and scores from SBR and stores them in a separate SQLite DB.
2. **Collect player + availability signals**: `Get_Player_Data` imports individual player game logs from the public SportsDataverse/ESPN archive or the NBA endpoint. It stores stable team/player IDs, opponents, tip times, starter/DNP context, positions, box scores, provenance, and immutable optional injury/lineup/roster snapshots in `Data/PlayerData.sqlite`.
3. **Build game features**: `Create_Games` merges team stats, odds, scores, days-rest, and home/away player, injury, and lineup aggregates into a training dataset.
4. **Train models**: XGBoost/NN scripts in `src/Train-Models` fit moneyline and totals models.
5. **Predict today**: `main.py` fetches today’s schedule, builds matchup features, loads trained models, and prints predictions, expected value, and optional Kelly Criterion sizing.

## Requirements
- Python 3.11
- Packages: Tensorflow, XGBoost, NumPy, Pandas, Colorama, Tqdm, Requests, Scikit-learn

Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Quick start
```bash
python3 main.py -xgb -odds=fanduel
```

Odds will be fetched automatically when `-odds` is provided. Supported books:
`fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`

If `-odds` is omitted, the script will prompt for manual odds and totals.

Optional flags:
- `-nn` run neural network model
- `-xgb` run XGBoost model
- `-A` run all models
- `-kc` show Kelly Criterion bankroll fraction

## Flask web app
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

```bash
cd Flask
flask --debug run
```

## Data pipeline and training
```bash
# Run these from the repository root
python src/Process-Data/Get_Data.py
python src/Process-Data/Get_Odds_Data.py
python src/Process-Data/Get_Player_Data.py --provider archive --backfill
python src/Process-Data/Get_Player_Data.py --status
python src/Process-Data/Create_Games.py

# Train models
python src/Train-Models/XGBoost_Model_ML.py --dataset dataset_2012-26_player_v2 --trials 100 --splits 5 --calibration sigmoid
python src/Train-Models/XGBoost_Model_UO.py --dataset dataset_2012-26_player_v2 --trials 100 --splits 5 --calibration sigmoid
python src/Train-Models/NN_Model_ML.py --dataset dataset_2012-26_player_v2
python src/Train-Models/NN_Model_UO.py --dataset dataset_2012-26_player_v2
python src/Train-Models/Logistic_Regression_ML.py --dataset dataset_2012-26_player_v2 --trials 50 --splits 5 --calibration sigmoid
python src/Train-Models/Logistic_Regression_UO.py --dataset dataset_2012-26_player_v2 --trials 50 --splits 5 --calibration sigmoid
```

`Create_Games` writes `dataset_2012-26_player_v2` by default. The current local database contains the 2012–13 through 2025–26 player archive and the built v2 game dataset. Use `--output-table <name>` to override.

The included player-aware XGBoost candidates use 196 moneyline and 197 totals features, each with a persisted sigmoid calibrator. Their latest chronological test-block results are 64.39% accuracy / 0.6197 log loss for moneyline and 50.38% accuracy / 0.7320 log loss for totals. These are model-selection diagnostics, not proof of betting profitability; evaluate against timestamped executable prices before wagering.

Live adapters are configured in `config.toml`:

- `player-injuries` retrieves the latest official NBA injury-report PDF for the slate, checking the published report times.
- `player-lineups` retrieves the NBA daily lineup feed and records the first five listed players as its current starter list, with the feed's confirmed/projected status.
- `player-rosters` stores every player in that same daily game feed as a game-day roster snapshot.

`main.py` refreshes all three immediately before it builds predictions. You can also refresh them without fetching game logs:

```bash
python src/Process-Data/Get_Player_Data.py --reports-only
```

Run that command again after a new official injury report or lineup update. The pipeline stores every distinct timestamped snapshot together with its original source response, URL, SHA-256 hash, content type, and parser version. It never converts a missing feed into a healthy or complete team. `primary_url`, `secondary_url`, and `json_path` remain available as provider overrides or local report inputs.

Reports must be timestamped JSON with `game_date`, timezone-aware `published_at`, `records`, and `complete_teams`. An empty `complete_teams` list means coverage is unknown. Import a saved report with:

```bash
python src/Process-Data/Get_Player_Data.py --reports-only --report-json report.json --kind injuries
```

If a report is not yet published or a team has not submitted one, the affected coverage stays `0.0`; player-rotation estimates are then provisional. Newly trained XGBoost and NN models save a `<model>_features.json` sidecar; prediction refuses legacy models without the exact v2 feature contract.

`[bet-eligibility]` is configured to issue **no bet** unless both teams have complete, fresh injury and game-day roster coverage plus five confirmed starters. This intentionally defers a prediction when the live information needed to price player availability is not yet known.

### Neural network notes
- The NN training scripts default to `dataset_2012-26_player_v2`.
- They save models into `Models/` with timestamped names plus feature sidecars for prediction-time alignment.

### Backfilling missing data
Get_Data normally fetches only new dates in the current season. To fill missing dates:
```bash
cd src/Process-Data
python -m Get_Data --backfill
```

To backfill a single season:
```bash
cd src/Process-Data
python -m Get_Data --backfill --season 2025-26
```

### Backfilling odds data
Get_Odds_Data normally fetches only new dates in the current season. To fill missing odds dates:
```bash
cd src/Process-Data
python -m Get_Odds_Data --backfill
```

To backfill a single season:
```bash
cd src/Process-Data
python -m Get_Odds_Data --backfill --season 2025-26
```

## Contributing
Contributions are welcome. If you change model behavior or data pipelines, add a note in the README and update any related scripts or docs.
