# NBA betting project review — September 29, 2026

## Assessment

Three agents reviewed data/features, models, and betting/evaluation in parallel. This review covers the current working tree, including existing uncommitted changes and local databases. Application code and model artifacts were not modified. The existing 78 unit tests passed. No new model was trained and no live bet was placed.

The project has a useful foundation, but it has not demonstrated an executable betting edge. The biggest improvements are trustworthy game/price alignment, timestamped market data, historical availability, and a consistent evaluation protocol. Larger neural networks should follow those changes.

Current data: 17,749 game rows, 206 dataset columns, November 2, 2012–June 13, 2026; moneyline uses 196 features and totals 197. Player-history coverage is approximately 99.93%. Normalized game/team identity, strictly earlier team snapshots and player-game history, report provenance, and feature manifests are valuable existing safeguards.

## Critical findings

### 1. The saved 122% ROI is invalid because games and prices are misaligned

`src/Train-Models/XGBoost_Model_ML.py:231` sorts the identity dataframe by date. `prepare_data()` sorts it again at line 55. The default unstable sort can reorder games with the same date. Features/labels/probabilities follow the second order, while `final_frame` at line 239 follows the first. The backtest combines one game's outcome and prediction with another game's prices. The totals trainer has the same double sorting pattern at lines 54 and 216.

The agent reproduced the saved erroneous market log loss, **0.739546**, exactly using that misalignment. A correctly joined market baseline on the same 1,306 final games has log loss **0.576260**. The archived home/away quote mapping itself was correct in this check.

Read-only replay of the saved 68.6% moneyline model and calibrator, with correctly aligned identities and the same 2% EV / 54% probability rule:

| Metric | Saved, misaligned report | Aligned diagnostic replay |
|---|---:|---:|
| Bets | 620 | 355 |
| Profit, one-unit stakes | +758.93 | +27.93 |
| ROI | 122.41% | 7.87% |
| Win rate | 69.19% | 55.21% |
| Maximum drawdown, units | 4.00 | 15.97 |

**The corrected 7.87% is still not executable-profit evidence.** Quote times are unknown, the policy was originally chosen using misaligned prices, and the final period has already been inspected. It is a diagnostic replay, not a fresh strategy test. The model's reported final log loss of 0.60554 is worse than the aligned archived market baseline, though aggregate probability loss and selective betting returns measure different things.

Repair: order the data once by immutable game identity and date, preserve identity through all transformations, join predictions to prices by game ID rather than array position, and assert alignment. Add a regression test with multiple games on one date and deliberately distinguishable prices/outcomes. Regenerate affected policy/backtest reports.

### 2. Historical availability is absent, despite availability feature columns

`PlayerData.sqlite.player_report_snapshots` has zero rows. Twenty-two dataset columns are constant; injury coverage, lineup coverage, and roster confirmation are all zero. The models have not learned the relationship between actual pregame availability and outcomes. Live availability also changes projected rotations, introducing feature conditions missing from historical training.

Historical features default to midnight New York time (`src/Players/features.py:40,112`), while live prediction uses the current timestamp (`main.py:107`). Importing same-day injury reports will not solve this unless the training cutoff changes too.

Repair: persist scheduled tipoff and a fixed decision time, initially 60 minutes before tipoff; reconstruct stats, reports, roster and odds available by that time. Model a later confirmed-lineup strategy separately. Never use the target game's eventual DNP or starter box-score flag as pregame evidence.

### 3. The price store cannot replay a real betting decision

`src/Storage/normalized.py:44` stores one price per game/book/market/side. Upserts at line 140 overwrite it. There are no quote observation timestamps, and total/spread rows have lines without side payout prices. All 97,280 canonical quote rows are labeled FanDuel, including migrated records whose original book provenance needs verification.

Add append-only quote observations with game ID, tipoff UTC, sportsbook, market, side, line, price, provider time, observed/fetched time, source payload reference, suspension status, and available limits when supplied. Capture opening, decision-time and closing observations at multiple books. Select contemporaneous prices; avoid choosing the best price retrospectively across different times.

Use no-vig probabilities as a market forecast benchmark, but use the actual offered price including vig for payout/EV. Closing prices are later evaluation data and must not enter an earlier decision's features.

### 4. The final holdout is used to select models

`src/Predict/model_selection.py:83` promotes candidates by final-test log loss. Repeating that process makes the final period a development selection period. The calibration and policy windows are also separated only after hyperparameter CV has already used those pre-final rows (`XGBoost_Model_ML.py:253,269`).

Reserve chronological blocks before any search: development with inner early stopping, later calibration, later betting-policy selection, then an untouched final period. Select candidates using development results; freeze candidate and policy before evaluating final data. Keep whole game dates together. The upcoming prospective season is more useful than repeatedly retuning on the already inspected 2025–26 period.

The selected ML registry entry is an older 68.34% / 0.60211 log-loss model without an evaluation fingerprint. The newer audited candidate is 68.61% / 0.60554, so increased accuracy did not improve probability loss. These entries are not directly comparable without a shared protocol.

### 5. Live eligibility and betting policy are incomplete

`src/Predict/service.py:15` checks lineup coverage. `src/Players/features.py:233–239` counts projected and confirmed starters toward that coverage. Therefore a projected five-player lineup can satisfy the configuration's requirement for confirmed starters. Check five confirmed starters explicitly when that is the intended policy.

The live board marks games READY after availability checks, but does not apply the saved EV/probability policy as a complete BET/PASS decision. Add market-specific eligibility, minimum acceptable price, quote freshness, reason codes, stake caps, and a persistent prediction/bet/settlement ledger. Evaluate both sides: the most likely winner is not always the side with positive EV.

Kelly currently returns percentages and rounds net payout early (`src/Utils/Kelly_Criterion.py`). Make units explicit, preserve precision, validate inputs, and expose configurable fractional Kelly plus per-game/daily exposure caps. Compare fixed stakes and conservative fractions in simulation. The appropriate fraction and limits should be chosen using evidence, rather than treating estimated probabilities as exact. Same-game moneyline/spread/total exposures are correlated.

### 6. Neural-network and totals paths need modernization

TensorFlow/Keras already implements Dense MLPs. The two stored `.keras` models have no feature manifests and are excluded by `src/Predict/NN_Runner.py:41`; retraining is required. Do not invent manifests for uncertain old feature orders.

NN training uses row-based chronological splits, per-row L2 normalization, one validation window, no saved probability calibration or market evaluation, and filename accuracy for production selection. Use the same date-based evaluation as trees and train-only per-column normalization embedded in the exported model.

The totals classifier can assign push probability to half-point lines, where a push cannot occur. Current selected totals accuracy is 49.23%, log loss 0.70637, and there is no priced totals backtest. Compare against natural class frequencies and an appropriate market baseline; do not infer profitability from accuracy. A score-distribution model is a better research direction than simply deepening this classifier.

## Additional data and features, in priority order

| Priority | Data/feature group | Useful derived features |
|---|---|---|
| First | Timestamped multi-book odds, both payout sides | No-vig consensus, current price versus consensus, synchronized book disagreement, movement known at decision time, executable best price |
| First | Historical pregame injuries, transactions, rosters, starters, minutes restrictions | Participation probability, projected minutes, minutes-weighted player impact, quality of replacement minutes, uncertainty of team strength |
| Next | Possession-based team game logs and opponent strength | Pace, offensive/defensive/net efficiency, opponent-adjusted 5/10/20-game and season form, home-minus-away differences |
| Next | Shooting and possession components | Effective shooting efficiency, turnovers per possession, offensive rebound share, free-throw rate, shot-location mix versus opponent defense |
| Next | Schedule and venue history | Back-to-back, three games in four nights, travel distance, time-zone changes, road-trip length, recent overtime and player minute load |
| Next | Season/playoff context and roster continuity | Playoff flag/round/series state, rest and rotation changes, prior-season strength with shrinkage, traded-player adjustment |
| Later | Play-by-play, lineup stints and tracking | Regularized offensive/defensive player impact, lineup interaction, rim protection, transition and half-court matchup tendencies |
| Later | Referees and coach changes | Carefully tested incremental effects after controlling for stronger signals |

Current team requests use regular-season per-game base statistics (`config.toml:1`). June playoff rows retain GP=82 while player history includes playoff games. Update team form consistently and track season stage. The 2023–24 dataset stops April 28 despite player history continuing through June; log missing games and exclusion reasons. Opening games need prior-season/preseason priors rather than disappearing for lack of a previous snapshot.

Retain continuous margin and separate team scores, spread, both moneylines, game identity and price metadata in the analytical dataset. The current builder discards margin and most market fields. Keep metadata separate from model inputs so outcome fields cannot accidentally become features.

Availability projections currently use heuristic injury weights and proportional minutes allocation. A stronger model estimates probability of playing, minutes conditional on playing, replacement rotation, and impact per possession. All fitting must use information available before each target game.

Add feature groups one at a time and remove them one at a time on identical covered games/folds. A gain from changing coverage is different from a gain from the feature. Record formulas, units, versions, missingness and source timing. Treat new features as hypotheses, not established improvements.

## Model experiments

Approximately 17,749 games is a medium-sized tabular task. Hundreds of thousands of player rows are not hundreds of thousands of independent game outcomes. A more complex model should earn its place against simple baselines.

| Order | Model | Rationale |
|---|---|---|
| 1 | No-vig market, regularized logistic regression, dynamic Elo/team-strength model | Establish what the project adds above a market and simple sports baseline |
| 2 | XGBoost, CatBoost and LightGBM | Strong tabular comparisons under identical inputs, splits and tuning budgets |
| 2 | Market residual model | Learn a regularized correction to a contemporaneous market probability: sigmoid(logit(p_market) + f(context)) |
| 3 | Small Keras residual MLP | Per-feature normalization, two or three 64–128-unit residual blocks, dropout/L2, sigmoid win head; calibrate on a separate period |
| 3 | Margin/total distribution model | Predict mean and scale, possibly covariance; derive win, spread and totals probabilities at available lines |
| 3 | Chronological stacking | Blend market, Elo, trees and MLP using only out-of-time base predictions |
| 4 | Shared home/away Conv1D/TCN or GRU encoders | Learn form from genuine prior 10–20-game sequences, with schedule gaps, masks and context |
| 4 | Player-set model with minute-weighted pooling or attention | Represent expected active players and replacements directly, once historical availability is trustworthy |
| Later | FT-Transformer, TabNet, lineup graph network | Research comparisons only after data and simpler architectures establish stable gains |

For the initial Keras model, begin with numeric context only: training-adapted Normalization → compact residual Dense blocks → probability head. Then test shared home/away encoders, home-minus-away embedding differences, player pooling, and temporal inputs as separate changes.

A richer architecture could combine a shared player encoder, masked minute-weighted home/away player pools, separate recent-game sequence encoders, team/context inputs and timestamp-matched market information. A shared trunk can predict margin and total distributions. Derive moneyline probability from the margin distribution or reconcile a separate win head so outputs do not contradict each other.

Start distributional modeling with a simple location/scale model and calibrated residual distribution; only test mixtures when the data justify them. Continuous distributions alone assign zero probability to exact integer pushes. Use a discrete score formulation or carefully validated discretization, and force zero push probability for half-point lines. Account for score correlation and overtime.

Avoid direct optimization of noisy historical dollar profit as the first model loss. Fit probabilities/distributions using proper losses, then choose betting rules on an independent period. Deep reinforcement learning and large transformers add complexity before the project has a trustworthy execution simulator.

## Implementation sequence and acceptance criteria

1. **Repair evaluation:** immutable game identity, single ordering, key-based prediction/quote joins, regression test, rebuilt reports. Mark the old ROI report invalid.
2. **Build replayable data:** fixed pregame decision cutoffs, append-only odds, historical availability, stage-consistent team form, expected-game coverage audit.
3. **Unify experiments:** identical date blocks for all model families; reserve calibration/policy before search; log out-of-time predictions and freeze final selection.
4. **Run baseline and feature experiments:** sports-only, market-only and combined; then compare trees, compact MLP and distributional targets.
5. **Run prospective paper betting:** log every forecast, quote, BET/PASS reason, intended stake, hypothetical fill, settlement and closing price before risking capital.

Measure log loss and Brier versus the aligned no-vig market, reliability in the subset actually selected for betting, ROI per amount staked, bet count, average odds, closing-line value, drawdown and concentration. Report uncertainty using blocks of game dates and performance by season/book/odds range, with enough observations to avoid mistaking noise for a subgroup edge. Closing-line value is supporting evidence, not a guaranteed return.

The practical objective is robust bankroll growth under uncertainty and exposure limits, not raw win percentage. A 70% forecast at -300 has negative EV; a 55% forecast at -110 has positive EV if the probability estimate is reliable. No model family guarantees winnings.

## Primary research and documentation

- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): independent calibration data and reliability assessment; Brier/log loss are not pure calibration measures.
- [Tree versus deep tabular benchmarks](https://arxiv.org/abs/2207.08815): supports retaining boosted-tree baselines; does not establish the best NBA model.
- [Keras structured-data classification](https://keras.io/examples/structured_data/structured_data_classification_from_scratch/): per-feature preprocessing and exported layers; adapt examples to chronological splits.
- [Keras TabTransformer example](https://keras.io/examples/structured_data/tabtransformer/): an implementation reference, not evidence of NBA betting gains.
- [Keras mixture-density network example](https://keras.io/examples/keras_recipes/approximating_non_function_mappings/): distributions rather than only point predictions; verify dependency compatibility before implementation.
- [Official NBA injury reports](https://official.nba.com/nba-injury-report-2025-26-season/): source for timestamped availability observations; historical coverage still requires auditing.
