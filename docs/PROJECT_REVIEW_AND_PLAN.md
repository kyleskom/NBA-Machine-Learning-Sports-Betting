# NBA prediction and betting system: review and implementation plan

Reviewed September 12, 2026, against the current working tree, including existing uncommitted work.

## Assessment

The intended product is a daily NBA betting decision system: estimate probabilities, compare them with available prices, select worthwhile bets, manage exposure, and learn from settled results. The repository currently provides a prediction prototype with odds display and moneyline EV/Kelly calculations. It does not yet establish a reproducible, tradable edge.

The most urgent improvement is trustworthy evaluation. Existing model accuracy cannot be treated as evidence of profitability because historical feature leakage and live inference defects are present. A lower accuracy after fixing these problems would be a more honest starting point.

Winning more bets and making more money are different objectives. At -300, a 70% win probability produces approximately -6.67% expected return per dollar staked; at -110, a 55% win probability produces +5%. Both depend on probabilities being accurate. The system needs calibrated probabilities and favorable executable prices, plus the ability to recommend no bet. No model architecture guarantees profit.

## Scope and verification

Reviewed the ingestion and dataset code, all six training scripts, both prediction runners, utilities, tests, feature-alignment script, configuration, model metadata, SQLite databases, Flask backend/templates, front-end behavior, notebook, README, and notes. Existing application changes were preserved.

Verification performed:

- `env/bin/python -m unittest discover -s Tests -v`: all 40 tests passed.
- `env/bin/python -m pip check`: no broken installed requirements reported. This is not a clean-environment installation test.
- Read-only database inspection: largest dataset has 16,969 rows, dated 2012-11-04 through 2026-01-07; no duplicate date/home/away groups and no null totals, scores, or home-rest values in that table.
- Team database has 4,320 daily tables and stops at 2026-01-07. `PlayerData.sqlite` contains zero tables. The configured default `dataset_2012-26_player_v1` does not exist.
- In 538 of 539 checkable games in `odds_2025-26`, the home team's same-day snapshot has one more game played than the prior day's snapshot, and its change in wins matches that game's result. This is a targeted check of that season, not an audit of every historical row.
- Both saved XGBoost calibration files failed to load in a fresh process with `AttributeError` for `__main__.BoosterWrapper`; the production loader returns `None` for those failures.
- A synthetic frame using the stored team feature order and configured player columns confirmed that legacy totals inference truncates away the actual OU line.
- An isolated temporary-directory check confirmed that XGBoost model discovery can select a `_features.json` sidecar as a model.

No expensive retraining, external feed purchase, live bet placement, or historical profitability estimate was performed. Model filenames are labels, not independently reproduced performance evidence.

## Findings, ordered by importance

### 1. Historical features contain the outcome being predicted — critical

`src/Process-Data/Get_Data.py` queries cumulative statistics through a date and stores them under that date. `src/Process-Data/Create_Games.py:202` joins games to that same-date table. Backfilled snapshots therefore contain the game's own performance.

Concrete example: Golden State vs. Denver on 2025-10-23 ended with Golden State winning by six. Golden State's prior-day snapshot is GP=1, W=1; the same-day snapshot is GP=2, W=2. The existing training dataset contains GP=2, W=2 for that game. This is postgame information in prediction inputs.

Player aggregate features have the same design risk: `_rolling_team_features` includes the current observation, and raw observations are daily season-average snapshots rather than individual game logs. Shifting player baselines alone does not fix the unshifted team aggregates.

Required: feature construction with explicit pregame cutoffs. Prefer completed game logs and immutable source snapshots. For legacy daily data, use strictly earlier snapshots and label their availability assumptions; an earlier date is a minimum repair, not proof of historical publication time. Changing a target game's result must never change its pregame features.

### 2. Live prediction does not reliably reproduce training — critical

- **Calibration fails silently:** `src/Predict/XGBoost_Runner.py:41` catches all calibration-load errors. Saved objects depend on a training-script-local `BoosterWrapper`. EV and Kelly can therefore use different probabilities from the ones evaluated during training.
- **Totals feature corruption:** the stored XGBoost models have 106 moneyline and 107 totals inputs, with no feature sidecars. The current live frame adds 68 player fields. Truncating to the first 107 inputs drops the actual OU line. The stored dataset's totals feature tail is `OU, Days-Rest-Home, Days-Rest-Away`; the fallback feeds `Days-Rest-Home, Days-Rest-Away, H_PLR_MIN_SUM` instead. Exact original artifact provenance is absent, but omission of the live line is directly reproducible.
- **Artifact discovery:** the glob at `XGBoost_Runner.py:29` also matches feature sidecars. Since it chooses by modification time first, newly written metadata can be selected as the model.
- **Team identity:** dataset and live builders use fixed row positions, without verifying the selected row's team ID. Changes in ordering can silently attach the wrong team statistics. Historical training skips any snapshot without exactly 30 teams, excluding early-season games.
- **Time/rest mismatch:** live rest compares naive local time with a CSV containing UTC times, adds one day to an elapsed-time calculation, and does not share historical rest logic. It can include a game already started that day. Historical rest is also derived from odds coverage, which may omit played games.
- **Missingness mismatch:** historical player joins require exact dates, while live joins reuse any older team record with no age or season limit. Missing model inputs are silently filled with zeros.

Required: one shared feature builder, stable IDs, UTC-aware timestamps, explicit feature schema, persisted preprocessing/calibration, and a promoted-model manifest. Reject incompatible or stale inputs instead of manufacturing a normal-looking bet recommendation.

### 3. Player/injury/lineup capability exists in code but is not operational

The player database is empty and injury/lineup feed URLs and CSV path are blank. Player details displayed in Flask are a separate API path; viewing an injury in the UI does not make it a model feature.

The current design also needs stronger semantics:

- Rolling 7/14/30 observations are daily snapshots of cumulative averages, not the last 7/14/30 games.
- Name-based player grouping and joins need stable IDs, dated roster membership, and season boundaries.
- The configured Base stats endpoint does not supply the intended advanced usage field; the implementation can substitute a constant usage proxy.
- Injury status weights such as 0.5 for questionable are assumptions, not learned play probabilities.
- Feed request success is treated as injury coverage for all teams on that date, without verifying completeness.
- A URL returning current injuries can be stamped with a requested historical date; historical source timestamps are not verified.
- Latest injury and lineup information must refresh intraday; incremental ingestion currently skips dates already collected.

Required: timestamped reports, source-level coverage, projected active rotations/minutes, and an estimate of player impact relative to replacement minutes. Account for uncertainty through active/out/limited scenarios. Do not backfill today's injury state into past games.

### 4. No reproducible betting backtest or result ledger

Odds rows contain moneylines, spread, and total, but no quote time, explicit bookmaker column, or over/under prices. `Create_Games` drops moneyline/spread information from the produced dataset. There is no persistent prediction/recommendation/bet/settlement ledger, execution simulation, or closing-price comparison.

Accuracy alone does not establish an edge against sportsbook prices. The application cannot presently answer which bets were available when predicted, which were taken, at what stake and price, and what happened afterward.

Required: historical price snapshots, outcomes with settlement rules, walk-forward simulated execution, flat-stake results before variable sizing, and a complete paper-trading ledger.

### 5. Training needs better probability and selection discipline

Useful foundations already exist: chronological sorting, `TimeSeriesSplit`, log-loss hyperparameter selection, separate test tails, logistic baselines, and calibration options.

Remaining problems:

- NN and logistic scripts impute with medians from the entire dataset before splitting. Fit imputation within each training fold and persist it for inference.
- XGBoost lists `merror` last, so early stopping uses classification error while trial selection uses log loss. Native Booster prediction also uses all retained trees unless restricted; explicitly use the best iteration or save the best tree slice. This behavior is documented in [XGBoost's prediction guide](https://xgboost.readthedocs.io/en/stable/prediction.html).
- The final XGBoost calibration block also serves as its early-stopping block. Separate model fitting/selection from calibration.
- Class balancing changes natural outcome frequencies. This is especially consequential for the approximately 1.07% push rate in the largest dataset: balancing makes pushes disproportionately influential. Compare unweighted probability models and evaluate any weighting on untouched data.
- Date-based splits must keep all snapshots of a game together and prevent outcomes unavailable at a fold cutoff from entering training.
- NN selection uses accuracy encoded in filenames; XGBoost selection uses file modification time. Neither is a controlled promotion process.
- Logistic scripts print evaluations but do not save deployable model bundles. NN probabilities have no calibration stage.
- No persisted per-game out-of-sample predictions, Brier scores, reliability plots, market comparison, or uncertainty intervals.

Log loss and Brier score measure broader probability quality; reliability plots are needed to inspect calibration specifically. See [scikit-learn's calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

### 6. Betting decisions and totals settlement are incomplete

Moneyline EV is correctly expressed as expected dollars per $100 staked for valid odds and probabilities. The Kelly expression uses net payout odds and is broadly the full-Kelly formula, despite a misleading conversion function name. It rounds payout odds early, lacks input validation, and offers no fractional sizing or aggregate exposure control. Runners calculate Kelly even when display is disabled, so absent odds can still crash.

Totals models have classes under/over/push, but runners label class 2 as OVER. They do not ingest actual over and under prices or compute totals EV.

Required: a shared market pricing and settlement module. With decimal return `d`, net payout `b=d-1`, and win/loss/push probabilities, expected return is `p_win*b - p_loss`; pushes refund the stake. Betting eligibility must use actual executable odds. Removing vig is useful for estimating a market benchmark, not for changing the payout used in EV.

Add explicit BET / WAIT / PASS states, freshness and availability requirements, validation-selected edge thresholds, and configurable bankroll constraints. High confidence is not sufficient to bet.

### 7. Application reliability and presentation need repair

- Flask launches CLI subprocesses and parses formatted text. XGBoost prints winner first, while Flask assumes home first, so an away winner can reverse matchup identity. Replace parsing with shared structured results.
- The homepage hardcodes approximately 69% moneyline and 55% totals accuracy. Stored model filenames label totals accuracy as 50.1% and 49.2%; no evaluation report supports the 55% display. Show verified metrics, dates, sample sizes, and model versions.
- Several network calls lack timeouts and robust status/schema checks. There is no demonstrated daily orchestration, freshness monitoring, or durable failure status.
- Historical odds ingestion skips whole dates once present and does not consistently enforce final-game status, allowing incomplete days/results to persist. Table-name preference can select an older `_new` table over a more complete table; reconcile sources by game ID.
- Regular-season-only stats are used across season ranges that include playoffs. Current schedule/season constants need a configuration-driven rollover.
- A RapidAPI credential is hardcoded in `Flask/app.py`. Remove it from source, rotate it through the provider, and use environment configuration; credential validity was not tested.
- The Colab notebook includes broad file deletion commands. Replace those with a scoped setup directory.
- The feature-alignment script has a separate feature exclusion list and mostly prints discrepancies. It should reuse the authoritative schema and fail on mismatches.

## Consolidated action plan

This is the single implementation backlog, replacing the earlier phase-based plan. It combines the review findings, expanded data collection, ten additional feature experiments, and model architecture work. Each workstream has a stable ID so a future request can name it directly, such as “implement W03 player ingestion” or “run the W07 architecture comparison.” All work below remains planned; the prior audit and this document are complete.

Priority: P0 = correctness prerequisite; P1 = core capability; P2 = measured improvement; P3 = later research. All feature and architecture rankings are proposed research priorities, not proven rankings on this project's data.

| ID | Workstream | Priority | Depends on | Deliverable |
|---|---|---|---|---|
| W01 | Repair inference and model artifacts | P0 | None | Reliable, reproducible prediction service |
| W02 | Build a point-in-time data foundation | P0 | None | Shared pregame feature builder and versioned data |
| W03 | Collect player, roster, injury, and lineup data | P1 | W02 schema; collection adapters can start immediately | Usable historical and live player context |
| W04 | Collect market prices and execution context | P1 | W02 schema; provider evaluation can start immediately | Timestamped multi-book prices |
| W05 | Build baseline features and ten additional metrics | P1/P2 | W02; W03/W04 for dependent features | Feature registry and ablation experiments |
| W06 | Establish evaluation and betting backtesting | P0/P1 | W01, W02; W04 for betting evaluation | Reproducible probability and ROI reports |
| W07 | Compare and improve model architectures | P2 | W06 and available W05 features | Validated model leaderboard and model bundles |
| W08 | Implement bet selection and bankroll controls | P1 | W04, W06, validated W07 model | BET / WAIT / PASS engine and ledger |
| W09 | Build the daily application and monitoring | P1 | W01; W08 for betting UI | Observable collection, prediction, and settlement workflow |
| W10 | Paper trade and govern model promotion | P1 | W06, W08, W09 | Prospective evidence and promotion decision |

### W01 — Repair inference and model artifacts

- [ ] W01.1 Preserve existing models/datasets as unvalidated reference artifacts; stop presenting their filename accuracy as established betting performance.
- [ ] W01.2 Replace discovery by glob/filename accuracy with a manifest containing model path, ordered feature schema, class map, preprocessing, calibrator, training cutoff, dataset hash, code/dependency versions, and evaluation report.
- [ ] W01.3 Move calibration wrappers into an importable module; verify fresh-process serialization. Fail clearly if a required calibrator cannot load.
- [ ] W01.4 Remove positional feature truncation and silent zero-fill of required inputs. Preserve the offered totals line by name. Allow optional missing values only under a model's explicit missingness policy.
- [ ] W01.5 Correct push handling, odds validation, and Kelly calculation when odds are absent. Keep full precision until display.
- [ ] W01.6 Return structured results with immutable game/home/away IDs for both CLI and Flask. Correct away-winner reversal and replace static accuracy claims with verified reports.

**Code affected:** `src/Predict/`, `src/Utils/PlayerContext.py`, EV/Kelly utilities, `main.py`, `Flask/app.py`, and new shared model-bundle/result modules.

**Done when:** round-trip loading preserves probabilities; column reordering changes no prediction; missing required inputs reject prediction; feature sidecars cannot be loaded as models; away wins preserve identity; pushes and absent odds are handled correctly.

### W02 — Build a point-in-time data foundation

- [ ] W02.1 Normalize `games`, `team_game_logs`, `player_game_logs`, `roster_memberships`, source snapshots, feature snapshots, and dataset versions. Start with indexed SQLite and versioned raw files; a distributed platform is unnecessary at this scale.
- [ ] W02.2 Store stable IDs, season/type, venue, scheduled/actual UTC tipoff, game status, source event/publication time where available, fetched time, and raw payload reference/hash. Keep event time separate from when information became available.
- [ ] W02.3 Implement one `build_features(game_id, as_of)` path for training, replay, and live prediction. Labels stay separate; completed games must precede the cutoff. For historically reconstructed records, track that actual original publication time may be unknown.
- [ ] W02.4 Replace same-day postgame joins with earlier completed-game histories. Add priors for season openers, rookies, trades, missing history, and teams absent from early daily snapshots.
- [ ] W02.5 Backfill through the latest completed season, reconcile overlapping odds tables, refresh incomplete dates/results, and produce coverage reports per season/source/team. Do not prefer a stale table because its name ends in `_new`.
- [ ] W02.6 Calculate rest from the complete schedule/results, not odds coverage. Use UTC-aware elapsed rest plus explicitly defined local-calendar back-to-backs. Include travel/venue/neutral-site and playoff context; remove hardcoded season constants.
- [ ] W02.7 Make upserts idempotent at record/snapshot level. Preserve revisions and source provenance; quarantined, duplicate, missing, and excluded records need reasons.

**Data policy:** retain existing older data, but compare recent-season training windows with all-history training. Backfill basic logs first; obtain advanced/tracking history only over verified coverage periods. A million player observations still do not create a million independent game-result labels. Split all observations belonging to the same game together.

**Done when:** modifying a target game's score leaves its pregame features unchanged; no future report/price can enter an earlier cutoff; rerunning ingestion creates no duplicates; all included features have lineage; historical/live replay at an identical cutoff yields the same features.

### W03 — Collect more player and basketball data

Build adapters around a normalized contract so a provider can be replaced. NBA player and lineup statistics are starting points for source evaluation: [player statistics](https://www.nba.com/stats/players/traditional), [lineup statistics](https://www.nba.com/stats/lineups/advanced). These pages do not establish a reliable ingestion API, complete archives, or redistribution rights; verify those during adapter work.

| Task | Dataset to obtain | Required fields | Intended use / refresh |
|---|---|---|---|
| W03.1 | Individual player game logs | Player/game/team/opponent IDs, minutes, starter/DNP reason, shots made/attempted by type, free throws, rebounds, assists, turnovers, fouls, steals, blocks | True game-based recent form, workload, and rotation models; ingest after final games and reconcile revisions |
| W03.2 | Advanced player and team statistics | Possessions, usage, offensive/defensive context, shooting efficiency, rebounding/turnover rates | Rate-based ability estimates; derive transparent rates where possible, label approximations |
| W03.3 | Historical rosters and transactions | Effective dates for team membership, trades, signings, waivers, coach changes | Correct player-team attribution and roster transitions; refresh when changed |
| W03.4 | Injury/availability report snapshots | Player/game IDs, status, reason, publication time, fetched time, source, completeness | Estimate active/out/limited scenarios; preserve every change before tipoff |
| W03.5 | Projected/confirmed lineups and restrictions | Player/game IDs, projected versus confirmed status, timestamp, reported minutes limit and provenance | Expected rotations and minutes; refresh near tipoff and on reports |
| W03.6 | Play-by-play and substitution stints | Period/clock, event type, score, players on court, possessions and substitutions | Regularized player impact, teammate continuity, transition and lineup context; later tier after logs work |
| W03.7 | Shot-zone and tracking summaries | Rim/paint/midrange/three-point attempts, shot quality or contest context if available, potential assists, drives, transition events | Matchup and creation metrics; optional tier with coverage/missingness flags |
| W03.8 | Full schedule and venue history | Actual game times, venue coordinates/timezone, neutral-site flag, overtime periods | Travel, rest, fatigue, and schedule density; reconcile postponements |

**Availability collection:** official NBA injury reports change throughout the day, so one daily fetch is insufficient. Use the [official report source](https://official.nba.com/nba-injury-report-2025-26-season/) or a provider with verified timestamped history. Proposed decision snapshots are morning, 60 minutes before tipoff, and a later pregame refresh, subject to source limits and available reports. Archive published updates when feasible. This is a collection plan, not a scheduler created in this task.

- [ ] W03.9 Replace constant questionable/doubtful weights with a calibrated participation model using only pregame reports/history. Sparse groups need pooled estimates. A missing report is unknown, not healthy.
- [ ] W03.10 Build a conditional minutes model for active players; combine it with participation probabilities and injury scenarios. Project the whole regulation rotation to 240 team minutes, cap individual regulation minutes at 48, and model overtime separately if needed. Starter status alone does not determine minutes.
- [ ] W03.11 Estimate offensive/defensive impact relative to replacement. Start with transparent, strongly regularized historical estimates; add adjusted stint-based estimates when possession data passes validation. Never use retrospectively published full-season ratings as early-season inputs.
- [ ] W03.12 Preserve report uncertainty and source conflict; avoid double-counting an absence already reflected in recent team performance. Require a model validated for incomplete player coverage when falling back.

**Procurement gate:** test at least ordinary games, injury-heavy games, trade transitions, and postseason games before a large backfill or subscription. Document fields, history, timestamps, update latency, rate limits, costs, and permitted use. Reconstructing actual participation after a game does not reconstruct what was known before it.

**Done when:** every player maps by ID; roster dates are respected; reports cannot be backdated; minutes reconcile; injured players' replacements are represented; coverage is measured per team/game rather than inferred from HTTP success. Publish participation calibration and minutes error separately from game prediction performance.

### W04 — Collect market prices and execution context

- [ ] W04.1 Persist quote ID, game/book/market/side, exact line, both side prices, bookmaker update time, fetched time, and open/suspended status where supplied.
- [ ] W04.2 Collect moneyline, spread, and totals prices across books actually accessible to the user. Keep closing snapshots separate from earlier decision inputs.
- [ ] W04.3 Build time-aligned consensus probabilities, vig estimates, line movement, and best executable price. Normalize both sides within each book before aggregation; do not combine unrelated total/spread lines.
- [ ] W04.4 Trial a historical provider, then backfill only verified coverage. [The Odds API documents timestamped historical snapshots](https://the-odds-api.com/historical-odds-data/); final choice depends on books, date coverage, latency, and budget. Start recording forward history immediately once configured.
- [ ] W04.5 Record quote expiry, observed acceptance/rejection, price changes, and execution constraints. A scraped quote is not proof of an accepted bet.

**Done when:** replay retrieves only quotes at or before the cutoff; totals have both prices; stale/suspended/missing quotes are ineligible; market and model comparisons use the same times/games.

### W05 — Build basketball features and ten additional metrics

First retain and repair the previously proposed fundamentals: opponent-adjusted offense/defense and net rating, pace, Elo, game-based recent form, shooting/turnover/rebounding/free-throw rates, home advantage, rest/travel, season/playoff context, and contemporaneous market strength. Use shrunk last-5/10/20-game summaries and exponentially weighted form as candidate windows; choose decay/windows within training validation. Preserve level, trend, sample size, missingness, and age. Never average daily cumulative snapshots and label the result recent game form.

The ten metrics below extend those fundamentals. Each row is a defined feature family that may produce home/away/difference and uncertainty columns. They are my prioritized hypotheses, to be tested through W06; no source proves these are the ten best for this repository.

| ID | Additional metric | Initial operational definition | Why test it / data requirement |
|---|---|---|---|
| M01 | Expected rotation impact change | Sum `(projected_minutes - reference_minutes) / 48 * shrunk_player_impact`, with offensive/defensive channels. Reference is the rotation underlying the baseline team rating. | Measures today's lineup change rather than raw injury count; player impact, roster, projected minutes |
| M02 | Availability uncertainty | Scenario-weighted variance of projected team impact across plausible active/out/limited rotations; also retain top-player play probabilities. | Distinguishes stable edges from picks dependent on unresolved news; report histories and participation model |
| M03 | Lineup continuity | Projected-pair-weighted `log(1 + prior possessions together)` plus share of projected minutes from returning players. | Tests disruption from trades/new rotations without memorizing five-man lineup IDs; stints and roster histories |
| M04 | Missing offensive creation | Lost projected creation from unavailable/restricted players versus replacement creation, using prior potential assists and self-created attempts; AST/usage proxies when tracking is absent. | Tests losses season PPG averages overlook; distinguish real tracking from proxies |
| M05 | Rim-pressure/protection mismatch | Own shrunk rim attempt frequency/efficiency interacted with opponent rim attempts/efficiency allowed, adjusted for projected interior rotation. | Tests style-specific scoring advantages; shot zones and active defenders |
| M06 | Three-point regression and volatility | Recent 3P% minus shrunk player/shot-mix expectation, alongside projected 3PA and shooting uncertainty. | Separates potentially unsustainable shooting from changes in shot volume/quality; long-run priors fitted before cutoff |
| M07 | Transition exposure mismatch | Own transition frequency/points per possession interacted with opponent live-ball turnover rate and transition defense. | Tests fast-break scoring not captured by average pace alone; play-by-play; label any turnover-only proxy |
| M08 | Rotation-weighted recent workload | Weight each projected player's prior 72-hour/7-day minutes and overtime load by expected minutes share; add elapsed rest/travel interactions. | Captures player-specific fatigue beyond team days off; player logs and schedule |
| M09 | Replacement/bench quality gap | Expected quality of nonstarter and injury-replacement minutes versus the opponent's comparable rotation, with role and sample-size controls. | Tests whether a team can absorb absences and maintain performance; regularized player estimates and minute projections |
| M10 | Synchronized market disagreement | Robust dispersion (such as median absolute deviation) of per-book no-vig probabilities at the same cutoff, plus deviation of offered price from consensus. | Tests price disagreement and possible stale-quote artifacts; timestamped prices, source freshness checks, no later closing data |

- [ ] W05.1 Implement a feature registry with formula/version, units, source fields, availability rule, missingness policy, season reset, and tests.
- [ ] W05.2 Start with M01, M02, M08, M09, M10 as their source prerequisites become reliable; then add the tracking/stint-dependent metrics.
- [ ] W05.3 Run add-one-group and remove-one-group experiments on identical games/folds. Separate improvement from a change in covered sample. Use missing-data flags and a common-coverage comparison where needed.
- [ ] W05.4 Control overlapping signals: M01 and M09 can overlap, and M10 may reveal feed problems rather than predictive value. Keep a metric only if it adds stable information after existing features.

**Done when:** feature values can be explained and replayed; no future inputs; performance gains survive later-period evaluation; missing advanced data cannot masquerade as a zero basketball effect.

### W06 — Establish probability evaluation and betting backtesting

- [ ] W06.1 Split by chronological game/date blocks with inner tuning, separate calibration, later strategy validation, and a sealed final period. All snapshots of a game stay together; training labels must have settled by the training cutoff.
- [ ] W06.2 Move imputation/scaling/feature selection inside each training fold. Separate XGBoost early stopping from calibration, stop on probability loss, and save/use the correct tree iteration.
- [ ] W06.3 Establish home-win-rate, bookmaker-favorite, no-vig market, Elo, and regularized logistic baselines. Use natural class frequencies; test weighting only as an explicit experiment.
- [ ] W06.4 Persist per-game out-of-sample probabilities, source cutoffs, model/run IDs, coverage, and outcomes. Compare stats-only, market-only, and combined models.
- [ ] W06.5 Replay a fixed 60-minute-pregame strategy first. Test late lineup refreshes separately. Simulate quote age, realistic execution delay, unavailable/rejected quotes, exact prices/lines, stake limits, pushes, voids, overtime rules, and settlement timing.
- [ ] W06.6 Report probability quality: log loss, Brier score, reliability plots, sample sizes, and results by season, market, probability band, odds band, and data coverage. Accuracy is supplementary.
- [ ] W06.7 Report betting quality: flat-stake ROI, bet count, turnover, P&L, drawdown, CLV, and price/fill sensitivity; include explicit costs. Test variable sizing only after flat-stake results are understood.
- [ ] W06.8 Estimate uncertainty with date/week blocks, account for repeated model/threshold searches, and preregister the final comparison. Keep totals/spread line movement separate from odds movement unless comparing on a common line.

**Done when:** a run ID reproduces the full report and bet ledger; market comparisons are matched; thresholds were not chosen on test results; inconclusive results are labeled inconclusive. CLV supports diagnosis but is not proof of profitability. Previously inspected test periods become development data; future paper trading is a new holdout.

### W07 — Compare and improve model architectures

#### Current neural networks: what they actually are

Saved model configs were inspected directly during this expansion:

| Model | Saved architecture | Main limitations |
|---|---|---|
| Moneyline NN | 106 inputs -> Dense 192 -> 96 -> 48 -> 32 -> 32 -> 32 -> two-class softmax; ReLU, batch normalization, about 31.4% dropout in each hidden layer | Deep narrowing MLP on team aggregates; no player/rotation representation or probability calibration |
| Totals NN | 107 inputs -> Dense 192 -> 96 -> three-class softmax; ReLU, about 7.0% dropout | Direct under/over/push classification; limited line/distribution structure and broken legacy feature handling |

Training searches 2–6 hidden layers, uses row-wise L2 normalization, whole-dataset median imputation before chronological splitting, and a single 80/10/10 train/validation/test split by default. Each runner chooses from model filenames rather than a validated promotion manifest. The saved models do not consume the proposed player feature set. Their filename accuracies are not clean performance evidence.

The two-class softmax itself is valid; replacing it with a sigmoid is not a substantive accuracy improvement. Depth and dropout may be suboptimal, but this audit did not prove overfitting. Row normalization divides every feature by a game-specific norm, coupling the scaling of rest, scores, percentages, and other inputs; compare train-fitted per-feature scaling instead. Fixing the data and preprocessing is the first NN improvement.

#### Model comparison order

| Task | Candidate | Role and decision |
|---|---|---|
| W07.1 | Regularized logistic regression and Elo | Cheap, interpretable probability baselines; persist their complete pipelines |
| W07.2 | Repaired XGBoost and CatBoost | First serious tabular contenders using identical features/folds; CatBoost is an alternative boosting implementation, not a presumed winner ([documentation](https://catboost.ai/docs/en/concepts/algorithm-main-stages)) |
| W07.3 | Small residual MLP | First NN replacement experiment: simpler feature scaling, skip connections, modest regularization, repeated seeds |
| W07.4 | TabM | Preferred advanced tabular NN challenger; efficient MLP ensembling is supported by the [TabM paper](https://arxiv.org/abs/2410.24210), not yet by NBA results in this repository |
| W07.5 | Calibrated market-plus-model blend | Likely practical deployment candidate if it improves genuinely later predictions and bet evaluation; shrink uncertain adjustments toward market baseline |
| W07.6 | Player-set encoder with team/context branch | Later basketball-specific NN after W03/W05 pass; explicitly represent available players and rotations |
| W07.7 | FT-Transformer; optional GRU/temporal convolution | Lower-priority research: FT-Transformer for tabular interactions, sequence models only for actual pregame sequences of past games |

The [tabular architecture comparison by Gorishniy et al.](https://arxiv.org/abs/2106.11959) motivates residual-network and FT-Transformer baselines and finds no universal winner against boosted trees. These are reasons to run controlled comparisons, not predictions of better NBA betting returns. Avoid a large Transformer/LSTM migration before basic contenders earn an edge.

#### Concrete residual-NN experiment

- [ ] Fit median imputation plus missing indicators and per-feature scaling on each training fold only; save them with the model.
- [ ] Initial architecture: numeric input -> Dense(128) -> two width-128 residual blocks -> Dense(64) -> output. Each residual block uses normalization, two dense transforms, SiLU/ReLU, a skip connection, and modest dropout.
- [ ] Treat width 64/128, one/two blocks, dropout 0.05–0.20, and modest weight decay as a small validation search, not claimed optimal settings. Use early stopping on validation log loss and fixed reproducible seeds.
- [ ] Train a few independent seeds; compare both single-model and averaged probabilities. Calibrate the final deployed ensemble on a later untouched calibration block.
- [ ] For moneyline, use binary cross-entropy and an independent probability calibrator. Keep an unweighted model as the default; do not use focal loss merely to emphasize rare outcomes.
- [ ] Compare against TabM under the same data, feature, calibration, and approximate compute budget. Benchmark runtime/memory on the current machine before scaling experiments or changing frameworks.

#### Market combination experiment

Use a regularized correction such as `logit(p_home) = logit(p_market_asof) + alpha * f(basketball_features)`. Fit the correction on historical pregame market data; constrain/shrink its complexity and tune `alpha` only on development data. Alternatively fit a simple nonnegative probability blend of market, tree, and NN predictions generated out of sample. Calibrate the exact final ensemble, not only its components. If a second-stage learner uses model predictions, they must be chronological out-of-fold predictions. Do not feed the market to a component and then arbitrarily add it again at full weight.

#### Player-aware architecture experiment

Each player's input includes prior rate/impact estimates, projected minutes, participation probability, workload, and data coverage. Apply one shared small player encoder, pool player embeddings using projected minutes and explicit masks, then compare the home and away team representations alongside team/context features. Pooling must not depend on player ordering. Use strong shrinkage and an unknown-player fallback; player ID embeddings are optional and must not replace evidence of ability.

This structure tests whether learned player interactions improve on M01/M09's transparent aggregates. Add limited pair interactions only if pooling proves useful. A roster permutation must preserve the prediction; changes in projected participation must influence the correct team. Fit minutes/participation models on earlier data and use out-of-fold projections in downstream training; actual target-game minutes are never inputs. Shared game targets should still be evaluated at game level.

#### Totals/spreads architecture experiment

First repair the existing line-conditioned classifier and retain it as a benchmark. Then predict a distribution over final score total and home-minus-away margin, with game-dependent uncertainty. Compare a regularized residual-distribution baseline to a small NN distribution head; discrete score simulation, residual resampling, or appropriately discretized distributions are candidates. Check fit rather than assuming independent Poisson scores or fixed normal variance.

For a total T and line L, derive `P(T>L)`, `P(T<L)`, and `P(T=L)` consistently. Half-point lines have zero push mass; over probability must decline as L increases. Keep overtime/settlement definitions consistent. If constructing a joint total/margin model, enforce valid scores and parity (total and margin have matching parity) and derive the win probability coherently. If moneyline and totals remain separate models, do not imply their outputs form a joint distribution. Multi-task learning is an experiment, not a prerequisite.

**Done when:** candidates are compared on identical chronological data with persisted per-game predictions; bundle round-trips pass; calibration is assessed in the bands actually bet; the selected model beats the predefined benchmark objective and has separately evaluated betting results. No promotion from filename accuracy or a single favorable season. Feature gains and architecture gains are reported separately.

### W08 — Implement bet selection and bankroll controls

- [ ] W08.1 Price all eligible sides using actual odds and win/loss/push probabilities: `EV = p_win * net_payout - p_loss` per unit stake. Vig removal is for market estimation; actual payout determines EV.
- [ ] W08.2 Produce BET / WAIT / PASS with reason, uncertainty, minimum acceptable odds, exact book/line/side, data ages, and unresolved injury scenarios. Thresholds are tuned on strategy-validation data only.
- [ ] W08.3 Add configurable fractional Kelly, per-bet/game/day limits, unsettled-capital reservation, correlated moneyline/spread/totals exposure controls, duplicate prevention, and drawdown/model-health stops. Do not treat an ensemble's disagreement as a complete probability confidence interval.
- [ ] W08.4 Persist predictions, recommendations, paper/actual accepted bets, quote IDs, stakes, settlements, bankroll events, and closing quotes separately. Record rejected and skipped opportunities too.
- [ ] W08.5 Make settlement idempotent, with explicit pushes, voids, postponements, overtime rules, and later corrections. Never assume an order was filled because it was recommended.

**Done when:** bankroll reconciles with open and settled positions; repeated runs create no duplicate positions; invalid/stale prices block betting; price and uncertainty sensitivity are included in evaluation.

### W09 — Build the daily application and monitoring

- [ ] W09.1 Complete the shared CLI/web result contract and remove text parsing. Show game identity, model probability, market benchmark, offered/fair/minimum price, EV, source freshness, and why a bet is BET / WAIT / PASS.
- [ ] W09.2 Display actual dated sample-size-qualified results, model version, ROI/drawdown, and calibration. Remove unsupported static success claims and stale “updated today” labels.
- [ ] W09.3 Add collection, feature generation, prediction, settlement, and reporting jobs with configured timezones/season rollover, bounded retries/timeouts, observable failures, freshness limits, and outage behavior.
- [ ] W09.4 Remove hardcoded credentials and arrange provider-side rotation. Scope notebook setup safely, lock dependencies, reuse the authoritative feature schema in the alignment checker, and add critical tests to CI.
- [ ] W09.5 Monitor data coverage, schema/model drift, participation/minutes errors, calibration, quote availability, and source costs. Use model promotion/rollback with a durable version trail.

**Done when:** one failed feed cannot silently produce a normal recommendation; every displayed figure traces to stored results; the full slate can be replayed and settled; secrets are external configuration.

### W10 — Paper trade and govern promotion

- [ ] W10.1 Freeze a strategy and decision time; record prospective decisions before outcomes and log real-time executable quotes and simulated acceptance assumptions.
- [ ] W10.2 Predefine minimum evidence, meaningful effect size, review dates, and drawdown/exposure tolerances. Use uncertainty and power analysis rather than an arbitrary “100 winning bets” gate.
- [ ] W10.3 Review calibration, achievable prices, CLV, ROI intervals, costs, drawdown, and stability across dates/markets. Do not stop a trial just because its running profit turns positive.
- [ ] W10.4 Promote only after clean data/inference checks, favorable reproducible out-of-sample evidence, sufficient prospective evidence, and functioning exposure controls. Otherwise retain paper status and record the unresolved hypothesis.

Hundreds of bets can still be inconclusive for a small edge. More staking cannot establish whether an edge exists. Any later real-money rollout needs the user's book access, bankroll preferences, and risk limits; this plan does not place bets or configure automated wagering.

## Recommended execution order and checkpoints

1. **Correctness checkpoint:** W01 + W02. Add invariant tests before retraining. Remove exposed credentials promptly under W09.4.
2. **Data checkpoint:** start W03/W04 source trials and forward snapshot collection as soon as the W02 schemas exist; backfill basic logs and verified history. Build W06 while advanced data is being collected.
3. **First decision-quality checkpoint:** W05 fundamentals + W06 + W07.1/W07.2. Produce a leakage-free moneyline-versus-market report and reproducible paper-bet replay.
4. **Improvement checkpoint:** W03 projections + W05 ten-metric experiments + W07.3/W07.4/W07.5. Keep only demonstrated improvements; repair/evaluate totals independently.
5. **Operational checkpoint:** W08 + W09 + W10. Deploy prospective paper trading with transparent results and no-bet behavior.
6. **Research checkpoint:** W07.6/W07.7 and advanced joint score models only if the simpler system/data justify them.

The first actionable slice is **W01 + W02, followed by basic player game-log ingestion W03.1 and historical/live odds W04**. The preferred initial modeling stack is a market benchmark, Elo/logistic baseline, XGBoost/CatBoost contenders, and a residual-MLP/TabM challenger. The eventual deployed model may be a calibrated blend, but it must earn that role in W06.

Future source-selection decisions: accessible sportsbooks, monthly data budget, historical coverage requirements, preferred betting time, and acceptable exposure. Their absence does not prevent correctness work, schema design, or source evaluation. No data purchase, collection automation, or model migration has been executed as part of this planning update.
