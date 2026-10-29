# Current-season data backfill — October 26, 2026

Verified at 2026-10-26T13:24:17.701132-04:00.

| Data | Result |
|---|---|
| Completed 2026–27 games | 43; October 20–25 |
| Player logs | 1,191 player/game records across 43 games |
| Team snapshots | All six current-season dates through October 25 are stored |
| Consensus closing totals and spreads | 43 games; scores validated against player logs |
| Rebuilt training dataset | 17,776 rows; 2012-11-02 through 2026-10-25 |
| Current-season training rows | 27 |

Each completed game was cross-checked against ESPN daily scoreboards for October 20–26. No completed games were missing or extra. October 26 had nine scheduled games and none completed at verification.

## Availability snapshots

- injuries: latest stored publication time `2026-10-26T17:15:00+00:00`.
- lineups: latest stored publication time `2026-10-26T17:15:47.464556+00:00`.
- rosters: latest stored publication time `2026-10-26T17:15:47.853971+00:00`.

## Remaining gaps and exclusions

- SBR returned no 2026–27 boards. Executable sportsbook moneyline payout prices remain unavailable for all 43 current-season completed games. Consensus fair probabilities were not converted into payout prices.
- Sixteen opening games remain outside the training table because the existing feature builder requires both teams in a prior snapshot within seven days. Their raw scores, player logs, and consensus lines are stored. No same-game results were substituted for pregame features.
- Twelve archived official injury snapshots were imported for October 20–25: the prior-day and same-day 11:45 p.m. reports for each slate. Features retain the publication cutoff and exclude later reports. Historical lineup/roster snapshots remain unavailable from the configured ingestion workflow; only today’s lineup/roster snapshots were refreshed. Teams that have not submitted an official report retain incomplete coverage.
- No models were retrained.

## Code repair and verification

Fixed consensus importing: pandas renames tuple fields that begin with an underscore, so accessing `row._date` crashed. The importer now uses `game_day`, with regression checks for dates, chronological rest calculations, missing moneyline prices, and duplicate games.

Forty-nine relevant tests passed. All five databases passed SQLite quick checks; canonical foreign-key checks returned no violations. The rebuilt dataset has no duplicate date/home/away keys. All 27 current-season training rows have full player-history coverage. Injury coverage remains incomplete where the official report was not submitted before the historical prediction cutoff.

Sources: existing NBA stats collectors; SportsDataverse ESPN player archive; [attributed consensus dataset](https://huggingface.co/datasets/minmaxxxer/nba-odds-2026-27); [NBA official injury reports](https://official.nba.com/nba-injury-report-2026-27-season/); ESPN public daily scoreboards.
