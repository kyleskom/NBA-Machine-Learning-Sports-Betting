"""Named feature contracts shared by training and serving (no ML dependencies)."""

TEAM_STATS = (
    "GP", "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
    "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD",
    "PTS", "PLUS_MINUS",
)
TEAM_FEATURES = TEAM_STATS + tuple(f"{name}_RANK" for name in TEAM_STATS)
MATCH_FEATURES = TEAM_FEATURES + tuple(f"{name}.1" for name in TEAM_FEATURES)
REST_FEATURES = ("Days-Rest-Home", "Days-Rest-Away")
XGB_ML_FEATURES = MATCH_FEATURES + REST_FEATURES
NN_ML_FEATURES = MATCH_FEATURES + REST_FEATURES
XGB_UO_FEATURES = MATCH_FEATURES + ("OU",) + REST_FEATURES
NN_UO_FEATURES = MATCH_FEATURES + REST_FEATURES + ("OU",)


def select_features(frame, columns):
    """Select by name, never by the incoming frame's incidental order."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    if not frame.columns.is_unique:
        raise ValueError("Duplicate feature column names are not supported")
    return frame.loc[:, list(columns)].copy()
