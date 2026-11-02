"""360 features built only from each shot's own 360 frame, never from its event freeze
frame.

frame_features (src/data/features.py) is reused unchanged; only the player positions it
is given come from the 360 data instead of the freeze frame. src/data/features.py itself
is untouched.
"""

import pandas as pd

from src.data.features import FEATURE_SETS, FREEZE_FRAME, GOAL_X, frame_features

# the freeze-frame features from the 360 shot frame, for the 360 study. Fit on train and
# scored on the valid shots that have a 360 frame, adding them to base (base_360) lowers
# log-loss by about 0.0036 (LightGBM and logistic) and moves LightGBM's calibration
# slope from 1.11 to 1.00. On top of base_ff, which already has the event's freeze
# frame, they don't help (LightGBM 0.2675 -> 0.2690).
S360_FF = [f"s360ff_{c}" for c in FREEZE_FRAME]

# each of FEATURE_SETS with the 360 columns added, never the 360 columns on their own:
# geometry_360 (4 + 18), base_360 (44 + 18), base_ff_360 (62 + 18). load_dataset
# (src/dataset.py) accepts these names too. A set added to FEATURE_SETS gets a _360
# version.
FEATURE_SETS_360 = {
    f"{name}_360": cols + S360_FF for name, cols in FEATURE_SETS.items()
}


def shot_frame_360_features(shots, players_360):
    """frame_features computed from each shot's own 360 frame (columns S360_FF),
    measured from the shot's event location like the freeze-frame features. The
    shooter (the frame's actor) is left out, as he is from a freeze frame. NaN for
    shots without their own 360 frame. shots needs shot_id, x, y. players_360:
    data/interim/360_players.parquet"""
    p0 = players_360[players_360["rel"] == 0]
    rows_of = p0.groupby("shot_id").indices
    x, y, team, keeper, actor = (
        p0[c].to_numpy() for c in ["x", "y", "shooting_team", "keeper", "actor"]
    )
    rows = []
    for shot_id, sx, sy in zip(
        shots["shot_id"], shots["x"].clip(upper=GOAL_X), shots["y"], strict=True
    ):
        i = rows_of.get(shot_id)
        if i is None:
            rows.append({})
            continue
        i = i[~actor[i]]
        rows.append(frame_features(sx, sy, x[i], y[i], team[i], keeper[i]))
    f = pd.DataFrame(rows, index=shots.index, columns=FREEZE_FRAME).astype(float)
    f.columns = S360_FF
    return f


def add_360_columns(shots, interim):
    """shots (with shot_id, x, y) plus the 360 columns (S360_FF), from the 360 players
    table in interim (the data/interim folder). NaN for shots without their own 360
    frame"""
    players = pd.read_parquet(interim / "360_players.parquet")
    players = players[players["shot_id"].isin(shots["shot_id"])]
    return pd.concat([shots, shot_frame_360_features(shots, players)], axis=1)
