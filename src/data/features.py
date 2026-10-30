"""Shot features. All coordinates in StatsBomb yards, shooting toward the goal at x = 120"""

import numpy as np
import pandas as pd

GOAL_X, GOAL_Y, POST_LOW, POST_HIGH = (
    120.0,
    40.0,
    36.0,
    44.0,
)  # the goal is 8 yards wide
DEF_RADIUS, GK_RADIUS = (
    0.5,
    1.0,
)  # yards of goal a defender / the keeper blocks around himself
GOAL_SAMPLES = np.linspace(
    POST_LOW, POST_HIGH, 33
)  # points across the goal mouth, every 0.25 yd


# --- geometry -----------------------------------------------------------------------------


def distance(x, y):
    """Yards from (x, y) to the centre of the goal."""
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def goal_angle(x, y):
    """Angle (radians) the goal mouth subtends at (x, y): 0.64 at the penalty spot"""
    return np.abs(
        np.arctan2(POST_HIGH - y, GOAL_X - x) - np.arctan2(POST_LOW - y, GOAL_X - x)
    )


def in_cone(px, py, sx, sy):
    """Is the point P inside the triangle shot -> post -> post (the shooting cone)?"""
    px, py = np.asarray(px, float), np.asarray(py, float)

    def side(ax, ay, bx, by):  # which side of the edge A->B the point P is on
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax)

    d1 = side(sx, sy, GOAL_X, POST_LOW)
    d2 = side(GOAL_X, POST_LOW, GOAL_X, POST_HIGH)
    d3 = side(GOAL_X, POST_HIGH, sx, sy)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def blocked_lines(px, py, radius, sx, sy):
    """For each goal sample point: does any player (a disc of `radius`) cut the straight
    line from the ball to it? px, py: players' positions; returns 33 booleans"""
    gy = GOAL_SAMPLES[None, :]
    px, py = np.asarray(px, float)[:, None], np.asarray(py, float)[:, None]
    dx, dy = GOAL_X - sx, gy - sy
    length2 = np.maximum(
        dx**2 + dy**2, 1e-12
    )  # a shot on the goal line has zero-length lines
    t = np.clip(
        ((px - sx) * dx + (py - sy) * dy) / length2, 0, 1
    )  # closest point on each line
    return (np.hypot(px - (sx + t * dx), py - (sy + t * dy)) < radius).any(axis=0)


# --- features from the shot event ----------------------------------------------------------

# values that get their own 0/1 column; the unlisted value is the baseline
CATEGORIES = {
    "body_part": (
        "bp",
        ["Head", "Other"],
    ),  # feet are the baseline (we don't know the preferred foot)
    "technique": (
        "tech",
        ["Volley", "Half Volley", "Lob", "Backheel", "Diving Header", "Overhead Kick"],
    ),
    "shot_type": (
        "st",
        ["Free Kick", "Corner", "Kick Off"],
    ),  # open play is the baseline
    "play_pattern": (
        "pp",
        [
            "From Corner",
            "From Free Kick",
            "From Throw In",
            "From Counter",
            "From Goal Kick",
            "From Keeper",
            "From Kick Off",
            "Other",
        ],
    ),
}
FLAGS = [
    "under_pressure",
    "first_time",
    "one_on_one",
    "open_goal",
    "aerial_won",
    "follows_dribble",
    "redirect",
    "is_rebound",
]
SET_PIECE_PASSES = ["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"]


def column(prefix, value):
    return f"{prefix}_{value.lower().replace(' ', '_')}"


def shot_features(shots):
    x = shots["x"].clip(upper=GOAL_X)  # 6 shots are recorded at x = 120.1
    f = pd.DataFrame(
        {
            "distance": distance(x, shots["y"]),
            "angle": goal_angle(x, shots["y"]),
            "dx": GOAL_X - x,
            "dy_abs": (shots["y"] - GOAL_Y).abs(),
        },
        index=shots.index,
    )
    for col, (prefix, values) in CATEGORIES.items():
        for v in values:
            f[column(prefix, v)] = (shots[col] == v).astype(int)
    for flag in FLAGS:
        f[flag] = shots[flag].astype(int)
    for flag in [
        "kp_present",
        "kp_cross",
        "kp_cut_back",
        "kp_through_ball",
        "kp_switch",
        "kp_under_pressure",
    ]:
        f[flag] = shots[flag].astype(int)
    f["kp_high"] = (shots["kp_height"] == "High Pass").astype(int)
    f["kp_low"] = (shots["kp_height"] == "Low Pass").astype(int)
    f["kp_set_piece"] = shots["kp_type"].isin(SET_PIECE_PASSES).astype(int)
    f["kp_headed"] = (shots["kp_body_part"] == "Head").astype(int)
    f["kp_length"] = shots["kp_length"]
    f["prev_carry"] = (shots["prev_event_type"] == "Carry").astype(int)
    f["poss_duration_s"] = shots["poss_duration_s"].clip(lower=0)
    f["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)
    return f


# --- features from the shot's freeze frame -------------------------------------------------


def frame_features(sx, sy, x, y, teammate, keeper):
    """Where the keeper and defenders were when one shot was taken.
    x, y, teammate, keeper: one entry per player in the frame"""
    defender = ~teammate & ~keeper
    gk = ~teammate & keeper
    to_ball = np.hypot(x - sx, y - sy)
    to_goal = distance(x, y)
    cone = in_cone(x, y, sx, sy)
    f = {
        "ff_available": 1,
        "ff_n_opponents": (~teammate).sum(),
        "ff_n_teammates": teammate.sum(),
        "n_def_in_triangle": (defender & cone).sum(),
        "n_att_in_triangle": (teammate & cone).sum(),
        "n_def_within_2y": (defender & (to_ball <= 2)).sum(),
        "n_def_within_5y": (defender & (to_ball <= 5)).sum(),
        "n_def_goal_side": (defender & (to_goal < distance(sx, sy))).sum(),
        "nearest_def_dist": to_ball[defender].min() if defender.any() else np.nan,
        "gk_found": int(gk.any()),
    }
    if gk.any():
        k = np.flatnonzero(gk)[
            to_goal[gk].argmin()
        ]  # if two keepers are listed, the one in goal
        kx, ky = x[k], y[k]
        ball_to_goal = distance(sx, sy)
        f["gk_dist_to_shot"] = to_ball[k]
        f["gk_dist_to_goal_line"] = GOAL_X - kx
        f["gk_dist_to_goal_center"] = to_goal[k]
        # how far the keeper stands off the line from the ball to the centre of the goal
        cross = abs((GOAL_X - sx) * (ky - sy) - (GOAL_Y - sy) * (kx - sx))
        f["gk_offset_from_shot_line"] = (
            cross / ball_to_goal if ball_to_goal > 0 else to_ball[k]
        )
        f["gk_in_triangle"] = float(cone[k])
        f["gk_goal_side"] = float(to_goal[k] < ball_to_goal)
    by_defenders = blocked_lines(x[defender], y[defender], DEF_RADIUS, sx, sy)
    by_keeper = blocked_lines(x[gk], y[gk], GK_RADIUS, sx, sy)
    f["goal_visible_frac"] = 1 - (by_defenders | by_keeper).mean()
    f["goal_visible_frac_no_gk"] = 1 - by_defenders.mean()
    return f


def freeze_frame_features(shots, freeze_frames):
    """One row of frame_features per shot (rows without a frame get ff_available = 0)"""
    rows_of = freeze_frames.groupby(
        "shot_id"
    ).indices  # shot_id -> row numbers in freeze_frames
    x, y = freeze_frames["x"].to_numpy(), freeze_frames["y"].to_numpy()
    teammate, keeper = (
        freeze_frames["teammate"].to_numpy(),
        freeze_frames["keeper"].to_numpy(),
    )
    rows = []
    for shot_id, sx, sy in zip(
        shots["shot_id"], shots["x"].clip(upper=GOAL_X), shots["y"], strict=True
    ):
        i = rows_of.get(shot_id)
        if i is None:
            rows.append({"ff_available": 0})
        else:
            rows.append(frame_features(sx, sy, x[i], y[i], teammate[i], keeper[i]))
    f = pd.DataFrame(rows, index=shots.index, columns=FREEZE_FRAME)
    counts = [c for c in FREEZE_FRAME if c.startswith(("ff_", "n_")) or c == "gk_found"]
    f[counts] = f[counts].fillna(0).astype(int)
    return f


def add_features(shots, freeze_frames):
    """The extracted shot rows with every feature added (feature columns replace raw ones)"""
    features = pd.concat(
        [shot_features(shots), freeze_frame_features(shots, freeze_frames)], axis=1
    )
    return pd.concat(
        [shots.drop(columns=features.columns, errors="ignore"), features], axis=1
    )


# --- feature sets --------------------------------------------------------------------------

GEOMETRY = ["distance", "angle", "dx", "dy_abs"]
BASE = (
    GEOMETRY
    + [column(p, v) for p, values in CATEGORIES.values() for v in values]
    + FLAGS
    + [
        "kp_present",
        "kp_cross",
        "kp_cut_back",
        "kp_through_ball",
        "kp_switch",
        "kp_under_pressure",
        "kp_high",
        "kp_low",
        "kp_set_piece",
        "kp_headed",
        "kp_length",
        "prev_carry",
        "poss_duration_s",
    ]
)
FREEZE_FRAME = [
    "ff_available",
    "ff_n_opponents",
    "ff_n_teammates",
    "n_def_in_triangle",
    "n_att_in_triangle",
    "n_def_within_2y",
    "n_def_within_5y",
    "n_def_goal_side",
    "nearest_def_dist",
    "gk_found",
    "gk_dist_to_shot",
    "gk_dist_to_goal_line",
    "gk_dist_to_goal_center",
    "gk_offset_from_shot_line",
    "gk_in_triangle",
    "gk_goal_side",
    "goal_visible_frac",
    "goal_visible_frac_no_gk",
]
FEATURE_SETS = {"geometry": GEOMETRY, "base": BASE, "base_ff": BASE + FREEZE_FRAME}

# never model inputs: the label and the provider's prediction
NOT_FEATURES = ["is_goal", "outcome", "statsbomb_xg"]
