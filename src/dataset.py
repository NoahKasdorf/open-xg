from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import FEATURE_SETS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SHOTS = DATA / "shots.parquet"  
PREDICTIONS = ROOT / "predictions"

# for slicing results and comparing with the provider; never a model input
META = [
    "shot_id",
    "match_id",
    "split",
    "competition_name",
    "season_name",
    "competition_gender",
    "data_version",
    "team_name",
    "player_name",
    "shot_type",
    "body_part",
    "play_pattern",
    "is_penalty",
    "has_360",
    "statsbomb_xg",
]


def load_dataset(split="train", feature_set="base_ff", penalties=False):
    """Return X, y, meta for "train", "valid" or "test"

    feature_set: "geometry", "base" or "base_ff" (see FEATURE_SETS in src/data/features.py).
    Penalties are left out by default: every model, the provider's included, gives them
    about the same xG, so they would only flatter the metrics.
    Keeper columns are NaN for the 0.1% of shots with no keeper in the freeze frame.
    """
    df = pd.read_parquet(SHOTS)
    splits = [split] if isinstance(split, str) else list(split)
    df = df[
        df["split"].isin(splits) & (penalties | (df["is_penalty"] == 0))
    ].reset_index(drop=True)
    return df[FEATURE_SETS[feature_set]], df["is_goal"].astype(int), df[META]


def save_predictions(shot_id, p_goal, name):
    """Save predictions/<name>.parquet, the only hand-off from a model to evaluation.

    Evaluation joins on shot_id to get labels, splits and slices, so nothing else is needed.
    """
    df = pd.DataFrame(
        {"shot_id": np.asarray(shot_id), "p_goal": np.asarray(p_goal, dtype=float)}
    )
    if df["shot_id"].duplicated().any():
        raise ValueError("duplicate shot_id")
    if not df["p_goal"].between(0, 1, inclusive="neither").all():
        raise ValueError(
            "p_goal must be strictly between 0 and 1 (clip it, e.g. to [1e-6, 1 - 1e-6])"
        )
    PREDICTIONS.mkdir(exist_ok=True)
    df.to_parquet(PREDICTIONS / f"{name}.parquet", index=False)
