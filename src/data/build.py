"""Build data/shots.parquet from the downloaded StatsBomb files.

Step 1 extracts the shots, freeze frames and 360 frames from every match into data/interim/.
Step 2 adds the features and splits, and writes data/shots.parquet and
predictions/statsbomb_xg.parquet. Pass --features-only to run step 2 alone.
"""

import json
import sys

import numpy as np
import pandas as pd

from src.data.download import RAW
from src.data.extract import extract_360, extract_shots
from src.data.features import FEATURE_SETS, NOT_FEATURES, add_features
from src.dataset import DATA, SHOTS, save_predictions

INTERIM = DATA / "interim"

# Held-out competitions, every season of each; everything else is train
VALID = ["Ligue 1", "UEFA Women's Euro", "Frauen Bundesliga"]
TEST = ["Serie A", "UEFA Euro", "Women's World Cup", "NWSL"]


def read(rel_path):
    return json.loads((RAW / rel_path).read_bytes())


def extract_all():
    shots, freeze_frames, frames_360, players_360 = [], [], [], []
    competitions = read("competitions.json")
    for n, comp in enumerate(competitions, start=1):
        for match in read(f"matches/{comp['competition_id']}/{comp['season_id']}.json"):
            match_id = match["match_id"]
            events = read(f"events/{match_id}.json")
            metadata = match.get("metadata") or {}
            info = {
                "match_id": match_id,
                "competition_name": comp["competition_name"],
                "season_name": comp["season_name"],
                "competition_gender": comp["competition_gender"],
                "match_date": match["match_date"],
                "data_version": metadata.get("data_version"),
                "shot_fidelity_version": metadata.get("shot_fidelity_version"),
            }
            match_shots, match_frames = extract_shots(events)
            shots += [{**info, **shot} for shot in match_shots]
            freeze_frames += match_frames
            if (RAW / f"three-sixty/{match_id}.json").exists():
                try:
                    f, p = extract_360(events, read(f"three-sixty/{match_id}.json"))
                except (
                    ValueError
                ) as err:  # three-sixty/3845506.json is corrupt upstream
                    print(f"  skipping 360 data of match {match_id}: {err}")
                    continue
                frames_360 += f
                players_360 += p
        print(
            f"  {n}/{len(competitions)} competition-seasons, {len(shots):,} shots",
            flush=True,
        )

    INTERIM.mkdir(parents=True, exist_ok=True)
    for table, rows in [
        ("shots", shots),
        ("freeze_frames", freeze_frames),
        ("360_frames", frames_360),
        ("360_players", players_360),
    ]:
        pd.DataFrame(rows).to_parquet(INTERIM / f"{table}.parquet", index=False)


def build_table():
    shots = pd.read_parquet(INTERIM / "shots.parquet")
    freeze_frames = pd.read_parquet(INTERIM / "freeze_frames.parquet")
    frames_360 = pd.read_parquet(
        INTERIM / "360_frames.parquet", columns=["shot_id", "rel"]
    )

    df = add_features(shots, freeze_frames)
    df["has_360"] = df["shot_id"].isin(
        frames_360.loc[frames_360["rel"] == 0, "shot_id"]
    )

    unknown = set(VALID + TEST) - set(df["competition_name"])
    if unknown:
        raise ValueError(f"not in the data, check the spelling: {unknown}")
    df["split"] = np.select(
        [df["competition_name"].isin(TEST), df["competition_name"].isin(VALID)],
        ["test", "valid"],
        "train",
    )
    for name, cols in FEATURE_SETS.items():
        if set(cols) & set(NOT_FEATURES):
            raise ValueError(
                f"feature set {name} contains the label or the provider's xG"
            )

    df.to_parquet(SHOTS, index=False)
    save_predictions(
        df["shot_id"], df["statsbomb_xg"].clip(1e-6, 1 - 1e-6), "statsbomb_xg"
    )

    summary = (
        df[df["is_penalty"] == 0]
        .groupby("split")
        .agg(
            matches=("match_id", "nunique"),
            shots=("shot_id", "size"),
            goals=("is_goal", "sum"),
            goal_rate=("is_goal", "mean"),
            provider_xg=("statsbomb_xg", "mean"),
            with_360=("has_360", "sum"),
        )
    )
    print(
        f"wrote {SHOTS}: {len(df):,} shots\n\nnon-penalty shots by split:\n{summary.round(3)}"
    )


if __name__ == "__main__":
    if "--features-only" not in sys.argv:
        extract_all()
    build_table()
