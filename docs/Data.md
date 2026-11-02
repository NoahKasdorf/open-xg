# The data

This covers the shot table we're all working from: where it comes from, how it gets built, what's in it, and a few things that will trip you up if you don't know about them. If anything here doesn't match what you see in the data, let me know.

## Getting it

Everyone builds the data themselves by running the pipeline. Nothing in `data/` goes in git: it's in `.gitignore`, both because StatsBomb's terms don't allow passing the data on and because the raw files are about 15 GB. From the repo root:

```
python -m src.data.download     # StatsBomb JSON into data/raw/ (~15 GB)
python -m src.data.build        # builds data/shots.parquet
```

In Git Bash, `./run.sh all` does both, or `./run.sh download` and `./run.sh data` run them one at a time. The build takes a few minutes.

That gives you `data/shots.parquet`, which is all you need for modelling and evaluation. The 360 study also uses `360_frames.parquet` and `360_players.parquet`, which the build writes to `data/interim/`.

## How it's built

The source is StatsBomb's open data on GitHub (hudl/open-data). It's laid out as a list of competitions, one match list per season, one events file per match (every pass, carry, shot and so on, about 3,500 per game), and a separate 360 file for matches that have 360 coverage.

`download.py` copies what we need into `data/raw/` with the same layout. It skips files it already has, so if it dies halfway you just run it again. We don't download the lineup files since there's nothing in them we use.

`build.py` does the rest in two steps. First it goes through every match and uses `extract.py` to pull the shots out of the events: the shot itself, the pass that set it up, the shot's freeze frame, a bit of context from the events before it (was it a rebound, how long the possession had lasted), and the 360 frames around it. These go into `data/interim/`. Then `features.py` turns those rows into model features, every shot gets its train/valid/test label, and the result is written to `data/shots.parquet`. StatsBomb's own xG gets saved as `predictions/statsbomb_xg.parquet` too, so evaluation has something to compare against right away.

The first step is the slow one because it reads 12 GB of JSON. If you only change a feature or the splits, `python -m src.data.build --features-only` (or `./run.sh features`) reruns just the second step, which is much quicker.

## What's in it

One row per shot in match play. Penalty shoot-out kicks are dropped (363 of them) because they aren't part of the match.

| | |
|---|---:|
| competitions / seasons / matches | 24 / 80 / 3,961 (1958 to 2025) |
| shots | 100,864, including 1,118 penalties |
| non-penalty shots | 99,746 |
| goals among them | 10,233 (10.3%) |
| women's football | 34% of shots |
| shots with a freeze frame | all non-penalty shots |
| shots with their own 360 frame | 10,233 non-penalty |

`load_dataset()` leaves penalties out by default. StatsBomb gives every penalty the same xG (0.7835; 76.3% actually went in), so they don't tell us anything about a model and would just make the numbers look better.

## Splits

The program doc asks for evaluation on held-out competitions, so we split by whole competitions, all seasons of each. Nothing from a test competition is ever seen in training, tuning or calibration. The lists are at the top of `src/data/build.py`.

| split | competitions | matches | shots | goal rate | StatsBomb xG | with 360 |
|---|---|---:|---:|---:|---:|---:|
| train | the other 17 (La Liga, FA WSL, Premier League, World Cup, ...) | 2,560 | 64,649 | 10.7% | 9.9% | 3,091 |
| valid | Ligue 1, UEFA Women's Euro, Frauen Bundesliga | 629 | 15,234 | 10.4% | 9.7% | 3,106 |
| test | Serie A, UEFA Euro, Women's World Cup, NWSL | 772 | 19,863 | 8.7% | 8.9% | 4,036 |

Test has a men's league, a men's tournament, a women's tournament and a women's league, and every split has 360 shots. Train fits models, valid is for tuning and fitting the calibrator, and test gets scored once, at the end, for every model at the same time. Test has a lower goal rate than train, but StatsBomb's xG drops with it, so it's the mix of shots in those competitions and not a problem with the split.

## Features

All coordinates are in StatsBomb's pitch units: the pitch is 120 by 80, the team shooting always attacks toward x = 120, and the goal runs from y = 36 to y = 44. StatsBomb draws every pitch at that size, so "yards" are close to real yards but not exact.

| group | columns | what they are |
|---|---|---|
| location | `distance`, `angle`, `dx`, `dy_abs` | distance to the centre of the goal, how wide the goal looks from the shot (0.64 radians from the penalty spot), distance to the goal line, distance off the middle |
| body part, technique | `bp_head`, `bp_other`, `tech_*` (6) | 0/1 columns; feet and a normal strike are the baseline |
| shot type, play pattern | `st_*` (3), `pp_*` (8) | free kick, corner, kick-off; how the possession started (corner, counter, throw-in, ...) |
| StatsBomb flags | `under_pressure`, `first_time`, `one_on_one`, `open_goal`, `aerial_won`, `follows_dribble`, `redirect` | straight from the data |
| rebound | `is_rebound` | the same team shot within the last 3 seconds |
| the pass before the shot | `kp_present`, `kp_cross`, `kp_cut_back`, `kp_through_ball`, `kp_switch`, `kp_under_pressure`, `kp_high`, `kp_low`, `kp_set_piece`, `kp_headed`, `kp_length` | details of the pass that set the shot up, all 0 if there wasn't one (29% of shots) |
| context | `prev_carry`, `poss_duration_s` | whether the shooter's team carried the ball just before; seconds since the possession started |
| freeze frame: players | `ff_available`, `ff_n_opponents`, `ff_n_teammates`, `n_def_in_triangle`, `n_att_in_triangle`, `n_def_within_2y`, `n_def_within_5y`, `n_def_goal_side`, `nearest_def_dist` | who was around the ball. "Triangle" means the shooting cone between the ball and the two posts |
| freeze frame: keeper | `gk_found`, `gk_dist_to_shot`, `gk_dist_to_goal_line`, `gk_dist_to_goal_center`, `gk_offset_from_shot_line`, `gk_in_triangle`, `gk_goal_side` | where the keeper was; the offset is how far the keeper stood from the line between the ball and the middle of the goal |
| freeze frame: visible goal | `goal_visible_frac`, `goal_visible_frac_no_gk` | share of the goal the shooter had a clear line to, with and without the keeper. Defenders block half a yard either side of where they stand, the keeper a full yard |

These come in named sets so every result can say exactly what it used (`FEATURE_SETS` in `src/data/features.py`):

- `geometry` (4): location only, the classic baseline
- `base` (44): everything except the freeze frame
- `base_ff` (62): everything
- `geometry_360` (22), `base_360` (62), `base_ff_360` (80): each of the three plus 18 columns from the shot's own 360 frame, for the 360 study (`FEATURE_SETS_360` in `src/study360/features.py`, see "The 360 tables" below)

The only missing values are in the keeper columns, for the 100 shots where no keeper shows up in the freeze frame, and `nearest_def_dist` for 8 shots with no defender in it. Gradient boosting (XGBoost, LightGBM) handles that on its own. For logistic regression, fill them in using the training data only.

The `_360` sets have many more: their 360 columns are blank for every shot without its own 360 frame (about 90% of shots), and the keeper ones also for the 5% of 360 shots where the camera didn't see the keeper. The same applies: LightGBM handles it, and for logistic regression fill them in from the training data.

## Other columns

Besides the features, each row has ids and context for slicing results (`shot_id`, `match_id`, competition, season, `competition_gender`, `match_date`, `data_version`, team, player), the raw values the features came from (`x`, `y`, `shot_type`, `body_part`, `kp_type` and so on), and `split`, `is_penalty` and `has_360`.

`is_goal` is the label. `outcome` and `statsbomb_xg` are there to check things and compare against, and must never be used as features. Neither is anything recorded after the shot was hit, which is why `deflected` and where the shot ended up aren't in the table at all.

## The 360 tables

These are for the 360 study. A 360 frame is a snapshot of every player visible on the broadcast at the moment of an event. Unlike the freeze frame it exists for most events, not just shots, so you can see what happened in the seconds before a shot. Players are anonymous though: you only get whether they're on the shooting team, whether they're the keeper, and whether they're the player making that event.

`360_frames.parquet` has one row per frame: `shot_id`, `rel` (0 for the shot itself, then -1, -2 and so on for earlier events in the same possession, up to 5), `event_type`, `dt_s` (seconds before the shot), `flipped` and `visible_area` (the corners of the part of the pitch the camera could see). `360_players.parquet` has one row per player per frame: `shot_id`, `rel`, `x`, `y`, `shooting_team`, `actor` and `keeper`. Join them on `shot_id` and `rel`.

A few things to know. StatsBomb records each frame from the point of view of whoever made that event, so frames from the defending team's events (about 21% of the earlier ones, mostly pressures and duels) have already been mirrored to the shooting team's view. The earlier frames are a median of 0.2, 1.2, 2.1, 3.1 and 4.0 seconds before the shot, and 8,458 shots have all five. About 4% of shots in 360 matches have no frame of their own, but their earlier frames are still there. On average about 15 players are visible per frame, and anyone outside `visible_area` is unknown, not absent. StatsBomb's 360 spec also says `visible_area` can be empty when the camera wasn't on the pitch, and a player can now and then sit outside it because they were placed by hand. Finally, the 360 frame and the freeze frame for the same shot are separate recordings and typically put the keeper about a yard apart, so don't read small differences between them as information.

The `_360` feature sets turn each shot's own 360 frame (`rel` = 0) into the 18 freeze-frame features above, computed the same way (`frame_features`) and named with an `s360ff_` prefix, like `s360ff_n_def_in_triangle`. They're measured from the shot's event location, the shooter is left out as in a freeze frame, and only players the camera saw count. They're computed when you load a `_360` set rather than stored in `shots.parquet` (`src/study360/features.py`), so that load takes about 2 seconds instead of a fraction of one. `meta["has_360"]` marks the shots that have them. On valid, they improve on `base` for the 360 shots but add nothing on top of `base_ff`, which already has the freeze frame.

## Things to watch out for

StatsBomb's data has changed over the years. About 95% of shots are data version 1.1.0. That version widened what counts as `under_pressure` (StatsBomb's spec says duels, dribbles and the like now pick it up even without a pressure event), so it's flagged on 25% of shots against 13 to 18% in the older versions. It's worth breaking results down by `data_version`.

Barcelona takes 39% of La Liga's shots, since the open data includes most of their matches from 2004 to 2021. That's 8% of every shot in the table.

`one_on_one` and `open_goal` are tagged by hand from video rather than measured, and whoever tags them can see how the shot ended. They're kept, but report a version of the model without them too.

The 360 file for match 3845506 (England v Sweden, Women's Euro 2022) is corrupt on StatsBomb's side, so that match has shots but no 360 data. Six shots are recorded just past the goal line (x = 120.1 to 120.5) and get treated as x = 120.

About 95% of shots come from matches with `shot_fidelity_version` 2, where StatsBomb records shot and freeze-frame locations more precisely. The rest use the standard, coarser locations, so small differences in position mean less in those matches.

## Using it

```python
from src.dataset import load_dataset, save_predictions

X, y, meta = load_dataset("train", feature_set="base_ff")
X_valid, y_valid, meta_valid = load_dataset("valid", feature_set="base_ff")

model.fit(X, y)
save_predictions(meta_valid["shot_id"], model.predict_proba(X_valid)[:, 1], "gbm_base_ff")
```

`meta` holds the ids, competition, StatsBomb's xG and so on for slicing, never for the model itself. Every model saves a predictions file like this, and that's the only thing evaluation needs. Run Python from the repo root so the `src.` imports work.

The 360 study works the same way with a `_360` set, like `load_dataset("train", feature_set="base_360")`. To score only the shots with a 360 frame, keep the rows where `meta["has_360"]` is true.

Data: StatsBomb open data, under their Public Data User Agreement (`LICENSE.pdf` in hudl/open-data). In short: it's for research, not commercial use, we can't pass the data on to anyone outside the agreement, and anything we publish has to credit StatsBomb and carry their logo (it's in their Media Pack). They also ask everyone using it to register at statsbomb.com/resource-centre.