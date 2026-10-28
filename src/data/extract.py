"""Turn one match's StatsBomb events into shot rows, freeze-frame rows and 360 rows"""

REBOUND_WINDOW_S = 3  # a shot within 5 s of the same team's previous shot is a rebound
N_PREV_360 = 5  # 360 frames kept per shot: the shot's own plus up to 5 earlier events


def name(obj, key):
    """StatsBomb stores categoricals as {"id": .., "name": ..}; return the name or None."""
    value = (obj or {}).get(key)
    return value["name"] if value else None

def seconds(timestamp):
    """'00:12:34.567' to 754.567"""
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def extract_shots(events):
    """Return (shots, freeze_frames) as lists of dicts. Shoot-out kicks are skipped"""
    events = sorted(events, key=lambda e: e["index"])
    by_id = {e["id"]: e for e in events}
    possession_start = {}  # (period, possession) -> time of its first event
    last_shot = {}  # (team, period) -> time of the team's previous shot
    last_action = {}  # (team, period, possession) -> type of the team's previous event
    shots, freeze_frames = [], []

    for e in events:
        etype, team, period, poss = name(e, "type"), name(e, "team"), e["period"], e["possession"]
        t = seconds(e["timestamp"])
        possession_start.setdefault((period, poss), t)
        if etype != "Shot":
            if etype != "Ball Receipt*":
                last_action[(team, period, poss)] = etype
            continue
        if period == 5:  # penalty shoot-out, not match play
            continue

        s = e["shot"]
        gap = t - last_shot[(team, period)] if (team, period) in last_shot else None
        kp = by_id.get(s.get("key_pass_id"), {})  # the pass that set up the shot, if any
        p = kp.get("pass", {})

        shots.append({
            "shot_id": e["id"],
            "period": period,
            "minute": e["minute"],
            "team_name": team,
            "player_name": name(e, "player"),
            "play_pattern": name(e, "play_pattern"),
            "x": e["location"][0],
            "y": e["location"][1],
            "under_pressure": e.get("under_pressure", False),
            "shot_type": name(s, "type"),
            "body_part": name(s, "body_part"),
            "technique": name(s, "technique"),
            "first_time": s.get("first_time", False),
            "one_on_one": s.get("one_on_one", False),
            "open_goal": s.get("open_goal", False),
            "aerial_won": s.get("aerial_won", False),
            "follows_dribble": s.get("follows_dribble", False),
            "redirect": s.get("redirect", False),
            "prev_event_type": last_action.get((team, period, poss)),
            "is_rebound": gap is not None and gap <= REBOUND_WINDOW_S,
            "poss_duration_s": t - possession_start[(period, poss)],
            "kp_present": bool(kp),
            "kp_type": name(p, "type"),  # None = ordinary open-play pass
            "kp_height": name(p, "height"),
            "kp_body_part": name(p, "body_part"),
            "kp_cross": p.get("cross", False),
            "kp_cut_back": p.get("cut_back", False),
            "kp_switch": p.get("switch", False),
            
            # older data flags through balls; data v1.1.0 uses technique "Through Ball"
            "kp_through_ball": p.get("through_ball", False) or name(p, "technique") == "Through Ball",
            "kp_length": p.get("length", 0.0),
            "kp_under_pressure": kp.get("under_pressure", False),
            "statsbomb_xg": s["statsbomb_xg"],  # the provider's xG: compared against
            "outcome": name(s, "outcome"),
            "is_goal": name(s, "outcome") == "Goal",
        })

        # the other players at the moment of the shot
        for player in s.get("freeze_frame", []):
            freeze_frames.append({
                "shot_id": e["id"],
                "x": player["location"][0],
                "y": player["location"][1],
                "teammate": player["teammate"],
                "keeper": name(player, "position") == "Goalkeeper",
            })
        last_shot[(team, period)] = t
        last_action[(team, period, poss)] = "Shot"

    return shots, freeze_frames


def extract_360(events, frames):
    """Return (frames, players) for every shot: the 360 frame of the shot (rel = 0) and of
        up to N_PREV_360 earlier events in the same possession (rel = -1, -2, ...)
    
    A 360 frame is oriented for the team of ITS event, and `teammate` is relative to that
    event's actor. Frames of events by the defending team (a pressure, a duel) are therefore
    mirrored here so every row is from the shooting team's view, attacking x = 120
    """
        
    by_id = {f["event_uuid"]: f for f in frames}
    events = sorted(events, key=lambda e: e["index"])
    frame_rows, player_rows = [], []
    
    for i, shot in enumerate(events):
        if name(shot, "type") != "Shot" or shot["period"] == 5:
            continue
        chain = [shot]  # the shot, then earlier events of the same possession with a frame
        for e in reversed(events[:i]):
            if len(chain) > N_PREV_360 or e["possession"] != shot["possession"] or e["period"] != shot["period"]:
                break
            if e["id"] in by_id:
                chain.append(e)

        rel = 0
        for e in chain:
            frame = by_id.get(e["id"])
            if frame is None:  # only the shot itself can be missing a frame
                rel -= 1
                continue
            flip = name(e, "team") != name(shot, "team")
            area = frame.get("visible_area") or []
            if flip:
                area = [120 - v if k % 2 == 0 else 80 - v for k, v in enumerate(area)]
            frame_rows.append({
                "shot_id": shot["id"],
                "rel": rel,
                "event_type": name(e, "type"),
                "dt_s": seconds(shot["timestamp"]) - seconds(e["timestamp"]),
                "flipped": flip,
                "visible_area": [float(v) for v in area],  # polygon x1, y1, x2, y2, ...
            })
            for player in frame["freeze_frame"]:
                x, y = player["location"]
                player_rows.append({
                    "shot_id": shot["id"],
                    "rel": rel,
                    "x": 120 - x if flip else x,
                    "y": 80 - y if flip else y,
                    "shooting_team": player["teammate"] != flip,
                    "actor": player["actor"],
                    "keeper": player["keeper"],
                })
            rel -= 1
        return frame_rows, player_rows