"""Download the StatsBomb open data we use into data/raw/"""

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

def fetch(rel_path, missing_ok=False):
    """Download one file unless we have it. Return False for an allowed 404."""
    path = RAW / rel_path
    if path.exists():
        return True
    
    try:
        with urllib.request.urlopen(f"{BASE_URL}/{rel_path}", timeout=120) as response:
            content = response.read()
    except urllib.error.HTTPError as err:
        if err.code == 404 and missing_ok:
            return False
        raise
        
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(content)
    tmp.replace(path)
    return True


def main():
    fetch("competitions.json")
    todo = []
    for comp in json.loads((RAW / "competitions.json").read_bytes()):
        matches = f"matches/{comp['competition_id']}/{comp['season_id']}.json"
        fetch(matches)
        for match in json.loads((RAW / matches).read_bytes()):
            todo.append((f"events/{match['match_id']}.json", False))
            if comp["match_available_360"]:  # some matches in a 360 season still have no file
                todo.append((f"three-sixty/{match['match_id']}.json", True))

    print(f"{len(todo)} files: {RAW}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, _ in enumerate(pool.map(lambda job: fetch(*job), todo), start=1):
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("done")

if __name__ == "__main__":
    main()
