#!/bin/sh
# ./run.sh <command>   (on Windows, run it in Git Bash)
#   all        download + data: the whole pipeline
#   download   fetch the StatsBomb files into data/raw/ 
#   data       build data/shots.parquet from data/raw/ 
#   features   rebuild the features and shots table only 
set -eu
cd "$(dirname "$0")"  # always run from the repo root, so `import src` works
PY="${PY:-$(command -v python || command -v python3 || true)}"
[ -n "$PY" ] || { echo "no python found: activate your virtual environment first" >&2; exit 1; }

case "${1:-}" in
  all)      "$PY" -m src.data.download && "$PY" -m src.data.build ;;
  download) "$PY" -m src.data.download ;;
  data)     "$PY" -m src.data.build ;;
  features) "$PY" -m src.data.build --features-only ;;
  *)        sed -n '2,6p' "$0" | cut -c3- ; exit 1 ;;
esac
