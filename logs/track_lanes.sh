#!/bin/zsh
# vehicles bound to one lane per track: 3D view and dashboard for every drive on the map
set -o pipefail
cd /Users/danilovpavel/Documents/traffic_chisinau/chisinau-commute-upgrade
for d in 2026_10_24_17_08 2026_10_23_10_06 2026_10_26_10_02 2026_10_23_19_26 2026_10_24_15_48; do
  echo "== $d: 3D view ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.surround $d --objects || exit 1
done
nice -n 10 .venv-video/bin/python -W ignore -c "from src.video import drives, pipeline; pipeline.run_step('dashboard', drives.load('2026_10_26_10_02'))"
echo "TRACK LANES DONE ($(date +%H:%M))"
