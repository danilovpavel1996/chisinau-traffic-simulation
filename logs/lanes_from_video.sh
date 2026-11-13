#!/bin/zsh
# lane counts from the video for streets OSM has none for; the kerb in the sideways lock; then
# map position, 3D view and dashboard for every drive on the map
set -o pipefail
cd /Users/danilovpavel/Documents/traffic_chisinau/chisinau-commute-upgrade
D=(2026_10_26_10_02 2026_10_23_10_06 2026_10_24_17_08 2026_10_23_19_26 2026_10_24_15_48)
for d in $D; do echo "== $d: lane counts ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.lane_count $d 2>&1 | grep -v "surround objects" || exit 1; done
for d in $D; do
  echo "== $d: map position ($(date +%H:%M))"; .venv-video/bin/python -W ignore -m src.video.map_layer $d || exit 1
  echo "== $d: 3D view ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.surround $d --objects 2>&1 | grep -v "surround objects" || exit 1
done
nice -n 10 .venv-video/bin/python -W ignore -c "from src.video import drives, pipeline; pipeline.run_step('dashboard', drives.load('2026_10_26_10_02'))"
echo "LANES DONE ($(date +%H:%M))"
