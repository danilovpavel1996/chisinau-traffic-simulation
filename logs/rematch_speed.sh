#!/bin/zsh
# matcher odometer = speed.py: photo alignment, map position, 3D view and dashboard again, per drive
set -o pipefail
cd /Users/danilovpavel/Documents/traffic_chisinau/chisinau-commute-upgrade
for d in 2026_10_26_10_02 2026_10_23_10_06 2026_10_24_17_08 2026_10_23_19_26 2026_10_24_15_48; do
  if [ -f data/video/processed/$d/vpr/verified.parquet ] || [ $d = 2026_10_24_15_48 ]; then
    if [ $d != 2026_10_24_15_48 ]; then echo "== $d: photo alignment ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.vpr $d || exit 1; fi
  fi
  echo "== $d: map position ($(date +%H:%M))"; .venv-video/bin/python -W ignore -m src.video.map_layer $d || exit 1
  echo "== $d: 3D view ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.surround $d --objects || exit 1
done
nice -n 10 .venv-video/bin/python -W ignore -c "from src.video import drives, pipeline; pipeline.run_step('dashboard', drives.load('2026_10_26_10_02'))"
echo "REMATCH DONE ($(date +%H:%M))"
