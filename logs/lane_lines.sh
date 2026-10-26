#!/bin/zsh
# lane counts again with solid / dashed lines, all drives on the map; then validation on OSM-tagged streets
set -o pipefail
cd /Users/danilovpavel/Documents/traffic_chisinau/chisinau-commute-upgrade
rm -f data/video/lane_counts.parquet
for d in 2026_10_26_10_02 2026_10_23_10_06 2026_10_24_17_08 2026_10_23_19_26 2026_10_24_15_48; do
  echo "== $d: lane counts ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.lane_count $d 2>&1 | grep -v "surround objects" || exit 1
done
echo "LANE LINES DONE ($(date +%H:%M))"
