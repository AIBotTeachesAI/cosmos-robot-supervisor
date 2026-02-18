#!/usr/bin/env bash
# make_slowmo_and_sliding_clips.sh
#
# Run from repo root:
#   cd ~/cosmos-supervisor
#   bash src/approach2_exploration/video_utils/make_slowmo_and_sliding_clips.sh

set -euo pipefail

# --- Portable repo-root resolution (no usernames) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# --- Only path changes vs original ---
IN_ROOT="${REPO_ROOT}/videos/inputs"
OUT_ROOT="${REPO_ROOT}/videos/approach2_exploration/derived_inputs"

# Scenarios to process (same)
SCENARIOS=("success" "obstruction" "push_down" "miss_grasp")

# Slow-mo and clipping params (same)
SLOWMO_PTS="2.0"        # 0.5x speed => setpts=2.0*PTS
SLOWMO_FPS="30"
CLIP_WIN="4"            # seconds
CLIP_STRIDE="2"         # seconds
CLIP_FPS="8"            # sampling-friendly
SCALE="scale=-2:480"

mkdir -p "${OUT_ROOT}/slowmo" "${OUT_ROOT}/clips_slow_sliding"

for scenario in "${SCENARIOS[@]}"; do
  in_dir="${IN_ROOT}/${scenario}"
  [[ -d "$in_dir" ]] || continue

  mkdir -p "${OUT_ROOT}/slowmo/${scenario}"
  mkdir -p "${OUT_ROOT}/clips_slow_sliding/${scenario}"

  shopt -s nullglob
  videos=("${in_dir}"/*.mp4)
  shopt -u nullglob

  if [[ ${#videos[@]} -eq 0 ]]; then
    echo "[${scenario}] No mp4 found in ${in_dir}, skipping."
    continue
  fi

  echo "[${scenario}] Found ${#videos[@]} videos."

  for v in "${videos[@]}"; do
    base="$(basename "$v" .mp4)"
    slow="${OUT_ROOT}/slowmo/${scenario}/${base}_slow.mp4"
    outdir="${OUT_ROOT}/clips_slow_sliding/${scenario}"

    echo "  -> slowmo: ${scenario}/${base}.mp4"
    ffmpeg -y \
      -i "$v" \
      -filter:v "setpts=${SLOWMO_PTS}*PTS,fps=${SLOWMO_FPS},${SCALE}" \
      -an \
      "$slow" >/dev/null 2>&1

    dur="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$slow")"

    # last start time so that start+win <= dur
    last="$(python3 - <<PY
import math
dur=float("$dur")
win=float("$CLIP_WIN")
print(max(0, int(math.floor(dur-win))))
PY
)"

    echo "  -> sliding clips: ${scenario}/${base}_slow (dur=${dur}s, last_start=${last}s)"
    for s in $(seq 0 "$CLIP_STRIDE" "$last"); do
      ffmpeg -y -ss "$s" -t "$CLIP_WIN" -i "$slow" \
        -an -vf "fps=${CLIP_FPS},${SCALE}" \
        "${outdir}/${base}_${s}s.mp4" >/dev/null 2>&1
    done
  done
done

echo "Done."
echo "Slowmo is in: ${OUT_ROOT}/slowmo/{success,obstruction,push_down,miss_grasp}/"
echo "Clips are in: ${OUT_ROOT}/clips_slow_sliding/{success,obstruction,push_down,miss_grasp}/"

