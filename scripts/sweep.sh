#!/usr/bin/env bash
#
# Train, evaluate and report every experiment in a sweep, then build one comparison
# page over all of them.
#
#   ./scripts/sweep.sh                          # every configs/experiments/*.yaml except smoke
#   ./scripts/sweep.sh baseline logmel crnn     # named experiments only
#   EPOCHS=40 ./scripts/sweep.sh                # override epochs for the whole sweep
#   EXTRA="--set train.batch_size=128" ./scripts/sweep.sh
#
# Each experiment becomes one run named after its config file, so reports/<name>/ and
# checkpoints/<name>/ never collide. Re-running the sweep overwrites those runs.
#
# A failing experiment does not abort the sweep: it is recorded and the remaining
# experiments still run, because a six-hour sweep that dies on its second config and
# throws away the rest is worse than one that reports a gap.

set -uo pipefail

EXPERIMENTS_DIR="configs/experiments"
EPOCHS="${EPOCHS:-}"
EXTRA="${EXTRA:-}"
LOG_DIR="${LOG_DIR:-reports/sweep_logs}"

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOG_DIR"

# Which experiments? Explicit arguments win; otherwise every config except smoke,
# which exists to be fast rather than to be compared against anything.
if [ "$#" -gt 0 ]; then
  names=("$@")
else
  names=()
  for path in "$EXPERIMENTS_DIR"/*.yaml; do
    name="$(basename "$path" .yaml)"
    [ "$name" = "smoke" ] && continue
    names+=("$name")
  done
fi

if [ "${#names[@]}" -eq 0 ]; then
  echo "No experiments found in $EXPERIMENTS_DIR" >&2
  exit 1
fi

# "baseline" is the defaults with no experiment file — include it if asked for by name,
# or whenever the sweep is running unfiltered, so every comparison has a control.
if [ "$#" -eq 0 ]; then
  names=("baseline" "${names[@]}")
fi

echo "Sweep: ${names[*]}"
echo "Logs:  $LOG_DIR"
echo

declare -a failed=()
start_all=$SECONDS

for name in "${names[@]}"; do
  config_arg=()
  if [ "$name" != "baseline" ]; then
    config_path="$EXPERIMENTS_DIR/$name.yaml"
    if [ ! -f "$config_path" ]; then
      echo "!! $name: no such config ($config_path), skipping" >&2
      failed+=("$name")
      continue
    fi
    config_arg=(-c "$config_path")
  fi

  overrides=(--set "train.run_name=$name")
  [ -n "$EPOCHS" ] && overrides+=(--set "train.epochs=$EPOCHS")

  echo "── $name ──────────────────────────────────────────────"
  start=$SECONDS
  # shellcheck disable=SC2086  # EXTRA is deliberately word-split into flags
  if uv run nvcr run "${config_arg[@]}" "${overrides[@]}" $EXTRA 2>&1 | tee "$LOG_DIR/$name.log"; then
    echo "   ok in $((SECONDS - start))s"
  else
    echo "!! $name failed after $((SECONDS - start))s — see $LOG_DIR/$name.log" >&2
    failed+=("$name")
  fi
  echo
done

echo "── summary ────────────────────────────────────────────"
uv run nvcr summary
echo
echo "Sweep finished in $((SECONDS - start_all))s"

if [ "${#failed[@]}" -gt 0 ]; then
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
