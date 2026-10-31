#!/usr/bin/env bash
# Full manuscript settings, including all reviewer analyses and 40 year-block resamples.
# Usage: bash code/run_revision.sh [output_dir] [main_storage_csv] [later_snapshot_csv]
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$SCRIPT_DIR
if [ ! -d "$PROJECT_DIR/figure_sources" ]; then
  PROJECT_DIR=$(dirname -- "$SCRIPT_DIR")
fi
OUT=${1:-$PROJECT_DIR/results}
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
STORAGE=${2:-$SCRIPT_DIR/data/mastr_storage/2025-10-01/storage_by_state.csv}
SNAP=${3:-$SCRIPT_DIR/data/mastr_storage/2026-10-31/storage_by_state.csv}
PYTHON=${PYTHON:-python}
STORAGE=$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$STORAGE")
SNAP=$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$SNAP")
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd -- "$SCRIPT_DIR"
"$PYTHON" REACT_ESR_reproducible.py "$OUT" --mode all --storage-file "$STORAGE" > "$OUT/main_run.log" 2>&1 &
MAIN_PID=$!
"$PYTHON" REACT_revision.py "$OUT" --analysis m012,support_diag,dependence,overlap,socio --storage-file "$STORAGE" > "$OUT/revision_run.log" 2>&1 &
REVISION_PID=$!
"$PYTHON" REACT_revision.py "$OUT" --analysis weights,support,support_extra,storage,climate --storage-file "$STORAGE" --snapshot-file "$SNAP" > "$OUT/sensitivity_run.log" 2>&1 &
SENSITIVITY_PID=$!
"$PYTHON" REACT_revision.py "$OUT" --analysis bootstrap --n-boot 40 --jobs "${JOBS:-4}" --storage-file "$STORAGE" > "$OUT/bootstrap_run.log" 2>&1 &
BOOTSTRAP_PID=$!
trap 'kill "$MAIN_PID" "$REVISION_PID" "$SENSITIVITY_PID" "$BOOTSTRAP_PID" 2>/dev/null || true' EXIT
wait "$MAIN_PID"
wait "$REVISION_PID"
wait "$SENSITIVITY_PID"
wait "$BOOTSTRAP_PID"
trap - EXIT
"$PYTHON" REACT_revision.py "$OUT" --analysis permutation --storage-file "$STORAGE"
"$PYTHON" REACT_revision_summary.py "$OUT"
"$PYTHON" REACT_manuscript_numbers.py "$OUT" > "$OUT/manuscript_numbers.txt"
"$PYTHON" REACT_data_checks.py --storage-file "$STORAGE" --use-storage-file-in-panel > "$OUT/data_checks.md"
"$PYTHON" finalize_results.py "$OUT" --storage-file "$STORAGE" --snapshot-file "$SNAP"
echo "Complete: $OUT/manuscript_numbers.txt (full logs in the same directory)"
