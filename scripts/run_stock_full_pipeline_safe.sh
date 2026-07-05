#!/usr/bin/env bash
set -euo pipefail

# Safer wrapper around run_stock_full_pipeline.sh for laptops/desktops that become
# unstable under sustained GPU load. This does not guarantee an exact 80% GPU
# utilization cap, because TensorFlow/ROCm does not expose a simple portable
# percentage cap. Instead it reduces workload, avoids full GPU memory preallocation
# where TensorFlow supports it, and keeps runs resumable through the normal
# pipeline checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Conservative defaults. Override any of these on the command line if needed.
export TRIALS="${TRIALS:-10}"
export RANDOM_EPOCHS="${RANDOM_EPOCHS:-20}"
export WALK_EPOCHS="${WALK_EPOCHS:-15}"
export GPU="${GPU:-0}"

# Limit CPU thread pressure as well; useful when GPU training also stresses the system.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

echo "[SAFE MODE] TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"
echo "[SAFE MODE] TRIALS=$TRIALS RANDOM_EPOCHS=$RANDOM_EPOCHS WALK_EPOCHS=$WALK_EPOCHS GPU=$GPU"
echo "[SAFE MODE] OMP_NUM_THREADS=$OMP_NUM_THREADS TF_NUM_INTRAOP_THREADS=$TF_NUM_INTRAOP_THREADS TF_NUM_INTEROP_THREADS=$TF_NUM_INTEROP_THREADS"
echo

exec bash "$SCRIPT_DIR/run_stock_full_pipeline.sh" "$@"
