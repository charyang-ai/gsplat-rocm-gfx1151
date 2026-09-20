#!/usr/bin/env bash
# Run a command in the gfx1151 gsplat image with the GPU and the persistent state attached.
#
#   ./run.sh                                   # default CMD (the rasterize smoke test)
#   ./run.sh python tests/smoke_test.py rasterize
#   ./run.sh python tests/isect_correctness_test.py --render --exact
#   ./run.sh bash                              # interactive
#
# Three of the mounts below are not optional if the numbers are meant to mean anything:
#
#   tune/    TunableOp's rocBLAS winners and MIOpen's convolution cache. The image sets
#            PYTORCH_TUNABLEOP_TUNING=1, so without this an ephemeral container re-tunes
#            from scratch and the first iterations of every benchmark measure the tuner.
#   results/ benchmarks write table1_results.json and friends into the workspace; a --rm
#            container would take them with it.
#   data/    the Mip-NeRF 360 scenes, far too large to bake into the image.
#
# --device=/dev/kfd --device=/dev/dri and the render/video groups are what make the GPU
# visible; seccomp=unconfined is required because ROCm's userspace issues ioctls that the
# default Docker seccomp profile blocks.
set -euo pipefail

IMAGE="${IMAGE:-gsplat-rocm:gfx1151}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

mkdir -p "$ROOT/tune/tunableop" "$ROOT/tune/miopen" "$ROOT/results" "$ROOT/data" \
         "$ROOT/caches/torch"

# Interactive only when this is a terminal, so the script is still usable from a pipe.
TTY_FLAGS=()
[[ -t 0 && -t 1 ]] && TTY_FLAGS=(-it)

exec docker run --rm "${TTY_FLAGS[@]}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -v "$ROOT/tune:/opt/gsplat/.tune" \
  -v "$ROOT/results:/opt/gsplat/results" \
  -v "$ROOT/data:/opt/gsplat/data" \
  -v "$ROOT/caches/torch:/root/.cache/torch" \
  -w /opt/gsplat \
  "$IMAGE" "$@"
