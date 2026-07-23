#!/usr/bin/env bash
# [3/3] Compile ONNX -> BPU .hbm via the D-Robotics OpenExplorer container.
#
# The model directory is mounted at /ws, so every path inside the config is
# relative to it and the same config works on any host.
#
# Usage:
#   ./compile.sh [--config config.yaml] [--gpu N]
#
#   --config : which config to compile (only config.yaml here)
#   --gpu    : physical GPU index used for PTQ calibration. Needs Ampere or
#              newer — the calibration CUDA kernels fail with
#              cudaErrorInvalidDevice on older cards.
#
# Compile parallelism is `jobs` under compiler_parameters in the config, not a
# command-line flag: hb_compile has no --jobs option.
#
# Output: <model dir>/<working_dir>/<output_model_file_prefix>.hbm
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${OE_IMAGE:-registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0}

CONFIG=config.yaml
GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2;;
    --gpu)    GPU=$2;    shift 2;;
    *) echo "unknown arg: $1"; sed -n '2,15p' "$0"; exit 1;;
  esac
done
[ -f "$HERE/$CONFIG" ] || { echo "no such config: $HERE/$CONFIG"; exit 1; }

# --user keeps the outputs owned by the caller rather than root; HOME must point
# somewhere writable inside the mount or the toolchain fails on startup.
# GPU is selected by docker (`device=N`, an nvidia-smi index) rather than by
# CUDA_VISIBLE_DEVICES inside the container. Those are NOT the same numbering:
# CUDA defaults to CUDA_DEVICE_ORDER=FASTEST_FIRST, so it reorders by compute
# capability while nvidia-smi orders by PCI bus. Passing --gpus all with
# CUDA_VISIBLE_DEVICES=<nvidia-smi index> therefore silently selects a
# different card, and landing on a pre-Ampere one fails deep into calibration
# with `cudaErrorInvalidDevice: invalid device ordinal`. Handing the container
# exactly one device removes the ambiguity.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --gpus "device=$GPU" \
  -e HOME=/ws -e MPLCONFIGDIR=/ws/.mpl \
  --shm-size=15g \
  -v "$HERE":/ws -w /ws \
  "$IMAGE" -lc "hb_compile -c $CONFIG"

PREFIX=$(awk -F'"' '/output_model_file_prefix/{print $2}' "$HERE/$CONFIG")
WORKDIR=$(awk -F'"' '/working_dir/{print $2}' "$HERE/$CONFIG")
echo "[compile] DONE -> $HERE/${WORKDIR#./}/${PREFIX}.hbm"
