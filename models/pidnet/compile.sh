#!/usr/bin/env bash
# [3/3] Compile ONNX -> BPU .hbm via the D-Robotics OpenExplorer container.
#
# The model directory is mounted at /ws, so every path inside the config is
# relative to it and the same config works on any host.
#
# Usage:
#   ./compile.sh [--config config.yaml] [--gpu N]
#
#   --gpu : physical GPU index for PTQ calibration. Needs Ampere or newer.
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
    *) echo "unknown arg: $1"; sed -n '2,14p' "$0"; exit 1;;
  esac
done
[ -f "$HERE/$CONFIG" ] || { echo "no such config: $HERE/$CONFIG"; exit 1; }

# --user keeps outputs owned by the caller; HOME must be writable inside the
# mount. GPU is selected by docker (`device=N`, an nvidia-smi index), NOT
# CUDA_VISIBLE_DEVICES, which numbers differently and can land on a pre-Ampere
# card that fails deep in calibration.
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
