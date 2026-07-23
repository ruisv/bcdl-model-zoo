#!/usr/bin/env bash
# Run a command inside the D-Robotics OpenExplorer GPU container, with this model
# directory mounted at /ws and exactly one GPU handed in.
#
# Unlike the PTQ models, OSNet does not compile with `hb_compile -c config.yaml`.
# Its shipped build is QAT (horizon_plugin_pytorch + hbdk4), which is Python run
# inside the same container. So this script is a generic runner rather than an
# hb_compile wrapper, and the three steps are ordinary commands through it:
#
#   # 1. QAT self-distillation -> qat checkpoint (needs a GPU)
#   ./compile.sh --gpu 2 python qat.py \
#       --weights osnet_ain_x1_0_msmt17.pth --crops crops/ --out qat_osnet_ain.pth
#
#   # 2. QAT -> .hbm (the shipped build)
#   ./compile.sh --gpu 2 python deploy.py \
#       --weights osnet_ain_x1_0_msmt17.pth --qat qat_osnet_ain.pth --crops crops/ \
#       --out out/osnet_ain_qat_nashm_256x128.hbm
#
#   # 3. the REJECTED int8 PTQ build, to reproduce the collapse (see README)
#   ./compile.sh --gpu 2 hb_compile -c config.yaml
#
# --gpu is an nvidia-smi index handed to docker as `device=N`, NOT
# CUDA_VISIBLE_DEVICES (which numbers devices differently and can silently land
# on a pre-Ampere card that fails deep in calibration). The card must be Ampere
# or newer.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${OE_IMAGE:-registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0}

GPU=0
if [ "${1:-}" = "--gpu" ]; then
  GPU=$2; shift 2
fi
[ $# -gt 0 ] || { echo "usage: $0 [--gpu N] <command...>"; sed -n '2,20p' "$0"; exit 1; }

# --user keeps outputs owned by the caller; HOME must point somewhere writable
# inside the mount or the toolchain fails on startup. `"$*"` so the whole command
# (with its own flags) runs under the login shell.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --gpus "device=$GPU" \
  -e HOME=/ws -e MPLCONFIGDIR=/ws/.mpl \
  --shm-size=15g \
  -v "$HERE":/ws -w /ws \
  "$IMAGE" -lc "$*"
