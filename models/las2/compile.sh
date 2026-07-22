#!/usr/bin/env bash
# [3/3] 编译 ONNX → BPU .hbm (int16 全量化, march nash-m)。
# 在转换主机上用 D-Robotics OpenExplorer docker 镜像 (v3.7.0)。
#
# 用法:
#   ./compile.sh --workdir DIR --onnx REL.onnx --calib REL_CALIB_DIR \
#                [--prefix NAME] [--march nash-m] [--jobs N] [--gpu 0]
#
#   --workdir : 挂载到容器 /ws 的根目录; --onnx / --calib 都是相对它的路径
#   --calib   : 该目录下需有 left/ 和 right/ 两个子目录(gen_calib.py 的输出)
#   --gpu     : 物理 GPU 号(校准用; 显存不足时换一张空闲卡, 见 README)
#
# 产物: <workdir>/<prefix>_out/<prefix>.hbm
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${OE_IMAGE:-registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0}

usage() { sed -n '2,15p' "$0"; exit 1; }

WORKDIR=""; ONNX=""; CALIB=""; PREFIX=""; MARCH=nash-m; JOBS=$(nproc 2>/dev/null || echo 8); GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --workdir) WORKDIR=$2; shift 2;;
    --onnx)    ONNX=$2;    shift 2;;
    --calib)   CALIB=$2;   shift 2;;
    --prefix)  PREFIX=$2;  shift 2;;
    --march)   MARCH=$2;   shift 2;;
    --jobs)    JOBS=$2;    shift 2;;
    --gpu)     GPU=$2;     shift 2;;
    *) echo "unknown arg: $1"; usage;;
  esac
done
[ -n "$WORKDIR" ] && [ -n "$ONNX" ] && [ -n "$CALIB" ] || usage
WORKDIR=$(cd "$WORKDIR" && pwd)
PREFIX=${PREFIX:-$(basename "$ONNX" .onnx)_int16_nashm}
OUTDIR_REL=${PREFIX}_out
YAML="$WORKDIR/${PREFIX}.yaml"

sed -e "s|__ONNX__|${ONNX}|" \
    -e "s|__MARCH__|${MARCH}|" \
    -e "s|__OUTDIR__|${OUTDIR_REL}|" \
    -e "s|__PREFIX__|${PREFIX}|" \
    -e "s|__CALL__|/ws/${CALIB}/left|" \
    -e "s|__CALR__|/ws/${CALIB}/right|" \
    -e "s|__JOBS__|${JOBS}|" \
    "$HERE/config.yaml.template" > "$YAML"

echo "[compile] image=$IMAGE"
echo "[compile] yaml=$YAML  (march=$MARCH jobs=$JOBS gpu=$GPU)"
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" --shm-size=15g \
  -v "$WORKDIR":/ws -w /ws "$IMAGE" -lc "hb_compile -c $(basename "$YAML")"
echo "[compile] DONE -> $WORKDIR/$OUTDIR_REL/${PREFIX}.hbm"
