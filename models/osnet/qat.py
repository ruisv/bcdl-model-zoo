#!/usr/bin/env python3
"""[3/4] QAT self-distillation for OSNet-AIN: make the int8 embedding match FP32.

WHY SELF-DISTILLATION AND NOT REID TRAINING. The defect is precisely stated —
the quantized tower's embedding has drifted off the float one (cosine ~0.47 where
it should be ~0.99), which collapses Market-1501 Rank-1 from 84.8 to 51.4. That
is a function-matching problem, not an identity-learning problem, so the FP32
model is the only supervision needed and any pile of person crops (crops.py) will
do. No identity labels, no triplet mining, no ReID training set. This is why the
recipe is hours, not days.

WHY int8 PTQ CANNOT BE RESCUED without this (all measured on the board, not
node_info): AIN 51.4% / plain-BN 6.6% / IBN 5.2% Rank-1; three calibration
methods, calibration set 64->400 crops (400 was slightly WORSE). Counter-intuitive
finding worth keeping: InstanceNorm is a quantization ASSET, not a liability —
AIN's 51.4% dwarfs plain-BN's 6.6%. Changing backbone does not help either
(ResNet50 float cross-domain Rank-1 46.3 vs OSNet-AIN 70.1).

WHY THIS IS TRUSTWORTHY OFFLINE. The QAT-prepared graph reproduces the board's
failure almost exactly (cosine 0.5049 in torch vs 0.475 measured on the board),
so the number printed here tracks what the board will do, and iteration costs
seconds instead of a compile-deploy-evaluate cycle.

Market-1501 is never trained on — it stays the held-out benchmark.

Runs inside the OpenExplorer GPU container (needs torch + horizon_plugin_pytorch):
    ./compile.sh --gpu 2 python qat.py --weights osnet_ain_x1_0_msmt17.pth \\
                                       --crops crops/ --out qat_osnet_ain.pth
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import EmbeddingHead, load  # noqa: E402

from horizon_plugin_pytorch.march import March, set_march  # noqa: E402
from horizon_plugin_pytorch.quantization import (  # noqa: E402
    FakeQuantState,
    QuantStub,
    prepare,
    set_fake_quantize,
)
from horizon_plugin_pytorch.quantization.qconfig_template import (  # noqa: E402
    default_calibration_qconfig_setter,
    default_qat_qconfig_setter,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
H, W = 256, 128


class CropSet(Dataset):
    """Person crops, preprocessed exactly as the runtime does.

    The only augmentation is a horizontal flip: the student must match the
    teacher on the inputs it will actually see, and aggressive augmentation would
    spend capacity on pixels the deployment never produces.
    """

    def __init__(self, root: Path, flip: bool = True):
        self.files = sorted(root.glob("*.jpg"))
        if not self.files:
            raise FileNotFoundError(f"no crops in {root}")
        self.flip = flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(str(self.files[i]))
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.flip and np.random.rand() < 0.5:
            img = img[:, ::-1]
        rgb = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


class QatNet(torch.nn.Module):
    """EmbeddingHead behind a QuantStub, which is what makes the graph exportable.

    Without an explicit QuantStub the prepared model happily trains — but the
    export then refuses it ("the input should be QTensor"), because nothing in
    the graph turns the float input into a quantized one. The stub is also not
    just plumbing: it is where INPUT quantization error enters. Training without
    it means the student never sees that error, so its cosine is optimistic
    relative to what the board will do (measured cost of adding it: 0.9782 ->
    0.9778, small but the honest number).
    """

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.head = head

    def forward(self, x):
        return self.head(self.quant(x))


def deq(t):
    """Plain float tensor out of whatever the prepared model returned."""
    return t.dequantize() if hasattr(t, "dequantize") else t.as_subclass(torch.Tensor)


@torch.no_grad()
def eval_cosine(student, teacher, loader, device, max_batches=20):
    """Mean cosine between the quantized and float embeddings — the number that
    predicts the board result. Runs in the plugin's VALIDATION state, which is
    the one that mimics deployment."""
    was_training = student.training
    student.eval()
    set_fake_quantize(student, FakeQuantState.VALIDATION)
    total, n = 0.0, 0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        s = deq(student(x)).flatten(1)
        t = teacher(x).flatten(1)
        total += F.cosine_similarity(s, t).sum().item()
        n += x.shape[0]
    if was_training:
        student.train()
        set_fake_quantize(student, FakeQuantState.QAT)
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="osnet_ain_x1_0")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("qat_osnet_ain.pth"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--calib-batches", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_march(March.NASH_M)
    torch.manual_seed(0)

    dataset = CropSet(args.crops)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=True)
    eval_loader = DataLoader(CropSet(args.crops, flip=False), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)
    print(f"{len(dataset)} crops, {len(loader)} steps/epoch, device={device}")

    teacher = EmbeddingHead(load(args.arch, args.weights)).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = prepare(QatNet(EmbeddingHead(load(args.arch, args.weights))).eval(),
                      example_inputs=(torch.zeros(1, 3, H, W),),
                      qconfig_setter=(default_calibration_qconfig_setter,)).to(device)

    # Phase 1 — calibration. Observers collect activation ranges from real crops.
    # Starting QAT from PTQ thresholds rather than from nothing is what keeps the
    # fine-tune short.
    student.eval()
    set_fake_quantize(student, FakeQuantState.CALIBRATION)
    with torch.no_grad():
        for i, x in enumerate(eval_loader):
            if i >= args.calib_batches:
                break
            student(x.to(device))
    print(f"calibrated on {args.calib_batches * args.batch_size} crops; "
          f"cosine now {eval_cosine(student, teacher, eval_loader, device):.4f}")

    # Phase 2 — QAT. Re-prepare with the QAT qconfig, carrying the calibrated
    # state across, then train the weights against the float teacher.
    calibrated_state = student.state_dict()
    student = prepare(QatNet(EmbeddingHead(load(args.arch, args.weights))).eval(),
                      example_inputs=(torch.zeros(1, 3, H, W),),
                      qconfig_setter=(default_qat_qconfig_setter,)).to(device)
    missing, unexpected = student.load_state_dict(calibrated_state, strict=False)
    print(f"carried calibration into QAT model "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

    student.train()
    set_fake_quantize(student, FakeQuantState.QAT)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))

    best = -1.0
    for epoch in range(args.epochs):
        t0, run = time.perf_counter(), 0.0
        for step, x in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                t = teacher(x).flatten(1)
            s = deq(student(x)).flatten(1)
            # Cosine is the loss because cosine is what the tracker consumes:
            # the embedding is L2-normalized before use, so its magnitude is
            # not part of the contract and should not be part of the objective.
            loss = (1.0 - F.cosine_similarity(s, t)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.5f}")

        cos = eval_cosine(student, teacher, eval_loader, device)
        print(f"epoch {epoch}: mean loss {run / len(loader):.5f}  "
              f"eval cosine-vs-fp32 {cos:.4f}  ({time.perf_counter() - t0:.0f}s)")
        if cos > best:
            best = cos
            torch.save({"state_dict": student.state_dict(), "cosine": cos,
                        "arch": args.arch}, args.out)
            print(f"  saved {args.out} (best cosine {best:.4f})")

    print(f"done. best cosine-vs-fp32 {best:.4f} (PTQ baseline was ~0.50)")


if __name__ == "__main__":
    main()
