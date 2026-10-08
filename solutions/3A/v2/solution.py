"""Task 3A v2 - Vietnamese forest bird calls: end-to-end pipeline (reproduces the submitted CSVs).

Steps (modules in this folder):
  0. common.load_cache  : RMS-normalised mel-power caches (64 mel / n_fft 512 for the light SED-CNN,
                          128 mel / n_fft 512 for the ImageNet backbones)
  1. round 1 (hold-out) : models trained on synthetic soundscapes built from 80 % of the focal clips;
                          their two soundscape-like validation AUCs (SV synthetic, RV real test backgrounds) are logged
  2. pseudo-labels ps1  : 0.6 * ResNet18 round-1 + 0.4 * v1 ensemble probabilities on the 800 unlabelled test clips
  3. round 2 (full)     : all focal clips + real test clips as backgrounds (50 %) with their soft pseudo-labels
  4. round 3            : pseudo-labels ps2 from the round-2 ResNet (+ round-1 models), one more ResNet18
  5. predict.py         : family-weighted probability blend, 4x time-shift TTA -> public/private_submission.csv here

GPU auto-detected (trained on a shared GTX 1050 Ti: ~3 min per SED-CNN, ~25 min per ResNet18 run).
usage: python solution.py [--quick]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE  # noqa: E402

PY = sys.executable
V1_DIR = Path(os.environ.get("V1_3A", HERE.parent))  # v1 submission CSVs (used for RV background choice + pseudo-labels)

R18 = ["--arch", "resnet18", "--mels", "128", "--lr", "1e-3"]
EB0 = ["--arch", "efficientnet_b0", "--mels", "128", "--lr", "2e-3"]
SED = ["--arch", "sed", "--mels", "64"]

ROUND1 = {  # hold-out models (validated on SV / RV)
    "A0_sed": SED + ["--epochs", "40"],
    "A1_r18": R18 + ["--epochs", "40"],
    "A2_eb0": EB0 + ["--epochs", "30"],
}
ROUND2 = {  # all focal clips + real test backgrounds with ps1 soft labels
    "F1_r18_ps": R18 + ["--epochs", "40", "--full", "--pseudo", "PS1", "--seed", "1"],
    "F3_r18_ps": R18 + ["--epochs", "40", "--full", "--pseudo", "PS1", "--seed", "3"],
    "FS1_sed": SED + ["--epochs", "40", "--full", "--pseudo", "PS1", "--seed", "5"],
}
ROUND3 = {  # same, with the round-2 teacher ps2
    "F3_r18_ps2": R18 + ["--epochs", "40", "--full", "--pseudo", "PS2", "--seed", "3"],
}
# family weights from the RV blend study of the hold-out models (ResNet : EfficientNet : SED-CNN = 2 : 1 : 1)
FINAL = {"F1_r18_ps": 2 / 3, "F3_r18_ps": 2 / 3, "F3_r18_ps2": 2 / 3, "A2_eb0": 1.0, "FS1_sed": 1.0}


def run(*args: str) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=HERE, check=True, env=os.environ.copy())


def v1_probs() -> np.ndarray:
    f = CACHE / "v1_probs.npy"
    if not f.exists():
        p = np.concatenate([pd.read_csv(V1_DIR / f"{s}_submission.csv").iloc[:, 1:].to_numpy(np.float32)
                            for s in ("public", "private")])
        np.save(f, p)
    return np.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (2 epochs each)")
    a = ap.parse_args()

    def args_of(spec: list[str]) -> list[str]:
        spec = list(spec)
        if a.quick:
            spec[spec.index("--epochs") + 1] = "2"
            spec += ["--per_epoch", "64", "--tta", "1"]
        return spec

    v1 = v1_probs()
    t = lambda n: np.load(CACHE / f"{n}_test.npy")  # noqa: E731
    for name, spec in ROUND1.items():
        run("train.py", "--name", name, *args_of(spec))
    ps1, ps2 = CACHE / "ps1.npy", CACHE / "ps2.npy"
    np.save(ps1, (0.6 * t("A1_r18") + 0.4 * v1).astype(np.float32))
    for name, spec in ROUND2.items():
        run("train.py", "--name", name, *[str(ps1) if s == "PS1" else s for s in args_of(spec)])
    np.save(ps2, (0.5 * t("F1_r18_ps") + 0.3 * t("A1_r18") + 0.2 * t("A2_eb0")).astype(np.float32))
    for name, spec in ROUND3.items():
        run("train.py", "--name", name, *[str(ps2) if s == "PS2" else s for s in args_of(spec)])
    run("predict.py", *FINAL.keys(), "--weights", *map(str, FINAL.values()))


if __name__ == "__main__":
    main()
