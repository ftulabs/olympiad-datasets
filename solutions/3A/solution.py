"""Task 3A - Vietnamese forest bird calls: end-to-end pipeline.

Steps (each step is a module in this folder):
  1. common.load_cache   : log-mel power cache for train / public / private (RMS-normalised clips)
  2. train.py            : round 1 - SED-CNN trained on synthetic soundscape mixtures
                           (denoised focal clips + test-floor backgrounds + procedural distractors)
  3. predict.py          : round-1 model -> soft pseudo-labels on the (unlabelled) test soundscapes
  4. train.py --pseudo   : round 2 - real test clips used as backgrounds, with their pseudo-labels
  5. predict.py          : ensemble (round-1 + round-2, several seeds) with time-shift TTA
                           -> public_submission.csv / private_submission.csv in this folder

Device is auto-detected (GPU used when available; trained on host aorus-ts GTX 1050 Ti,
~6 min per model there; on a 2-thread CPU expect ~25-40 min per model).
usage: python solution.py [--epochs 40] [--quick]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE  # noqa: E402

PY = sys.executable


def run(*args: str) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=HERE, check=True, env=os.environ.copy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    a = ap.parse_args()
    ep = "2" if a.quick else str(a.epochs)
    common = ["--epochs", ep, "--per_epoch", "1600", "--width", "32"]

    # round 1: hold-out model (its soundscape-like validation AUC is logged) + full-data seeds
    run("train.py", "--name", "R1_full", *common)
    run("train.py", "--name", "F1", "--full", "--seed", "1", *common)
    run("train.py", "--name", "F2", "--full", "--seed", "2", *common)
    # pseudo-labels for the test soundscapes from the round-1 model (kept in the feature cache dir)
    pseudo = CACHE / "pseudo_R1.npz"
    run("predict.py", "R1_full", "--no_csv", "--pseudo_out", str(pseudo))
    # round 2: test clips as backgrounds with pseudo-labels
    run("train.py", "--name", "PF1", "--full", "--seed", "11", "--pseudo", str(pseudo), *common)
    run("train.py", "--name", "PF2", "--full", "--seed", "12", "--pseudo", str(pseudo), *common)
    # final ensemble (+ 4 time-shift TTA) -> CSVs in this folder
    run("predict.py", "R1_full", "F1", "F2", "PF1", "PF2", "--tta", "4")


if __name__ == "__main__":
    main()
