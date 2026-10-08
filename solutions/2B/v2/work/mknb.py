import json, sys
src = open(sys.argv[1]).read().split("\n")
def seg(a, b): return "\n".join(src[a - 1:b - 1]).strip("\n")
ln = {k: next(i + 1 for i, l in enumerate(src) if l.startswith(k)) for k in
      ["CLASSES=", "def typo_fixer", "# ------------------------------------------------------------------ models",
       "# ------------------------------------------------------------------ aug", "# ==========", "def main", "if __name__"]}
doc_end = next(i + 1 for i, l in enumerate(src) if l.startswith('"""') and i > 0) + 1
cells = [
 ("markdown", "# 2B v2 · Rice-leaf disease, multi-modal\nImage (ImageNet-pretrained ResNet-18) + Vietnamese text (TF-IDF LR) + tabular (HGB) experts, "
  "fold-clean pseudo-labelling of the test photos, LR stacking on out-of-fold log-probabilities. See README.md for ablations."),
 ("code", seg(1, ln["CLASSES="])),
 ("markdown", "## 1. Data\nText cleaning (accents, repeated letters, province masking, typo normalisation) and tabular features."),
 ("code", seg(ln["CLASSES="], ln["# ------------------------------------------------------------------ models"])),
 ("markdown", "## 2. Model\nTiny CNN (ablation only) and torchvision backbones with an avg+max pooled head."),
 ("code", seg(ln["# ------------------------------------------------------------------ models"], ln["# ------------------------------------------------------------------ aug"])),
 ("markdown", "## 3. Training & Inference\nAugmentation (dihedral, colour, Gaussian blur matching the test blur), soft-target training, 8-way TTA, "
  "experts, fold-clean pseudo-labels and the stacker."),
 ("code", seg(ln["# ------------------------------------------------------------------ aug"], ln["def main"])),
 ("markdown", "## 4. Predict\nRuns all experts (cached in `cache/`), stacks them and writes `public_submission.csv` / `private_submission.csv`."),
 ("code", seg(ln["def main"], ln["if __name__"]) + "\n\n\nmain()"),
]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True), **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
