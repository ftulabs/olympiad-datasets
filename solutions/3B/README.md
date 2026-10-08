# 3B · Molecule binding: solution (v1)

**Final score (score.py): public mAP 0.4837, private mAP 0.4265, Gold tier** (Gold needs ≥0.42; the baseline scores about 0.270 private).

| Submission | public | private |
|---|---|---|
| 3 GNN seeds + 3 FP seeds, rank blend 0.7/0.3 | 0.4823 | 0.4211 |
| **6 GNN seeds + 3 FP seeds, rank blend 0.8/0.2 (final, matches `solution.py` defaults)** | **0.4837** | **0.4265** |

score.py was run twice in total. Everything else was decided on my own grouped validation.

Files:
- `solution.py`: end-to-end script that writes `public_submission.csv` and `private_submission.csv`.
- `solution.ipynb`: student-readable version with Data, Model, Training & Inference and Predict sections. It has no submit cell.
- `experiments/`: the scripts behind the ablations (`lib.py`/`run.py` for grouped-CV training, `blockid.py`, `blockana*.py` and `rel.py` for the block-level analysis).

The final training ran on the remote host **minh-sager** (GTX 970M, about 10 min for the whole pipeline). Feature extraction there took 33 s with 4 workers. `solution.py` picks CUDA or CPU automatically. On a 2-thread CPU it runs but is slow: a GNN epoch takes several minutes, so use `--gnn_seeds 1` for a quick run. Use `DATA3B=<dataset dir>` to point it at the data.

## 1. Data (most of the effort)

**EDA**
- 40k train / 5k public / 5k private. Binder rates: BRD4 6.1%, HSA 6.3%, sEH 5.3%.
- Targets are almost independent: pairwise correlation is about 0, and only 393 molecules bind more than one target.
- Elements: C, N, O, S, F, Cl, Br. Bond types are 1, 2, 3 and aromatic. The baseline drops the bond type.
- Atom order inside a molecule is shuffled.

**Atom features (no RDKit)**
- Element, degree, implicit H from a valence rule (aromatic bond = 1.5), and aromatic flag.
- Ring membership: ring bonds are the bonds that are not bridges (Tarjan), so I compute this myself.
- Smallest ring size.

**Fingerprints.** I wrote an ECFP/Morgan-like WL hashing that includes bond types, radius 0 to 3.

**Scaffold + 3 building-block decomposition (the key data insight)**
- Method: cut the bridge bonds to get ring systems. The scaffold is the ring system with at least 3 large branches that sits at the tree centroid. The 3 branches are the building blocks, each identified by a WL hash of its subgraph.
- The decomposition is clean: train has exactly **5 scaffolds and 406 blocks**, and every block appears about 90 to 100 times.
- **Novelty in the test sets:**
  - Private test has 2 new scaffolds (1011 rows, about 20%) and about 200 new blocks.
  - Rows of the private test by number of unseen blocks: 0 → 1532, 1 → 1143, 2 → 1115, 3 → 1210.
  - Public is milder: 449 rows with a new scaffold and 509 rows where all 3 blocks are new.
- **Label structure:**
  - An additive logit model on block IDs plus scaffold ID reaches mAP **0.404** on a random split.
  - Block effects are reproducible: split-half correlation is 0.87 to 0.92.
  - Block *structure* predicts block effects with cross-validated correlation of only about 0.65 to 0.74 (ridge, random forest and Tanimoto-kernel all land there).
  - So the remaining problem is generalising to unseen blocks and scaffolds.

**Grouped validation** (used for every ablation):
- Hold out a random 1/8 of the blocks plus one scaffold.
- Validation = every molecule that contains a held-out part (about 24k); train = the rest (about 16k).
- I report mAP on all of validation and on `ge2` (rows with at least 2 new parts, the closest match to private).

## 2. Model

- **GNN-DeepSets (main model).**
  - Input: 27 one-hot atom features plus a one-hot of the part.
  - 4 GINE layers: message = ReLU(h_src + edge-type embedding), then a GIN MLP, BatchNorm, ReLU, dropout and a residual connection. Hidden size 128.
  - Atom states are pooled **per part** (scaffold, block 1 to 3) with sum and max.
  - A shared φ is applied to each part, then [φ(scaffold), Σ φ(blocks)] goes through ρ to 3 logits (multi-task).
  - Summing over blocks makes the model invariant to block order, and the per-part pooling matches how the molecules are built.
- **FP-DeepSets (blend partner).**
  - Per-part sum of ECFP (radius ≤ 2) key embeddings, plus a block/scaffold **ID embedding with ID-dropout 0.5**. Unseen IDs map to UNK, so the model learns to fall back on structure.
  - Same φ/ρ head as the GNN.

## 3. Training & inference

- Multi-task BCE, AdamW (wd 1e-2), one-cycle cosine with max LR 2e-3, batch 256, gradient clipping at 5.
- Fixed epoch budgets chosen on the grouped validation: GNN 15 epochs (still flat-to-improving at 15), FP 7 epochs (it overfits after about 8).
- Final fit on all 40k molecules with 6 GNN seeds and 3 FP seeds.
- Blend: per-target rank average, 0.8 GNN + 0.2 FP. The weight comes from the fold-0 out-of-fold predictions: 1.0 → 0.379, 0.8 → 0.386, 0.7 → 0.386, 0.5 → 0.381.
- No pos_weight or focal loss: AP only depends on ranking, and plain BCE was fine.

## Ablation log (grouped validation, fold 0 unless noted; mAP all / ge2)

| # | Model / change | val mAP | ge2 |
|---|---|---|---|
| 1 | whole-molecule ECFP r≤3 MLP, wd 1e-4 (peak at epoch 2–3, then heavy overfit to 0.25) | 0.318 | 0.278 |
| 2 | FP-DeepSets r≤1, wd 1e-2, dropout 0.3 | 0.312 | – |
| 3 | FP-DeepSets r≤2 (same settings) | 0.331 | 0.284 |
| 4 | FP-DeepSets r≤3 | 0.311 | 0.273 |
| 5 | #3 with stronger regularisation (emb 64, wd 5e-2, dropout 0.4) | 0.322 | 0.275 |
| 6 | #3 + block-ID embedding with ID-dropout 0.5 | 0.332 | 0.282 |
| 7 | #3 with purely additive head (Σ part logits) | 0.327 | 0.279 |
| 8 | #6 with additive head | 0.333 | 0.284 |
| 9 | hybrid GNN + FP per part + ID (stopped at epoch 7) | 0.306 | 0.272 |
| 10 | **GNN-DeepSets (GINE, per-part pooling), 15 epochs** | **0.379** | **0.346** |
| 11 | rank blend 0.8 × #10 + 0.2 × #6 | **0.386** | 0.347 |
| – | fold 1: FP-DeepSets+ID 0.315 / 0.287; GNN-DeepSets at epoch 7 (stopped) 0.334 / 0.351 | | |
| – | reference: additive block-ID model, random split (it cannot score unseen blocks) | 0.404 | – |

Validation is much harder than private: most validation rows contain an unseen part and a whole scaffold is held out. That is why validation sits around 0.38 while private is 0.43.

## Changes that mattered most

1. **Scaffold + building-block decomposition and per-part DeepSets pooling.** This gives the right inductive bias for novel combinations.
2. **GNN over fingerprints** for novel blocks: +0.05 on grouped validation. The edge-type-aware GINE generalises much better than ECFP embeddings, which memorise blocks.
3. **Novelty-aware grouped validation** (held-out blocks + scaffold). A random split would have favoured block-memorising models; by my estimate the ID model gets 0.40 on a random split but fails on novel blocks.
4. **Bond types and ring/aromatic/H atom features**, which the baseline dropped.
5. **Seed ensemble and rank blend** with the FP+ID model: private went from 0.4211 to 0.4265.

## Unfinished / ideas for v2

- The GNN was never tuned: hidden size, depth (5–6 layers), epochs (20–25), and the lr/wd sweep are all untested. On the GPU each fold-0 run takes only about 4 min. The obvious next step is to add a block-ID embedding with ID-dropout to the GNN (the hybrid in #9 also had fingerprints and was stopped early, so it is not conclusive).
- Message passing over the block tree (scaffold to block attention) or pairwise block interaction terms in ρ. Interaction effects were never tested.
- Augmentation for novelty: dropout of whole-block identity or "block masking" during GNN training, plus training on the full 3-fold grouped CV to pick the epoch.
- Better use of the strong per-block signal for *seen* blocks (30% of private rows): a stacked or additive block-ID logit added to the GNN logit (a 0.404-quality signal on seen blocks).
- Isolated-block fingerprints (computed on the block subgraph alone, independent of the scaffold) for the FP model: block-level correlation 0.74 with a random forest vs 0.69 for ridge on in-context keys.
- Semi-supervised / transductive use of the test structures (the new blocks repeat across many test molecules).
- Remote state: `minh-sager:~/olympiad_offload/3B_work/` holds the compact feature cache (`cache.npz`, 110 MB, built by `experiments/pre.py` + `compact.py`), `lib.py`/`run.py`, and `final/` with `solution.py` and its logs. Nothing is running there.
