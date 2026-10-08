# Olympiad AI — datasets

Public inputs for the FTU Olympiad AI warm-up rounds. Each release tag holds one
round; each asset is one task, and expands to `<task>/baseline.ipynb` plus
`<task>/dataset/`.

```
<task>/
  baseline.ipynb
  dataset/
    train/         inputs and their labels, for training
    public_test/   inputs only, scored live during the round
    private_test/  inputs only, scored at the end
```

The test folders contain a manifest of ids and features and no targets. The
answer keys, the graders and the task notes are deliberately not here and are not
public anywhere.

Reference solutions for every task (1A–4B) are in `solutions/`: a v1 Gold solution and a v2 "push to the
limit" solution per task, each with an end-to-end script, a sample notebook and a full experiment log. See
`solutions/README.md` for the results table and the common lessons.

Scores and standings: https://seleccion.ftds.online
