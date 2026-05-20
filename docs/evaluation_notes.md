# Evaluation Notes

Use the same split and metric implementation when comparing `end_to_end.py` and `end_to_end_fine_tune.py` runs.

## Metrics

Track at least:

- Quadratic weighted kappa.
- Mean absolute error.
- Root mean squared error.
- Per-rubric behavior when multiple score columns are predicted.

## Result Record

For each result, keep:

- Script entrypoint.
- Git commit.
- Dataset split revision.
- Random seed and deterministic-mode setting.
- AMP, gradient clipping, early stopping, and checkpoint reload settings.
- Output path under `./res/results` and checkpoint path under `./model`.

Small result summaries can be committed, but raw checkpoints and generated predictions should stay in external run folders.
