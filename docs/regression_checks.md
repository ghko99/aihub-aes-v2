# Regression Checks

Use this checklist before accepting a change to the AIHub AES v2 training or evaluation workflow.

## Fixed Inputs

- Use the same train, validation, and test CSV files.
- Keep the same random seed and deterministic-mode setting.
- Record whether AMP, gradient clipping, early stopping, and checkpoint reload are enabled.
- Preserve the same label scaling and score columns.

## Compare Outputs

- Overall QWK, MAE, and RMSE.
- Per-rubric metrics when available.
- Prediction distribution by score bucket.
- High-error examples from the previous and current runs.

## Acceptance Notes

A regression check should include the baseline commit, candidate commit, metric delta, and any expected behavior change. Keep raw predictions outside Git unless they are tiny curated examples.
