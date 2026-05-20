# Data Setup Notes

The training scripts expect AIHub essay data under `./aihub` with train, validation, and test CSV files.

## Expected Layout

```text
aihub/
  train.csv
  valid.csv
  test.csv
```

## Pre-Run Checks

- Confirm all CSV files use UTF-8 encoding.
- Verify essay text, prompt text, and rater score columns match the script configuration.
- Check split sizes before launching long runs.
- Keep raw AIHub exports outside Git unless an explicit data policy says otherwise.

## Local Paths

If data lives outside the repository, prefer a symlink or a documented environment variable over hard-coded absolute paths. Record the data revision with every result table.
