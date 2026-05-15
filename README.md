# AIHub AES v2

KoBERT + GRU automated essay scoring experiments for AIHub essay data.

## Main scripts

- `end_to_end.py`: end-to-end KoBERT CLS embedding and GRU training/evaluation workflow.
- `end_to_end_fine_tune.py`: fine-tuning variant with deterministic mode, AMP, gradient clipping, early stopping, and checkpoint reload support.

## Expected data layout

The scripts expect CSV files under `./aihub`:

```text
aihub/
  train.csv
  valid.csv
  test.csv
```

Required columns include essay text, prompt text, and rater score columns referenced in the scripts.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python end_to_end.py
python end_to_end_fine_tune.py
```

Outputs are written under `./res/results` and model checkpoints under `./model` where applicable.
