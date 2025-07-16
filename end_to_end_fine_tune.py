#!/usr/bin/env python
# coding: utf-8
"""
end_to_end_fine_tune.py
────────────────────────────────────────────────────────────
End-to-end Automated Essay Scoring (AES) using KoBERT + GRU.

Features
--------
✓ KoBERT fine-tuning (optional)
✓ Variant-independent reproducibility (fresh KoBERT per variant + per-variant seeds)
✓ Strict deterministic mode w/ CuBLAS env var check (friendly error)
✓ AMP mixed precision (optional)
✓ Gradient clipping (optional)
✓ Early stopping on validation loss
✓ Save & safely reload best checkpoint (weights_only=True if supported)
✓ Proper scaling for metrics (0~1 scale internally; 0~5 ints for CSV)

Expected Data Layout
--------------------
`cfg.data_dir/` contains: train.csv, valid.csv, test.csv
Columns include:
    answer@text            - full essay, sentences separated by "#@문장구분#"
    question@prompt        - prompt text
    rater{1,2}@{score_key} - integer scores 0~5 per trait

Output
------
Per-variant test predictions CSV at:
    ./res/results/y_pred_result_test_{variant}.csv

Usage
-----
$ export CUBLAS_WORKSPACE_CONFIG=:4096:8   # if strict determinism enabled
$ python end_to_end_fine_tune.py
"""

from __future__ import annotations

import os
import gc
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score

from kobert_transformers import get_kobert_model, get_tokenizer
# Optional alternative backbone (commented)
# from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm


# ───────────────────────────── 설정 ──────────────────────────────
@dataclass
class CFG:
    # paths
    data_dir: Path = Path("./aihub")       # train.csv, valid.csv, test.csv
    res_dir : Path = Path("./res/results")
    model_dir: Path = Path("./model")

    # batching
    batch_size  : int   = 128    # essays per batch in DataLoader
    sent_batch  : int   = 128    # number of sentences per sub-batch fed into KoBERT (VRAM control)
    max_len_raw : int   = 50
    max_len_lbl : int   = 70     # (adjust if prompts included increase length)

    # model
    hidden_dim  : int   = 128
    dropout     : float = .5

    # training
    n_epochs    : int   = 100
    patience    : int   = 10          # early stopping patience (epochs w/out val improvement)
    grad_clip   : Optional[float] = 1.0  # clip grad norm; None to disable

    # optim / lr
    finetune_bert: bool = False
    lr_bert      : float = 2e-5
    lr_head      : float = 1e-3
    weight_decay : float = 0.01
    # AMP
    use_amp: bool = False

    # reproducibility
    strict_determinism: bool = True   # user requested to keep strict; env var required!
    base_seed         : int  = 42     # will add variant index offset

    # which variants to run: (tag, with_prompt)
    variants    : Tuple[Tuple[str,bool], ...] = (
        ("kobert", False),
        ("labeled_kobert", True),
    )

    # logging
    verbose: bool = True


cfg = CFG()
cfg.res_dir.mkdir(parents=True, exist_ok=True)
cfg.model_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────── 전역 재현성 유틸 ────────────────────────
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # cudnn deterministic baseline; strict-level tuning handled in configure_determinism()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_determinism(cfg: CFG):
    """
    Configure deterministic behavior. If strict_determinism=True, we enforce
    deterministic algorithms AND require CuBLAS workspace config to be set
    *before* process start (user must export env var).
    """
    # baseline for reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if cfg.strict_determinism:
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            msg = (
                "\n[ERROR] strict_determinism=True but CUBLAS_WORKSPACE_CONFIG not set.\n"
                "For full determinism with CUDA >= 10.2, set one of:\n"
                "    export CUBLAS_WORKSPACE_CONFIG=:4096:8   # recommended\n"
                "    export CUBLAS_WORKSPACE_CONFIG=:16:8     # lower memory\n"
                "Then re-run this script.\n"
                "Or set cfg.strict_determinism=False to continue without full determinism.\n"
                "Docs: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility\n"
            )
            raise RuntimeError(msg)
        torch.use_deterministic_algorithms(True)
        if cfg.verbose:
            print("[INFO] Strict determinism enabled. CuDNN deterministic + torch.use_deterministic_algorithms(True).")
    else:
        torch.use_deterministic_algorithms(False)
        if cfg.verbose:
            print("[INFO] Strict determinism disabled. Reproducibility good but minor numerical drift possible.")


# ──────────────────────── 데이터셋 ─────────────────────────
KEYS = [
    "task_1", "content_1", "content_2", "content_3",
    "organization_1", "organization_2", "expression_1", "expression_2"
]


class EssayDataset(Dataset):
    """
    Stores each essay as a list of sentence strings.
    Optionally prefixes each sentence with "<prompt>###" when with_prompt=True.
    Labels are averaged rater1+rater2 then normalized to 0~1 scale (divide by 5).
    """
    def __init__(self, df: pd.DataFrame, with_prompt=False, label=True):
        sents = [t.split("#@문장구분#") for t in df["answer@text"].tolist()]
        if with_prompt:
            prompts = df["question@prompt"].tolist()
            sents = [[f"{p}#{s}" for s in es] for p, es in zip(prompts, sents)]
        MAX_SENTS = 128  # max sentences per essay; adjust if needed
        if len(sents) > 128:
            sents = [es[:MAX_SENTS] for es in sents]
        self.sent_groups = sents
        if label:
            r1 = df[[f"rater1@{k}" for k in KEYS]].values
            r2 = df[[f"rater2@{k}" for k in KEYS]].values
            self.labels = torch.tensor(((r1 + r2) / 2) / 5, dtype=torch.float32)  # 0~1
        else:
            self.labels = None

    def __len__(self): return len(self.sent_groups)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sent_groups[idx]
        return self.sent_groups[idx], self.labels[idx]


# ─────────────────── collate_fn: 토큰화 수행 ──────────────────
def make_collate(tokenizer, max_len):
    """
    Collate that:
      - Flattens all sentences in batch
      - Tokenizes to fixed length
      - Tracks group lengths (sentences per essay)
    Returns: input_ids, attention_mask, group_lens, labels
    """
    def collate(batch):
        sents, labels = zip(*batch)  # tuple length B
        flat = [s for g in sents for s in g]
        enc = tokenizer.batch_encode_plus(
            flat,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        group_lens = torch.tensor([len(g) for g in sents], dtype=torch.long)
        labels = torch.stack(labels)  # (B,8)
        return enc["input_ids"], enc["attention_mask"], group_lens, labels
    return collate


# ─────────────────────────── Model ────────────────────────────
class End2End(nn.Module):
    """
    End-to-end KoBERT + BiGRU scorer.
    forward(input_ids, attention_mask, group_lens) -> (B, 8 scores in 0~1)
    """
    def __init__(self, kobert, sent_batch: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.kobert = kobert
        self.sent_batch = sent_batch
        self.gru = nn.GRU(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.pred = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 8),
            nn.Sigmoid()
        )

    def _encode_cls(self, input_ids, attention_mask):
        """
        Encode a (N_sent, L) batch through KoBERT in sub-batches to reduce VRAM.
        Returns CLS embeddings shape (N_sent, 768).
        """
        outs = []
        for i in range(0, input_ids.size(0), self.sent_batch):
            chunk_ids  = input_ids[i:i+self.sent_batch]
            chunk_mask = attention_mask[i:i+self.sent_batch]
            o = self.kobert(input_ids=chunk_ids, attention_mask=chunk_mask).last_hidden_state[:, 0]
            outs.append(o)
        return torch.cat(outs, dim=0)

    def forward(self, input_ids, attention_mask, group_lens):
        cls = self._encode_cls(input_ids, attention_mask)  # (N_sent, 768)
        sent_list = cls.split(group_lens.tolist(), dim=0)
        essays_pad = pad_sequence(sent_list, batch_first=True)  # (B, maxS, 768)
        out, _ = self.gru(essays_pad)
        last = out[:, -1]  # (B, hidden*2)
        return self.pred(last)  # (B,8) 0~1


# ─────────────────────── Metrics ──────────────────────────────
def metrics(pred_01: np.ndarray, true_01: np.ndarray):
    """
    pred_01, true_01: arrays in [0,1] scale.
    Convert to 0-5 integers via rounding and compute per-trait accuracy & QWK.
    """
    y_p = np.rint(pred_01 * 5).astype(int)
    y_t = np.rint(true_01 * 5).astype(int)
    acc  = np.array([accuracy_score(y_t[:, i], y_p[:, i]) for i in range(8)])
    kappa= np.array([
        cohen_kappa_score(y_t[:, i], y_p[:, i], weights="quadratic")
        for i in range(8)
    ])
    return acc, kappa


# ─────────────────────── Train / Eval ─────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: Optional[float] = None,
    use_amp: bool = False,
) -> float:
    model.train()
    running = 0.0
    for i, (ids, mask, lens, y) in enumerate(loader):
        ids  = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        if scaler is not None and use_amp:
            with torch.cuda.amp.autocast():
                out = model(ids, mask, lens)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            out = model(ids, mask, lens)
            loss = criterion(out, y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

        running += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Train [{i+1}/{len(loader)}] Loss: {running/(i+1):.4f}", end="\r")

    return running / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device,
    use_amp: bool = False,
):
    model.eval()
    total = 0.0
    outs  = []
    for ids, mask, lens, y in loader:
        ids  = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                o = model(ids, mask, lens)
                loss = criterion(o, y)
        else:
            o = model(ids, mask, lens)
            loss = criterion(o, y)

        total += loss.item()
        outs.append(o.cpu().numpy())

    return total / len(loader), np.concatenate(outs, axis=0)


# ─────────────────────── Checkpoint I/O ───────────────────────
def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint_safe(path: Path, device):
    """
    Try torch.load(weights_only=True) (PyTorch >=2.5),
    fallback to classic torch.load if older version.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# ─────────────────────── Variant Runner ───────────────────────
def run_variant(
    cfg: CFG,
    tok,
    kobert_init_sd: Dict[str, torch.Tensor],
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    tag: str,
    use_lbl: bool,
    variant_idx: int,
) -> Dict[str, Any]:
    """
    Runs one variant (raw or labeled).
    Returns dict of metrics & paths.
    """
    # -------- per-variant seed --------
    var_seed = cfg.base_seed + variant_idx
    set_seed(var_seed)
    if cfg.verbose:
        print(f"\n[Variant {variant_idx}] tag={tag} with_prompt={use_lbl} seed={var_seed}")

    # -------- fresh KoBERT init per variant --------
    kobert = get_kobert_model()
    # alt: huggingface
    # kobert = AutoModel.from_pretrained("klue/roberta-base")
    # tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    kobert.load_state_dict(kobert_init_sd, strict=True)
    kobert.to(device)

    if not cfg.finetune_bert:
        for p in kobert.parameters():
            p.requires_grad_(False)

    # -------- datasets --------
    tr_ds = EssayDataset(tr_df, with_prompt=use_lbl, label=True)
    va_ds = EssayDataset(va_df, with_prompt=use_lbl, label=True)
    te_ds = EssayDataset(te_df, with_prompt=use_lbl, label=True)

    max_len = cfg.max_len_lbl if use_lbl else cfg.max_len_raw
    collate = make_collate(tok, max_len=max_len)

    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=0, collate_fn=collate,
                       pin_memory=torch.cuda.is_available())
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate,
                       pin_memory=torch.cuda.is_available())
    te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate,
                       pin_memory=torch.cuda.is_available())

    if cfg.verbose:
        print(f"  Data: Train={len(tr_ds)} Val={len(va_ds)} Test={len(te_ds)} max_len={max_len}")

    # -------- model & optim --------
    model = End2End(
        kobert=kobert,
        sent_batch=cfg.sent_batch,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()

    if cfg.finetune_bert:
        opt = torch.optim.AdamW([
            {
                "params": model.kobert.parameters(),
                "lr": cfg.lr_bert,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": list(model.gru.parameters()) + list(model.pred.parameters()),
                "lr": cfg.lr_head,
                "weight_decay": cfg.weight_decay,
            },
        ])
    else:
        opt = torch.optim.AdamW([
            {
                "params": list(model.gru.parameters()) + list(model.pred.parameters()),
                "lr": cfg.lr_head,
                "weight_decay": cfg.weight_decay,
            },
        ])

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and torch.cuda.is_available()))

    # -------- training loop w/ early stopping --------
    best_val = float("inf")
    patience = cfg.patience
    best_path = cfg.model_dir / f"{tag}.pth"

    for ep in range(cfg.n_epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, tr_dl, criterion, opt, device,
            scaler=scaler, grad_clip=cfg.grad_clip, use_amp=cfg.use_amp,
        )
        va_loss, _ = evaluate(
            model, va_dl, criterion, device, use_amp=cfg.use_amp
        )

        if cfg.verbose:
            dt = time.time() - t0
            print(f"[EP {ep:03d}] train {tr_loss:.4f} | val {va_loss:.4f} | best {best_val:.4f} | {dt:.1f}s")

        if va_loss < best_val:
            best_val = va_loss
            patience = cfg.patience
            save_checkpoint(model, best_path)
            if cfg.verbose:
                print(f"  ↳ new best! saved to {best_path}")
        else:
            patience -= 1
            if patience == 0:
                if cfg.verbose:
                    print(f"  ↳ early stopping at epoch {ep}")
                break

    # -------- load best checkpoint --------
    if best_path.exists():
        state = load_checkpoint_safe(best_path, device)
        model.load_state_dict(state)
        if cfg.verbose:
            print(f"[INFO] Loaded best checkpoint from {best_path} (val={best_val:.4f}).")
    else:
        print(f"[WARN] Best checkpoint not found at {best_path}; using last-epoch weights.")

    # -------- re-eval validation (best) --------
    val_loss_best, val_out_best = evaluate(
        model, va_dl, criterion, device, use_amp=cfg.use_amp
    )

    # -------- test --------
    te_loss, te_out = evaluate(
        model, te_dl, criterion, device, use_amp=cfg.use_amp
    )

    # Return 0-5 integer predictions for CSV
    y_pred_int = np.rint(te_out * 5).astype(int)
    # clip safety
    y_pred_int = np.clip(y_pred_int, 0, 5)

    pd.DataFrame(y_pred_int, columns=KEYS).to_csv(
        cfg.res_dir / f"y_pred_result_test_{tag}.csv",
        index=False
    )

    # cleanup
    torch.cuda.empty_cache(); gc.collect()

    return {
        "tag": tag,
        "use_lbl": use_lbl,
        "best_val_loss": best_val,
        "val_loss_best_reloaded": val_loss_best,
        "test_loss": te_loss,
        "test_pred01": te_out,      # float preds 0~1
        "test_pred_int": y_pred_int # ints 0~5
    }


# ─────────────────────────── Main ─────────────────────────────
def main():
    # configure determinism (may raise friendly error if strict True & env missing)
    configure_determinism(cfg)

    # tokenizer (shared across variants)
    tok = get_tokenizer()

    # alt backbone example:
    # tok = AutoTokenizer.from_pretrained("klue/roberta-base")

    # Initial KoBERT weights snapshot (variant resets)
    _kobert_init = get_kobert_model()
    # alt:
    # _kobert_init = AutoModel.from_pretrained("klue/roberta-base")
    kobert_init_sd = {k: v.cpu() for k, v in _kobert_init.state_dict().items()}
    del _kobert_init

    # load data
    tr_df = pd.read_csv(cfg.data_dir / "train.csv", encoding="utf-8-sig")
    va_df = pd.read_csv(cfg.data_dir / "valid.csv", encoding="utf-8-sig")
    te_df = pd.read_csv(cfg.data_dir / "test.csv",  encoding="utf-8-sig")

    # filter to single prompt seed (first prompt)
    prompts = tr_df["question@prompt"].tolist()
    seed_prompt = prompts[:5]
    tr_df = tr_df[tr_df["question@prompt"].isin(seed_prompt)].reset_index(drop=True)
    va_df = va_df[va_df["question@prompt"].isin(seed_prompt)].reset_index(drop=True)
    te_df = te_df[te_df["question@prompt"].isin(seed_prompt)].reset_index(drop=True)

    if cfg.verbose:
        print(f"Filtered by prompt: '{seed_prompt}'")
        print(f"Train: {len(tr_df)} | Valid: {len(va_df)} | Test: {len(te_df)}")

    # ground-truth test scores (integers)
    y_te_r1 = te_df[[f"rater1@{k}" for k in KEYS]].values.astype(int)
    y_te_r2 = te_df[[f"rater2@{k}" for k in KEYS]].values.astype(int)

    results = []

    for i, (tag, use_lbl) in enumerate(cfg.variants):
        res = run_variant(
            cfg=cfg,
            tok=tok,
            kobert_init_sd=kobert_init_sd,
            tr_df=tr_df,
            va_df=va_df,
            te_df=te_df,
            tag=tag,
            use_lbl=use_lbl,
            variant_idx=i,
        )
        results.append(res)

        # compute metrics against both raters (0~1 scale)
        pred01 = res["test_pred01"]
        acc1, kap1 = metrics(pred01, y_te_r1 / 5.0)
        acc2, kap2 = metrics(pred01, y_te_r2 / 5.0)

        print(
            f"\n[{tag}] TestLoss {res['test_loss']:.4f} | "
            f"R1 ACC {acc1.mean():.4f} QWK {kap1.mean():.4f} || "
            f"R2 ACC {acc2.mean():.4f} QWK {kap2.mean():.4f}"
        )

        print(f"[{tag}]  ACC (r1 | r2) : {np.round(acc1,4)} | {np.round(acc2,4)}")
        print(f"[{tag}]  QWK (r1 | r2) : {np.round(kap1,4)} | {np.round(kap2,4)}")
        print(f"[{tag}]  Overall ACC  r1 {acc1.mean():.4f} | r2 {acc2.mean():.4f}")
        print(f"[{tag}]  Overall QWK  r1 {kap1.mean():.4f} | r2 {kap2.mean():.4f}")

    # (optional) aggregate summary CSV
    summ_rows = []
    for res in results:
        tag = res["tag"]
        pred01 = res["test_pred01"]
        acc1, kap1 = metrics(pred01, y_te_r1 / 5.0)
        acc2, kap2 = metrics(pred01, y_te_r2 / 5.0)
        summ_rows.append({
            "variant": tag,
            "best_val_loss": res["best_val_loss"],
            "test_loss": res["test_loss"],
            "r1_acc_mean": acc1.mean(),
            "r1_qwk_mean": kap1.mean(),
            "r2_acc_mean": acc2.mean(),
            "r2_qwk_mean": kap2.mean(),
        })

    summ_df = pd.DataFrame(summ_rows)
    summ_path = cfg.res_dir / "variant_summary.csv"
    summ_df.to_csv(summ_path, index=False)
    print(f"\nSummary saved: {summ_path}")


if __name__ == "__main__":
    main()
