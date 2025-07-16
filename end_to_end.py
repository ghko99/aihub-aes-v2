#!/usr/bin/env python
# coding: utf-8
"""
e2e_kobert_gru.py
─────────────────────────────────────────────────────────
* KoBERT CLS 임베딩을 실시간으로 생성하여 GRU 에 입력
* topic‑labeled / raw 두 변형을 차례로 학습·테스트
* 결과 CSV: ./res/results/y_pred_result_test_{variant}_{exp}.csv
  (variant = kobert | labeled_kobert)
"""

from __future__ import annotations
import gc, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score
from kobert_transformers import get_kobert_model, get_tokenizer
from tqdm import tqdm
import os, random
import numpy as np


# ───────────────────────────── 설정 ──────────────────────────────
@dataclass
class CFG:
    data_dir: Path = Path("./aihub")       # train.csv, valid.csv, test.csv
    res_dir : Path = Path("./res/results")

    batch_size  : int   = 128    # 에세이 batch
    sent_batch  : int   = 128    # 문장 batch(KoBERT 입력) → VRAM 맞춰 조정
    max_len_raw : int   = 50

    hidden_dim  : int   = 128
    dropout     : float = .5
    lr          : float = 1e-4
    n_epochs    : int   = 100
    patience    : int   = 10

    tag = "kobert"  # 실험 태그

cfg = CFG();  cfg.res_dir.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ──────────────────────── 데이터셋 ─────────────────────────
KEYS = ["task_1","content_1","content_2","content_3",
        "organization_1","organization_2","expression_1","expression_2"]

class EssayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label=True):
        prompts = df["question@prompt"].tolist()
        sents = [[f"prompt:{p}"] + t.split("#@문장구분#") for t,p in zip(df["answer@text"].tolist(), prompts)]
        self.sent_groups = sents
        if label:
            r1 = df[[f"rater1@{k}" for k in KEYS]].values
            r2 = df[[f"rater2@{k}" for k in KEYS]].values
            self.labels = torch.tensor(((r1+r2)/2)/5, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self): return len(self.sent_groups)
    def __getitem__(self, idx):
        if self.labels is None:
            return self.sent_groups[idx]
        return self.sent_groups[idx], self.labels[idx]
# ─────────────────── seed 고정 ──────────────────
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ─────────────────── collate_fn: KoBERT 임베딩 ──────────────────
def make_collate(tokenizer, kobert, max_len):
    @torch.no_grad()
    def collate(batch):
        sents, labels = zip(*batch)
        flat = [s for group in sents for s in group]

        cls_vecs = []
        for i in range(0, len(flat), cfg.sent_batch):
            enc = tokenizer.batch_encode_plus(
                flat[i:i+cfg.sent_batch],
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(device)
            vec = kobert(**enc).last_hidden_state[:,0]  # (bs,768)
            cls_vecs.append(vec.cpu())
        cls = torch.cat(cls_vecs)                       # (total_sent,768)

        essays, idx = [], 0
        for group in sents:
            essays.append(cls[idx:idx+len(group)])
            idx += len(group)
        essays_pad = pad_sequence(essays, batch_first=True)  # (B, maxL,768)
        return essays_pad.to(device), torch.stack(labels).to(device)
    return collate

# ─────────────────────────── Model ────────────────────────────
class End2End(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(768, cfg.hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True,
                          dropout=cfg.dropout)
        self.pred = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim*2, 8),
            nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.pred(out[:,-1])

# ─────────────────────── Metrics ──────────────────────────────
def metrics(pred, true):
    y_p = np.rint(pred*5).astype(int)
    y_t = np.rint(true*5).astype(int)
    acc  = np.array([accuracy_score(y_t[:,i], y_p[:,i]) for i in range(8)])
    kappa= np.array([cohen_kappa_score(y_t[:,i], y_p[:,i], weights="quadratic")
                     for i in range(8)])
    return acc, kappa

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); total, outs = 0.0, []
    for x,y in loader:
        o = model(x)
        total += criterion(o,y).item()
        outs.extend(o.cpu().numpy())
    return total/len(loader), np.asarray(outs)

def train_one_epoch(model, loader, criterion, optim):
    model.train(); running=0.0
    for i, (x,y) in enumerate(loader):
        optim.zero_grad()
        loss = criterion(model(x),y)
        loss.backward()
        optim.step()
        running += loss.item()
        if (i+1) % 10 == 0:
            print(f"Train [{i+1}/{len(loader)}] Loss: {running/(i+1):.4f}", end="\r")
    return running/len(loader)

# ─────────────────────────── Main ─────────────────────────────
def main():
    kobert = get_kobert_model().to(device)
    tok    = get_tokenizer()
    kobert = kobert.eval()
    for p in kobert.parameters(): p.requires_grad_(False)
    tr_df = pd.read_csv(cfg.data_dir/"train.csv", encoding="utf-8-sig")
    va_df = pd.read_csv(cfg.data_dir/"valid.csv", encoding="utf-8-sig")
    te_df = pd.read_csv(cfg.data_dir/"test.csv",  encoding="utf-8-sig")

    prompts = tr_df["question@prompt"].tolist()
    # 첫 번째 프롬프트를 seed로 사용
    seed_prompt = prompts[0]

    tr_df = tr_df[tr_df["question@prompt"] == seed_prompt]
    va_df = va_df[va_df["question@prompt"] == seed_prompt]
    te_df = te_df[te_df["question@prompt"] == seed_prompt]
    set_seed(42)
    y_te_r1 = te_df[[f"rater1@{k}" for k in KEYS]].values.astype(int)
    y_te_r2 = te_df[[f"rater2@{k}" for k in KEYS]].values.astype(int)


    tr_ds = EssayDataset(tr_df, label=True)
    va_ds = EssayDataset(va_df, label=True)
    te_ds = EssayDataset(te_df, label=True)
    print(f"Train: {len(tr_ds)} | Valid: {len(va_ds)} | Test: {len(te_ds)}")
    collate = make_collate(tok, kobert, cfg.max_len_raw)

    print(f"Collate: {collate.__name__} (max_len={cfg.max_len_raw})")
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

    tot_acc_r1 = np.zeros(8);  tot_kap_r1 = np.zeros(8)
    tot_acc_r2 = np.zeros(8);  tot_kap_r2 = np.zeros(8)

    print(f"Training experiments...")
    model = End2End().to(device)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best, patience = float('inf'), cfg.patience
    for ep in range(cfg.n_epochs):
        tr_loss = train_one_epoch(model, tr_dl, crit, opt)
        va_loss, va_out = evaluate(model, va_dl, crit)
        print(f"[EP: {ep}/{cfg.n_epochs}] Train {tr_loss:.4f} | Val {va_loss:.4f} | BEST VAL {best:.4f}", end="\r")

        if va_loss < best:
            best, patience = va_loss, cfg.patience
            torch.save(model.state_dict(),
                        f"./model/{cfg.tag}.pth")
        else:
            patience -= 1
            if patience==0: break

    te_loss, te_out = evaluate(model, te_dl, crit)
    y_pred = np.rint(te_out * 5).astype(int)

    acc1, kap1 = metrics(y_pred, y_te_r1)   # 함수는 0~1 스케일 입력
    acc2, kap2 = metrics(y_pred, y_te_r2)

    tot_acc_r1 += acc1;  tot_kap_r1 += kap1
    tot_acc_r2 += acc2;  tot_kap_r2 += kap2

    print(f"\n TestLoss {te_loss:.4f} | "
            f"R1 ACC {acc1.mean():.4f} QWK {kap1.mean():.4f} || "
            f"R2 ACC {acc2.mean():.4f} QWK {kap2.mean():.4f}")

    # CSV 저장 (버그‑호환 8‑컬럼 헤더)
    pd.DataFrame(y_pred, columns=KEYS).to_csv(
        cfg.res_dir / f"y_pred_result_test_{cfg.tag}.csv", index=False
    )

    torch.cuda.empty_cache(); gc.collect()

    m1_acc, m2_acc = tot_acc_r1, tot_acc_r2
    m1_kap, m2_kap = tot_kap_r1, tot_kap_r2

    print(f"[{cfg.tag}]  ACC (r1 | r2) : {np.round(m1_acc,4)} | {np.round(m2_acc,4)}")
    print(f"[{cfg.tag}]  QWK (r1 | r2) : {np.round(m1_kap,4)} | {np.round(m2_kap,4)}")
    print(f"[{cfg.tag}]  Overall ACC  r1 {m1_acc.mean():.4f} | r2 {m2_acc.mean():.4f}")
    print(f"[{cfg.tag}]  Overall QWK  r1 {m1_kap.mean():.4f} | r2 {m2_kap.mean():.4f}")

if __name__ == "__main__":
    main()
