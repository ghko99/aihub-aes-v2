#!/usr/bin/env python
# coding: utf-8
"""
메모리 친화형 KoBERT 임베딩 → GRU Scorer 학습 스크립트
-----------------------------------------------------------------
* 기존 train.py 와 결과(모델, 지표)는 동일하지만 **RAM 사용량**을 몇 GB→수백 MB로 절감.
* 임베딩 CSV는 최초 1회만 **float32 .memmap(NPY)** 로 변환 후 캐시합니다.
* Dataset ↔ DataLoader 간 **lazy padding(collate_fn)** 구조라 전체 퍼드 패딩 텐서를 만들지 않습니다.

* 사용법: python train_memmap.py  (경로 등은 CFG 에서 조정)
"""
from __future__ import annotations

import gc, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score

# ────────────────────────────────  설정  ────────────────────────────────
@dataclass
class CFG:
    model_variants: Tuple[str, ...] = ("kobert", "labeled_kobert")   # ← 추가
    data_dir   : Path = Path("./aihub")          # train.csv / valid.csv 위치
    emb_dir    : Path = Path("./emb")            # 임베딩 CSV ‑‑> NPY 캐시 보관 폴더
    model_dir  : Path = Path("./model")
    res_dir    : Path = Path("./res/results")

    batch_size : int   = 512        # 메모리 여유되면 ↑
    hidden_dim : int   = 128
    dropout    : float = 0.5
    lr         : float = 1e-3
    n_epochs   : int   = 100
    patience   : int   = 10
    n_exp      : int   = 1
    
    out_dim    : int   = 8          # 평가지표 개수
    emb_dim    : int   = 768        # KoBERT sentence embedding dim
    max_label  : int   = 5          # 라벨 스케일 (0~5

cfg = CFG()
for p in (cfg.model_dir, cfg.res_dir, cfg.emb_dir):
    p.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

keys =["task_1", "content_1", "content_2", "content_3","organization_1", 
       "organization_2", "expression_1", "expression_2"]

# ─────────────────────────────  유틸 / IO  ──────────────────────────────

def _read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.data_dir / f"{name}.csv", encoding="utf-8-sig")


def _build_labels(df: pd.DataFrame) -> np.ndarray:
    r1 = df[[f"rater1@{k}" for k in keys]].values
    r2 = df[[f"rater2@{k}" for k in keys]].values
    return ((r1 + r2) / 2) / cfg.max_label 


def _split_sentences(text_col: pd.Series) -> List[List[str]]:
    """#@문장구분# 토큰 기준 분할"""
    return [t.split("#@문장구분#") for t in text_col.tolist()]


# ───────────────────────  임베딩 CSV → memmap 캐시  ──────────────────────

def _cache_emb_csv_to_npy(model_name: str, split: str, total_rows: int):
    """embedding CSV(헤더 없음, cp949) → float32 npy(memmap) [총 rows, 768]"""
    csv_path = cfg.emb_dir / f"{model_name}_emb_feat_{split}.csv"
    npy_path = cfg.emb_dir / f"{model_name}_emb_{split}_f32.npy"

    if npy_path.exists():
        return  # 이미 캐시 완료

    chunk_rows = 100_000  # 메모리 절약용 청크 사이즈
    mm = np.memmap(npy_path, dtype="float32", mode="w+", shape=(total_rows, cfg.emb_dim))

    start = 0
    for chunk in pd.read_csv(
        csv_path,
        header=None,
        chunksize=chunk_rows,
        dtype="float32",
        encoding="cp949",
    ):
        rows = len(chunk)
        mm[start : start + rows] = chunk.values
        start += rows
    mm.flush()
    del mm


def _load_emb_memmap(model_name: str, split: str, total_rows: int):
    npy_path = cfg.emb_dir / f"{model_name}_emb_{split}_f32.npy"
    if not npy_path.exists():            # <‑‑ 추가
        _cache_emb_csv_to_npy(model_name, split, total_rows)
    return np.memmap(npy_path, dtype="float32", mode="r",
                     shape=(total_rows, cfg.emb_dim))


# ──────────────────────  임베딩 인덱스 & Dataset  ──────────────────────

def build_index(lens: List[int]) -> List[Tuple[int, int]]:
    """각 에세이(=문장 시퀀스)의 (시작,row_len) 인덱스 생성"""
    idx_pairs, start = [], 0
    for l in lens:
        idx_pairs.append((start, l))
        start += l
    return idx_pairs


class EssayDataset(Dataset):
    def __init__(
        self,
        idx_pairs: List[Tuple[int, int]],
        labels: np.ndarray,
        emb_mem: np.ndarray,
    ):
        self.idx_pairs = idx_pairs
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.emb_mem = emb_mem  # memmap, RAM 소비 0 (페이지‑인)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        start, length = self.idx_pairs[i]
        emb_np = self.emb_mem[start:start+length].copy()  # ← copy
        return torch.from_numpy(emb_np), self.labels[i]


def pad_collate(batch):
    """Variable length batch → pad to max_len"""
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True)  # padding_value=0.0 기본
    return xs_pad, torch.stack(ys)


# ───────────────────────────────   Model   ──────────────────────────────
class GRUScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            cfg.emb_dim,
            cfg.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim * 2, cfg.out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1])
        return self.act(self.fc(x))


# ───────────────────────────  Metrics / Utils  ──────────────────────────

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    acc = np.array([
        accuracy_score(pred[:, i], true[:, i]) for i in range(pred.shape[1])
    ])
    kappa = np.array([
        cohen_kappa_score(pred[:, i], true[:, i], weights="quadratic") for i in range(pred.shape[1])
    ])
    overall_kappa = cohen_kappa_score(pred.flatten(), true.flatten(), weights="quadratic")
    overall_acc = accuracy_score(pred.flatten(), true.flatten())
    return acc, kappa , overall_acc, overall_kappa


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    loss, outs = 0.0, []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss += crit(out, y).item()
        outs.extend(out.cpu().numpy())
    return loss / len(loader), np.asarray(outs)


def train_one_epoch(model, loader, crit, opt):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader)


# ───────────────────────────────  학습  ────────────────────────────────

def main():
    # 1) CSV → 에세이 텍스트 & 라벨 로딩
    tr_df, va_df, te_df = (_read_split(s) for s in ("train", "valid", "test"))

    y_te_r1 = te_df[[f"rater1@{k}" for k in keys]].values.astype(int)
    y_te_r2 = te_df[[f"rater2@{k}" for k in keys]].values.astype(int)
    
    tr_es, va_es, te_es = map(_split_sentences,
                              (tr_df["answer@text"],
                               va_df["answer@text"],
                               te_df["answer@text"]))
    tr_y, va_y, te_y    = map(_build_labels, (tr_df, va_df, te_df))

    # 에세이별 문장 길이(=행 수)는 topic label 여부와 무관 → 한 번만 계산
    tr_lens, va_lens, te_lens = tr_lens, va_lens, te_lens = [list(map(len, split))
                             for split in (tr_es, va_es, te_es)]
    tr_idx, va_idx, te_idx    = map(build_index, (tr_lens, va_lens, te_lens))
    

    for tag in cfg.model_variants:
        tr_emb_mem = _load_emb_memmap(tag, "train", sum(tr_lens))
        va_emb_mem = _load_emb_memmap(tag, "valid", sum(va_lens))
        te_emb_mem = _load_emb_memmap(tag, "test", sum(te_lens))

        # 3) Dataset / DataLoader
        tr_ds = EssayDataset(tr_idx, tr_y, tr_emb_mem)
        va_ds = EssayDataset(va_idx, va_y, va_emb_mem)
        te_ds = EssayDataset(te_idx, te_y, te_emb_mem)

        tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0,  # ← 2 → 0
                        collate_fn=pad_collate, pin_memory=True)

        va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0,
                        collate_fn=pad_collate, pin_memory=True)
        
        te_dl = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0,
                        collate_fn=pad_collate, pin_memory=True)

        # 4) 학습 루프

        tot_acc_r1 = np.zeros(cfg.out_dim);  tot_kap_r1 = np.zeros(cfg.out_dim)
        tot_acc_r2 = np.zeros(cfg.out_dim);  tot_kap_r2 = np.zeros(cfg.out_dim)
        tot_overall_acc_r1 = 0.0; tot_overall_acc_r2 = 0.0
        tot_overall_kap_r1 = 0.0; tot_overall_kap_r2 = 0.0

        for exp in range(cfg.n_exp):
            model = GRUScore().to(device)
            crit = nn.MSELoss()
            opt = optim.Adam(model.parameters(), lr=cfg.lr)

            best_loss, patience = float("inf"), cfg.patience
            t0 = time.time()

            for ep in range(cfg.n_epochs):
                tr_loss = train_one_epoch(model, tr_dl, crit, opt)
                va_loss, va_out = evaluate(model, va_dl, crit)

                dt = time.time() - t0
                print(
                    f"[Exp {exp+1}/{cfg.n_exp}] Ep {ep+1:03}/{cfg.n_epochs} "
                    f"Train {tr_loss:.4f} | Val {va_loss:.4f} | Δt {dt:.1f}s"
                )
                t0 = time.time()

                if va_loss < best_loss:
                    best_loss, patience = va_loss, cfg.patience
                    best_out = va_out  # noqa: F841  # (필요 시 저장)
                    torch.save(model.state_dict(), cfg.model_dir / f"kobert_model_exp{exp+1}.pth")
                else:
                    patience -= 1
                    if patience == 0:
                        print("Early stopping")
                        break
            te_loss, te_out = evaluate(model, te_dl, crit)

            y_pred = np.rint(te_out * cfg.max_label).astype(int)

            acc1, kap1, overall_acc, overall_kappa = compute_metrics(y_pred, y_te_r1)
            acc2, kap2, overall_acc, overall_kappa = compute_metrics(y_pred, y_te_r2)

            print(f"Test Loss {te_loss:.4f} | "
                f"R1 ACC {acc1.mean():.4f} QWK {kap1.mean():.4f} || "
                f"R2 ACC {acc2.mean():.4f} QWK {kap2.mean():.4f}")
            f"Overall ACC r1 {overall_acc:.4f} | r2 {overall_acc:.4f} || "
            f"Overall QWK r1 {overall_kappa:.4f} | r2 {overall_kappa:.4f}"
            # test 예측 CSV 저장 (원본 포맷 유지)
            pd.DataFrame(
                y_pred,
            columns = keys
            ).to_csv(cfg.res_dir / f"y_pred_result_test_baseline_{tag}_{exp}.csv", index=False)
            tot_acc_r1 += acc1;  tot_kap_r1 += kap1
            tot_acc_r2 += acc2;  tot_kap_r2 += kap2
            tot_overall_acc_r1 += overall_acc
            tot_overall_acc_r2 += overall_acc

            tot_overall_kap_r1 += overall_kappa
            tot_overall_kap_r2 += overall_kappa
            # GPU 캐시 정리 (다중 experiment 구동 시)
            torch.cuda.empty_cache()
            gc.collect()

        
        mean_r1_acc = tot_acc_r1 / cfg.n_exp
        mean_r2_acc = tot_acc_r2 / cfg.n_exp
        mean_r1_kap = tot_kap_r1 / cfg.n_exp
        mean_r2_kap = tot_kap_r2 / cfg.n_exp
        mean_overall_acc_r1 = tot_overall_acc_r1 / cfg.n_exp
        mean_overall_acc_r2 = tot_overall_acc_r2 / cfg.n_exp
        mean_overall_kap_r1 = tot_overall_kap_r1 / cfg.n_exp
        mean_overall_kap_r2 = tot_overall_kap_r2 / cfg.n_exp
        print(f"[{tag}]  ACC (r1 | r2) : "
            f"{np.round(mean_r1_acc,4)} | {np.round(mean_r2_acc,4)}")
        print(f"[{tag}]  QWK (r1 | r2) : "
            f"{np.round(mean_r1_kap,4)} | {np.round(mean_r2_kap,4)}")

        print(f"[{tag}]  Overall  ACC  r1 {mean_r1_acc.mean():.4f} | "
            f"r2 {mean_r2_acc.mean():.4f}")
        print(f"[{tag}]  Overall  QWK  r1 {mean_r1_kap.mean():.4f} | "
            f"r2 {mean_r2_kap.mean():.4f}")
        print(f"[{tag}]  Overall  ACC  r1 {mean_overall_acc_r1:.4f} | "
            f"r2 {mean_overall_acc_r2:.4f}")
        print(f"[{tag}]  Overall  QWK  r1 {mean_overall_kap_r1:.4f} | "
            f"r2 {mean_overall_kap_r2:.4f}")


if __name__ == "__main__":
    main()
