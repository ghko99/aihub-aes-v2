from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Tuple

import torch
from kobert_transformers import get_kobert_model, get_tokenizer
import pandas as pd
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────
# 데이터 전처리
# ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./aihub")
EMB_DIR = Path("./emb")
EMB_DIR.mkdir(parents=True, exist_ok=True)  # 출력 폴더 자동 생성

PROMPT_COL = "question@prompt"
TEXT_COL = "answer@text"
SENT_SEP = "#@문장구분#"
LABEL_SEP = "###"


def load_split_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CSV 세 분할을 DataFrame으로 불러온다."""
    read_csv = lambda name: pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")
    return read_csv("train"), read_csv("valid"), read_csv("test")


def sentence_tokenize(df: pd.DataFrame, label: bool = False) -> List[List[str]]:
    """
    answer@text를 문장 단위로 분리하고,
    필요 시 앞에 question@prompt를 붙인다.
    """
    essays = df[TEXT_COL].str.split(SENT_SEP)
    if not label:
        return essays.tolist()

    # 라벨이 필요한 경우
    prompts = df[PROMPT_COL]
    labeled: List[List[str]] = []
    for prompt, essay in zip(prompts, essays):
        labeled.append([f"{prompt}{LABEL_SEP}{sent}" for sent in essay])
    return labeled


def get_datasets(label: bool = False) -> Tuple[List[List[str]], ...]:
    """train/valid/test 3-tuple 반환."""
    train, valid, test = load_split_files()
    return (
        sentence_tokenize(train, label),
        sentence_tokenize(valid, label),
        sentence_tokenize(test, label),
    )


# ────────────────────────────────────────────────────────────────────
# 임베딩 추출
# ────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def embed_and_save(
    model: torch.nn.Module,
    tokenizer,
    essays: List[List[str]],
    model_tag: str,
    split: str,
    label: bool = False,
) -> None:
    """
    각 essay(문장 리스트)를 KoBERT로 임베딩 후 CSV 저장.
    출력 파일은 ./emb/{(labeled_)model_tag}_emb_feat_{split}.csv 형태.
    """
    max_len = 70 if label else 50
    file_prefix = "labeled_" if label else ""
    out_path = EMB_DIR / f"{file_prefix}{model_tag}_emb_feat_{split}.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)

        for essay in tqdm(essays, desc=f"[{split}] extracting"):
            inputs = tokenizer.batch_encode_plus(
                essay, max_length=max_len, truncation=True, padding="max_length"
            )
            input_ids = torch.as_tensor(inputs["input_ids"], device="cuda")
            attention_mask = torch.as_tensor(inputs["attention_mask"], device="cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs[0][:, 0, :].cpu().numpy()  # (batch, hidden)
            writer.writerows(cls_emb)

            torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────────────
# 실행부
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    model = get_kobert_model().cuda().eval()
    tokenizer = get_tokenizer()

    # 1) 토픽 라벨 X
    for split_name, dataset in zip(("train", "valid", "test"), get_datasets(False)):
        embed_and_save(model, tokenizer, dataset, "kobert", split_name, label=False)

    # 2) 토픽 라벨 O
    for split_name, dataset in zip(("train", "valid", "test"), get_datasets(True)):
        embed_and_save(model, tokenizer, dataset, "kobert", split_name, label=True)


if __name__ == "__main__":
    main()
