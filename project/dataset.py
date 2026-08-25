from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# ---- コンペに合わせて変更 ----
TARGET_COL = "target"
ID_COL = "id"
# ----------------------------


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train.csv",
    test_path: Path = RAW_DATA_DIR / "test.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    logger.info("データを読み込んでいます...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info(f"train: {train.shape}, test: {test.shape}")

    # ---- 前処理をここに書く ----

    # 数値列の欠損値補完
    num_cols = train.select_dtypes(include="number").columns.drop(TARGET_COL, errors="ignore")
    for col in num_cols:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)

    # カテゴリ列の欠損値補完
    cat_cols = train.select_dtypes(include="object").columns
    for col in cat_cols:
        mode = train[col].mode()[0]
        train[col] = train[col].fillna(mode)
        test[col] = test[col].fillna(mode)

    # ---------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train_cleaned.csv", index=False)
    test.to_csv(output_dir / "test_cleaned.csv", index=False)
    logger.success(f"保存完了: {output_dir}")


if __name__ == "__main__":
    app()
