from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from project.config import PROCESSED_DATA_DIR

app = typer.Typer()

# ---- コンペに合わせて変更 ----
TARGET_COL = "target"
ID_COL = "id"
# ----------------------------


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_cleaned.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_cleaned.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    logger.info("特徴量を生成しています...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # ---- 特徴量エンジニアリングをここに書く ----

    # カテゴリ列をラベルエンコード
    cat_cols = train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(train[col].unique()))}
        train[col] = train[col].map(mapping)
        test[col] = test[col].map(mapping)

    # -------------------------------------------

    feature_cols = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]

    train[[ID_COL] + feature_cols + [TARGET_COL]].to_csv(
        output_dir / "train_features.csv", index=False
    )
    test[[ID_COL] + feature_cols].to_csv(
        output_dir / "test_features.csv", index=False
    )
    logger.success(f"特徴量を保存しました: {feature_cols}")


if __name__ == "__main__":
    app()
