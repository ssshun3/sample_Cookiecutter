from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train.csv",
    test_path: Path = RAW_DATA_DIR / "test.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    logger.info("データを読み込んでいます...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info(f"train shape: {train.shape}, test shape: {test.shape}")

    # 欠損値補完
    for df in [train, test]:
        df["age"] = df["age"].fillna(df["age"].median())
        df["fare"] = df["fare"].fillna(df["fare"].median())
        df["embarked"] = df["embarked"].fillna("S")

    # カテゴリ変数をラベルエンコード
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    for df in [train, test]:
        df["embarked"] = df["embarked"].map(embarked_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train_cleaned.csv", index=False)
    test.to_csv(output_dir / "test_cleaned.csv", index=False)
    logger.success(f"前処理済みデータを {output_dir} に保存しました")


if __name__ == "__main__":
    app()
