from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from project.config import PROCESSED_DATA_DIR

app = typer.Typer()

FEATURE_COLS = ["age", "fare", "pclass", "sex", "embarked"]


@app.command()
def main(
    train_path: Path = PROCESSED_DATA_DIR / "train_cleaned.csv",
    test_path: Path = PROCESSED_DATA_DIR / "test_cleaned.csv",
    output_dir: Path = PROCESSED_DATA_DIR,
):
    logger.info("特徴量を生成しています...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    for df in [train, test]:
        # 年齢区分
        df["age_bin"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100], labels=False)
        # 運賃の対数変換
        df["log_fare"] = df["fare"].clip(lower=0.01).apply(lambda x: x ** 0.5)
        # 客室等級 × 性別の交互作用
        df["pclass_sex"] = df["pclass"] * 10 + df["sex"]

    feature_cols = FEATURE_COLS + ["age_bin", "log_fare", "pclass_sex"]

    train[feature_cols + ["target"]].to_csv(output_dir / "train_features.csv", index=False)
    test[feature_cols].to_csv(output_dir / "test_features.csv", index=False)
    logger.success(f"特徴量を {output_dir} に保存しました (cols: {feature_cols})")


if __name__ == "__main__":
    app()
