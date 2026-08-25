from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from loguru import logger
import typer

from project.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# ---- コンペに合わせて変更 ----
TARGET_COL = "target"
ID_COL = "id"
# ----------------------------


@app.command()
def main(
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    submission_path: Path = PROCESSED_DATA_DIR / "submission.csv",
):
    logger.info("モデルを読み込んでいます...")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    models = saved["models"]
    feature_cols = saved["feature_cols"]

    logger.info("テストデータを読み込んでいます...")
    test = pd.read_csv(test_features_path)
    X_test = test[feature_cols]

    preds = np.mean([m.predict(X_test) for m in models], axis=0)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds})

    # ---- Signate の提出フォーマットに合わせて調整 ----
    # 二値分類（0/1）の場合: preds = (preds >= 0.5).astype(int)
    # -------------------------------------------------

    submission.to_csv(submission_path, index=False)
    logger.success(f"submission 保存完了: {submission_path} (shape: {submission.shape})")


if __name__ == "__main__":
    app()
