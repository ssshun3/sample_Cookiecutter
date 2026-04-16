from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from loguru import logger
import typer

from project.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

FEATURE_COLS = ["age", "fare", "pclass", "sex", "embarked", "age_bin", "log_fare", "pclass_sex"]


@app.command()
def main(
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    submission_path: Path = PROCESSED_DATA_DIR / "submission.csv",
):
    logger.info("モデルを読み込んでいます...")
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    logger.info("テストデータを読み込んでいます...")
    test = pd.read_csv(test_features_path)
    X_test = test[FEATURE_COLS]

    # 全foldのモデルで予測して平均を取る
    preds = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

    # submission 作成
    test_raw = pd.read_csv(RAW_DATA_DIR / "test.csv")
    submission = pd.DataFrame({"id": test_raw["id"], "target": preds})
    submission.to_csv(submission_path, index=False)
    logger.success(f"submission を {submission_path} に保存しました (shape: {submission.shape})")


if __name__ == "__main__":
    app()
