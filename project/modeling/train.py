from pathlib import Path
import pickle

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import typer

from project.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

FEATURE_COLS = ["age", "fare", "pclass", "sex", "embarked", "age_bin", "log_fare", "pclass_sex"]
TARGET_COL = "target"
N_SPLITS = 5


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("学習データを読み込んでいます...")
    df = pd.read_csv(features_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_preds = pd.Series(0.0, index=df.index)
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds.iloc[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        logger.info(f"Fold {fold+1}/{N_SPLITS} - AUC: {auc:.4f}")
        models.append(model)

    overall_auc = roc_auc_score(y, oof_preds)
    logger.info(f"OOF AUC: {overall_auc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    logger.success(f"モデルを {model_path} に保存しました")


if __name__ == "__main__":
    app()
