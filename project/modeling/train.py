from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import typer

from project.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# ---- コンペに合わせて変更 ----
TARGET_COL = "target"
ID_COL = "id"
N_SPLITS = 5
SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": SEED,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
# ----------------------------


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("学習データを読み込んでいます...")
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(df))
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        logger.info(f"Fold {fold + 1}/{N_SPLITS} - AUC: {auc:.4f}")
        models.append(model)

    oof_auc = roc_auc_score(y, oof_preds)
    logger.info(f"OOF AUC: {oof_auc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feature_cols}, f)
    logger.success(f"モデルを保存しました: {model_path}")


if __name__ == "__main__":
    app()
