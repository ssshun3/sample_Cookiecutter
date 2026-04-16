# project

Signate コンペ用プロジェクトテンプレートです。

## ディレクトリ構成

```
├── Makefile
├── README.md
├── data/
│   ├── raw/                    <- Signate からダウンロードしたデータを置く
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submit.csv
│   ├── interim/                <- 処理途中のデータ
│   └── processed/              <- モデルに入力する最終データ・submission
│
├── models/                     <- 学習済みモデル (.pkl)
│
├── notebooks/                  <- EDA・実験用ノートブック
│                                  命名規則: 1.0-eda.ipynb, 2.0-feature.ipynb
│
└── project/                    <- 再利用する Python コード
    ├── config.py               <- パス定義
    ├── dataset.py              <- データ読み込み・前処理
    ├── features.py             <- 特徴量エンジニアリング
    └── modeling/
        ├── train.py            <- LightGBM 学習 (CV付き)
        └── predict.py          <- 推論・submission 生成
```

## セットアップ

```bash
make requirements
```

## 実行フロー

```bash
# 1. data/raw/ に train.csv / test.csv を置く

# 2. データ前処理
python -m project.dataset

# 3. 特徴量生成
python -m project.features

# 4. 学習 (5-fold CV)
python -m project.modeling.train

# 5. 推論・submission 生成
python -m project.modeling.predict
```

完了すると `data/processed/submission.csv` が生成されます。

## コンペ開始時にやること

各ファイルの `# ---- コンペに合わせて変更 ----` セクションを編集します。

| ファイル | 変更箇所 |
|---|---|
| `dataset.py` | `TARGET_COL`, `ID_COL`, 前処理ロジック |
| `features.py` | `TARGET_COL`, `ID_COL`, 特徴量エンジニアリング |
| `modeling/train.py` | `LGB_PARAMS`, `TARGET_COL`, `ID_COL`, metric |
| `modeling/predict.py` | `TARGET_COL`, `ID_COL`, 提出フォーマット |

## パスの使い方

```python
from project.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

import pandas as pd
train = pd.read_csv(RAW_DATA_DIR / "train.csv")
```

## Makefile コマンド

```bash
make help         # コマンド一覧
make requirements # 依存パッケージのインストール
make data         # データ前処理の実行
make format       # コードフォーマット (ruff)
make lint         # リントチェック
make clean        # キャッシュ削除
```
