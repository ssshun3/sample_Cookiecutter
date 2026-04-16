# project

コンペ用の最小構成プロジェクトテンプレートです。

## ディレクトリ構成

```
├── Makefile                    <- よく使うコマンドのショートカット
├── README.md
├── data/
│   ├── raw/                    <- 元データ（変更しない）
│   │   ├── train.csv
│   │   └── test.csv
│   ├── interim/                <- 処理途中のデータ
│   └── processed/              <- モデルに入力する最終データ・submission
│
├── models/                     <- 学習済みモデル (.pkl)
│
├── notebooks/                  <- EDA・実験用ノートブック
│                                  命名規則: 1.0-eda.ipynb, 2.0-feature.ipynb
│
└── project/                    <- 再利用する Python コード
    ├── config.py               <- パス定義（変更不要）
    ├── dataset.py              <- データ読み込み・前処理
    ├── features.py             <- 特徴量エンジニアリング
    └── modeling/
        ├── train.py            <- 学習 (CV付き)
        └── predict.py          <- 推論・submission 生成
```

## セットアップ

```bash
make requirements
```

## 実行フロー

```bash
# 1. データ前処理
python -m project.dataset

# 2. 特徴量生成
python -m project.features

# 3. 学習 (5-fold CV)
python -m project.modeling.train

# 4. 推論・submission 生成
python -m project.modeling.predict
```

完了すると `data/processed/submission.csv` が生成されます。

## パスの使い方

`project/config.py` にパスがすべて定義されています。
ノートブックからも以下のようにインポートして使えます：

```python
from project.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

import pandas as pd
train = pd.read_csv(RAW_DATA_DIR / "train.csv")
```

## 仮データについて

`data/raw/` にはタイタニック風の二値分類ダミーデータが入っています。

| ファイル | 行数 | 説明 |
|---|---|---|
| train.csv | 800 | 学習データ（target列あり） |
| test.csv | 200 | テストデータ（target列なし） |

**カラム:** `id`, `age`, `fare`, `pclass`, `sex`, `embarked`, `target`

## Makefile コマンド

```bash
make help         # コマンド一覧
make requirements # 依存パッケージのインストール
make data         # データ前処理の実行
make format       # コードフォーマット (ruff)
make lint         # リントチェック
make clean        # キャッシュ削除
```
