from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from project.config import INTERIM_DATA_DIR, RAW_DATA_DIR

META_COLS = ["sample number", "species number", "樹種", "含水率"]
TARGET_SPECIES = ["ベイスギ", "チェリー", "トチ"]


def load_train() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    train = pd.read_csv(RAW_DATA_DIR / "train.csv", encoding="shift-jis")
    wave_cols = [c for c in train.columns if c not in META_COLS]
    wavenumbers = np.array(wave_cols, dtype=float)
    return train, wave_cols, wavenumbers


def nearest_col(wave_cols: list[str], wavenumbers: np.ndarray, target_wn: float) -> tuple[str, float]:
    idx = np.argmin(np.abs(wavenumbers - target_wn))
    return wave_cols[idx], wavenumbers[idx]


def detect_baisugi_anomaly(df_sp: pd.DataFrame, wave_cols: list[str]) -> np.ndarray:
    x = df_sp[wave_cols].to_numpy(dtype=float)
    left_edge = x[:, :40].mean(axis=1) - x[:, 100:250].mean(axis=1)
    right_edge = x[:, -40:].mean(axis=1) - x[:, -250:-100].mean(axis=1)
    edge_score = np.maximum(left_edge, right_edge)
    low_moisture = df_sp["含水率"].to_numpy() <= np.percentile(df_sp["含水率"], 45)
    return low_moisture & (edge_score >= np.percentile(edge_score, 90))


def detect_cherry_anomaly(df_sp: pd.DataFrame, low_wn_col: str) -> np.ndarray:
    absorbance = df_sp[low_wn_col].to_numpy(dtype=float)
    moisture = df_sp["含水率"].to_numpy(dtype=float)
    slope, intercept, *_ = stats.linregress(absorbance, moisture)
    residual = moisture - (slope * absorbance + intercept)
    high_absorbance = absorbance >= np.percentile(absorbance, 80)
    return high_absorbance & (residual <= -1.4 * residual.std())


def detect_tochi_anomaly(df_sp: pd.DataFrame, band_col: str) -> np.ndarray:
    absorbance = df_sp[band_col].to_numpy(dtype=float)
    baseline = IsotonicRegression(increasing=False).fit_transform(
        np.arange(len(absorbance)), absorbance
    )
    residual = absorbance - baseline
    low_moisture = df_sp["含水率"].to_numpy(dtype=float) <= np.percentile(df_sp["含水率"], 40)
    return low_moisture & (residual >= 1.2 * residual.std())


def species_anomaly_mask(
    species: str, df_sp: pd.DataFrame, wave_cols: list[str], low_wn_col: str, peak5200_col: str
) -> np.ndarray:
    if species == "ベイスギ":
        return detect_baisugi_anomaly(df_sp, wave_cols)
    if species == "チェリー":
        return detect_cherry_anomaly(df_sp, low_wn_col)
    if species == "トチ":
        return detect_tochi_anomaly(df_sp.sort_values("sample number").reset_index(drop=True), peak5200_col)
    return np.zeros(len(df_sp), dtype=bool)


def build_plot() -> Path:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "MS Gothic"

    train, wave_cols, wavenumbers = load_train()
    lowest_col, lowest_wn = wave_cols[-1], wavenumbers[-1]
    highest_col, highest_wn = wave_cols[0], wavenumbers[0]
    peak6900_col, peak6900_wn = nearest_col(wave_cols, wavenumbers, 6900)
    peak5200_col, peak5200_wn = nearest_col(wave_cols, wavenumbers, 5200)

    target_cols = [
        (f"最高波数\n{highest_wn:.1f} cm⁻¹", highest_col),
        (f"6900帯\n{peak6900_wn:.1f} cm⁻¹", peak6900_col),
        (f"5200帯\n{peak5200_wn:.1f} cm⁻¹", peak5200_col),
        (f"最低波数\n{lowest_wn:.1f} cm⁻¹", lowest_col),
    ]

    fig, axes = plt.subplots(len(TARGET_SPECIES), len(target_cols), figsize=(16, 10))

    for row, species in enumerate(TARGET_SPECIES):
        df_sp = (
            train[train["樹種"] == species]
            .sort_values("sample number")
            .reset_index(drop=True)
        )
        anomaly = species_anomaly_mask(species, df_sp, wave_cols, lowest_col, peak5200_col)
        moisture = df_sp["含水率"].to_numpy(dtype=float)

        for col_idx, (label, wave_col) in enumerate(target_cols):
            ax = axes[row, col_idx]
            absorbance = df_sp[wave_col].to_numpy(dtype=float)
            ax.scatter(absorbance[~anomaly], moisture[~anomaly], s=18, alpha=0.65, color="steelblue")
            if anomaly.any():
                ax.scatter(
                    absorbance[anomaly],
                    moisture[anomaly],
                    s=38,
                    alpha=0.95,
                    color="red",
                    label=f"異常値 {anomaly.sum()}件",
                    zorder=3,
                )
                if col_idx == len(target_cols) - 1:
                    ax.legend(fontsize=8, loc="best")

            corr = np.corrcoef(absorbance, moisture)[0, 1]
            ax.set_xlabel("吸光度", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{species}\n含水率 (%)", fontsize=10)
            if row == 0:
                ax.set_title(label, fontsize=10)
            ax.text(
                0.97,
                0.03,
                f"r={corr:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
            )

    fig.suptitle(
        "3樹種の異常値検知\n"
        "ベイスギ: 低含水率で波数端が突出する点 / "
        "チェリー: 低波数かつ高吸光度で含水率が低すぎる点 / "
        "トチ: 乾燥末期の跳ね上がり",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DATA_DIR / "species_anomalies.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    saved = build_plot()
    print(saved)
