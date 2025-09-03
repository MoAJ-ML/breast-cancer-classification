import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dirs(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    # Fallback to sklearn dataset (column names differ from UCI but fine for quick plots)
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=list(data.feature_names))
    # sklearn: target_names = ["malignant", "benign"], encoded as 0,1 respectively.
    # Project convention: malignant = 1, benign = 0
    df['target'] = pd.Series(data.target).map({0: 1, 1: 0})
    return df


def save_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df['target'].value_counts().rename(index={0: 'Benign (0)', 1: 'Malignant (1)'})
    plt.figure(figsize=(5, 4))
    counts.plot(kind='bar', color=['#69b3a2', '#d95f02'])
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / 'class_distribution.png', dpi=150)
    plt.close()


def save_histograms(df: pd.DataFrame, out: Path) -> None:
    # pick a few known sklearn names
    cols = [c for c in ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'worst radius'] if c in df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 3))
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i], color='#4C72B0')
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(out / 'histograms_selected.png', dpi=150)
    plt.close()


def save_correlation(df: pd.DataFrame, out: Path) -> None:
    num_cols = [c for c in df.columns if c != 'target']
    corr = df[num_cols + ['target']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap='RdBu_r', center=0)
    plt.title('Correlation Heatmap (Sample)')
    plt.tight_layout()
    plt.savefig(out / 'correlation_heatmap.png', dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    figures_dir = project_root / 'reports' / 'figures'
    ensure_dirs(figures_dir)
    df = load_data()
    save_class_distribution(df, figures_dir)
    save_histograms(df, figures_dir)
    save_correlation(df, figures_dir)
    print(f'Saved basic plots to {figures_dir}')


if __name__ == '__main__':
    main()


