"""Diagnose ensemble models and examine prediction relationships."""
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import KFold

DATA_ROOT = Path(r"G:\PhyMaize")
TRAIN_VAL_FILE = DATA_ROOT / "features" / "feature_selection_results" / "20_features" / "selected_features.xlsx"
TEST_FILE = DATA_ROOT / "features" / "features-Pn (test).xlsx"
TEST_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "test.csv"
MODEL_ROOT = DATA_ROOT / "traditional_ml_results" / "20_features"
OUTPUT_ROOT = MODEL_ROOT / "ensemble_diagnostic"
ID_COL = "Sample_ID"
TARGET = "Pn"
RANDOM_STATE = 42
CV_FOLDS = 5
BASE_MODELS = ["ExtraTrees", "RF", "GBR", "HistGBR", "SVR", "BPNN", "XGB"]
FONT_FAMILY = "Arial"
DPI = 300
CMAP_PRED = "RdPu"
CMAP_RESIDUAL = "YlGn"

sns.set_theme(style="white", context="paper", font=FONT_FAMILY)
plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 15, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": DPI})

def save_figure(fig, filename):
    """Generate or save figure."""
    fig.savefig(OUTPUT_ROOT / f"{filename}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{filename}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def short_model_name(name):
    """Perform short model name."""
    return {"ExtraTrees": "ET", "RF": "RF", "GBR": "GBR", "HistGBR": "HistGBR", "SVR": "SVR", "BPNN": "BPNN", "XGB": "XGB"}.get(name, name)

def adaptive_limits(corr, lower_pad=0.02):
    """Perform adaptive limits."""
    values = corr.values[np.triu_indices_from(corr, k=1)]
    values = values[np.isfinite(values)]
    vmin = max(-1.0, np.floor((values.min() - lower_pad) * 20) / 20)
    vmax = 1.0
    if vmax - vmin < 0.2: vmin = max(-1.0, vmax - 0.2)
    return vmin, vmax

def load_data():
    """Retrieve or load data for subsequent analysis."""
    train_df = pd.read_excel(TRAIN_VAL_FILE)
    test_df = pd.read_excel(TEST_FILE)
    split_df = pd.read_csv(TEST_SPLIT_FILE)
    for df, name in [(train_df, "training"), (test_df, "test")]:
        if ID_COL not in df.columns: raise ValueError(f"{ID_COL} not found in {name} file.")
        if TARGET not in df.columns: raise ValueError(f"{TARGET} not found in {name} file.")
    train_df = train_df.drop_duplicates(ID_COL).copy()
    test_df = test_df.drop_duplicates(ID_COL).copy()
    overlap = set(train_df[ID_COL]) & set(test_df[ID_COL])
    if overlap: raise ValueError(f"Train/Test overlap detected: {len(overlap)} samples.")
    test_ids = split_df[ID_COL].tolist()
    missing_test_ids = set(test_ids) - set(test_df[ID_COL])
    if missing_test_ids: raise ValueError(f"{len(missing_test_ids)} test IDs are missing from test file.")
    test_df = test_df.set_index(ID_COL).loc[test_ids].reset_index()
    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET]]
    missing = set(feature_cols) - set(test_df.columns)
    if missing: raise ValueError(f"Missing test features: {missing}")
    train_df = train_df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    test_df = test_df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    return train_df[feature_cols].copy(), train_df[TARGET].values, test_df[feature_cols].copy(), test_df[TARGET].values, feature_cols, train_df, test_df

def load_models():
    """Retrieve or load models for subsequent analysis."""
    models = {}
    for model_name in BASE_MODELS:
        model_file = MODEL_ROOT / model_name / "best_model.joblib"
        if not model_file.exists():
            print(f"[Warning] {model_name}: best_model.joblib not found. Skipped.")
            continue
        try:
            models[model_name] = joblib.load(model_file)
            print(f"[Loaded] {model_name}")
        except Exception as e:
            print(f"[Warning] Failed to load {model_name}: {e}")
    if "ExtraTrees" not in models: raise FileNotFoundError("ExtraTrees best_model.joblib is required.")
    return models

def generate_oof_predictions(models, X, y):
    """Perform generate oof predictions."""
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.DataFrame(index=np.arange(len(y)), columns=list(models.keys()), dtype=float)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\nOOF Fold {fold}/{CV_FOLDS}")
        for model_name, model in models.items():
            print(f"  {model_name}", end=" ")
            estimator = clone(model)
            estimator.fit(X.iloc[train_idx], y[train_idx])
            oof.loc[val_idx, model_name] = estimator.predict(X.iloc[val_idx])
            print("done")
    return oof

def plot_correlation_heatmap(corr, cmap, title, filename):
    """Generate or save correlation heatmap."""
    labels = [short_model_name(c) for c in corr.columns]
    plot_corr = corr.copy()
    plot_corr.index = labels
    plot_corr.columns = labels
    mask = np.triu(np.ones_like(plot_corr, dtype=bool), k=1)
    vmin, vmax = adaptive_limits(corr)
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    sns.heatmap(plot_corr, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, square=True, linewidths=0.8, linecolor="white", annot=True, fmt=".3f", annot_kws={"fontsize": 15, "fontfamily": FONT_FAMILY}, cbar_kws={"label": "Pearson correlation", "shrink": 0.82, "pad": 0.03}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.tick_params(axis="both", rotation=0, length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, filename)

def save_prediction_correlation(oof):
    """Generate or save prediction correlation."""
    corr = oof.corr(method="pearson")
    corr.to_excel(OUTPUT_ROOT / "oof_prediction_correlation.xlsx")
    plot_correlation_heatmap(corr, CMAP_PRED, "OOF prediction correlation", "oof_prediction_correlation")
    return corr

def save_residual_correlation(oof, y):
    """Generate or save residual correlation."""
    residuals = y[:, None] - oof.values
    residuals = pd.DataFrame(residuals, columns=oof.columns, index=oof.index)
    corr = residuals.corr(method="pearson")
    residuals.to_excel(OUTPUT_ROOT / "oof_residuals.xlsx", index=False)
    corr.to_excel(OUTPUT_ROOT / "oof_residual_correlation.xlsx")
    plot_correlation_heatmap(corr, CMAP_RESIDUAL, "OOF residual correlation", "oof_residual_correlation")
    return residuals, corr

def main():
    """Run the main analysis workflow."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("PhyMaize Ensemble Correlation Diagnostics")
    print("=" * 72)
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data()
    print(f"Train samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Features      : {len(feature_cols)}")
    print(f"Output        : {OUTPUT_ROOT}")
    print("\nLoading models...")
    models = load_models()
    print("\nGenerating OOF predictions...")
    oof = generate_oof_predictions(models, X_train, y_train)
    oof.insert(0, ID_COL, train_df[ID_COL].values)
    oof.to_excel(OUTPUT_ROOT / "oof_predictions.xlsx", index=False)
    oof_values = oof.drop(columns=[ID_COL])
    print("\nCalculating OOF prediction correlation...")
    prediction_corr = save_prediction_correlation(oof_values)
    print("\nCalculating OOF residual correlation...")
    residuals, residual_corr = save_residual_correlation(oof_values, y_train)
    print("\nPrediction correlation:")
    print(prediction_corr.round(4).to_string())
    print("\nResidual correlation:")
    print(residual_corr.round(4).to_string())
    print("\n" + "=" * 72)
    print("Correlation diagnostics completed.")
    print("=" * 72)
    print(f"Results saved to:\n{OUTPUT_ROOT}")
    print("=" * 72)

if __name__ == "__main__":
    main()