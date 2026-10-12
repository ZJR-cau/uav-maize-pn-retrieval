"""Measure the contributions of feature groups through ablation experiments."""
# ================================================================
# feature_group_ablation.py
# Publication-quality Feature Group Ablation
# Fixed ET_RidgeResidual_GBR+XGB Ensemble
# ================================================================
import warnings
warnings.filterwarnings("ignore")
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

DATA_ROOT = Path(r"G:\PhyMaize")
TRAIN_VAL_FILE = DATA_ROOT / "features" / "feature_selection_results" / "20_features" / "selected_features.xlsx"
TEST_FILE = DATA_ROOT / "features" / "features-Pn (test).xlsx"
TEST_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "test.csv"
MODEL_ROOT = DATA_ROOT / "traditional_ml_results" / "20_features"
OUTPUT_ROOT = MODEL_ROOT / "feature_group_ablation_v2"

ID_COL = "Sample_ID"
TARGET = "Pn"
EXPECTED_FEATURE_NUMBER = 20
RANDOM_STATE = 42
N_SPLITS = 5
RIDGE_ALPHA = 10.0
FIG_DPI = 600
FONT_FAMILY = "Arial"
ENSEMBLE_STRATEGY = "ET_RidgeResidual_GBR+XGB"
BASE_MODEL_NAMES = ["ExtraTrees", "GBR", "XGB"]
FIG_FORMATS = ["png", "tiff", "pdf", "svg"]

plt.rcParams.update({"font.family": FONT_FAMILY, "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11, "axes.linewidth": 0.9, "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 4, "ytick.major.size": 4, "legend.fontsize": 9, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": FIG_DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.08})

COLOR_MS = "#4C78A8"
COLOR_THERMAL = "#E45756"
COLOR_3D = "#59A14F"
COLOR_MS_THERMAL = "#7A6FA8"
COLOR_MS_3D = "#4F8F73"
COLOR_THERMAL_3D = "#B46A6A"
COLOR_FULL = "#333333"
COLOR_NEUTRAL = "#A7A7A7"

GROUP_COLORS = {"MS": COLOR_MS, "Thermal": COLOR_THERMAL, "3D": COLOR_3D, "MS+Thermal": COLOR_MS_THERMAL, "MS+3D": COLOR_MS_3D, "Thermal+3D": COLOR_THERMAL_3D, "All features (20)": COLOR_FULL}
DISPLAY_NAMES = {"MS": "Multispectral", "Thermal": "Thermal Infrared", "3D": "3D Point Cloud", "MS+Thermal": "MS + Thermal", "MS+3D": "MS + 3D", "Thermal+3D": "Thermal + 3D", "All features (20)": "All features (20)"}

GROUP_DISPLAY_NAMES = {
    "LiDAR_Statistics": "LiDAR Statistics",
    "Multispectral_Statistics": "Multispectral Statistics",
    "Multispectral_Texture": "Multispectral Texture",
    "3D_Structure_Phenotype": "3D Structure Phenotype",
    "Temperature_Index": "Temperature Index",
    "Vertical_Structure_Phenotype": "Vertical Structure Phenotype",
    "Thermal_Infrared_Texture": "Thermal Infrared Texture",
    "Thermal_Infrared_Statistics": "Thermal Infrared Statistics",
    "Vegetation_Index": "Vegetation Index"
}

FEATURE_SOURCE = {
    "LiDAR_intensity_mean": "3D Point Cloud",
    "Green_mean": "Multispectral",
    "Green_std": "Multispectral",
    "Green_homogeneity": "Multispectral",
    "COV": "3D Point Cloud",
    "CWSI": "Thermal Infrared",
    "Green_correlation": "Multispectral",
    "Canopy_S": "3D Point Cloud",
    "Canopy_Volume": "3D Point Cloud",
    "LiDAR_intensity_std": "3D Point Cloud",
    "Thermal_energy": "Thermal Infrared",
    "Thermal_dissimilarity": "Thermal Infrared",
    "NIR_contrast": "Multispectral",
    "Green_contrast": "Multispectral",
    "Thermal_correlation": "Thermal Infrared",
    "Canopy_LAI": "3D Point Cloud",
    "Temp_min": "Thermal Infrared",
    "MTCI_std": "Multispectral",
    "CIrededge_std": "Multispectral",
    "Canopy_K": "3D Point Cloud"
}

FEATURE_VARIETY = {
    "LiDAR_intensity_mean": "LiDAR_Statistics",
    "Green_mean": "Multispectral_Statistics",
    "Green_std": "Multispectral_Statistics",
    "Green_homogeneity": "Multispectral_Texture",
    "COV": "3D_Structure_Phenotype",
    "CWSI": "Temperature_Index",
    "Green_correlation": "Multispectral_Texture",
    "Canopy_S": "Vertical_Structure_Phenotype",
    "Canopy_Volume": "3D_Structure_Phenotype",
    "LiDAR_intensity_std": "LiDAR_Statistics",
    "Thermal_energy": "Thermal_Infrared_Texture",
    "Thermal_dissimilarity": "Thermal_Infrared_Texture",
    "NIR_contrast": "Multispectral_Texture",
    "Green_contrast": "Multispectral_Texture",
    "Thermal_correlation": "Thermal_Infrared_Texture",
    "Canopy_LAI": "3D_Structure_Phenotype",
    "Temp_min": "Thermal_Infrared_Statistics",
    "MTCI_std": "Vegetation_Index",
    "CIrededge_std": "Vegetation_Index",
    "Canopy_K": "Vertical_Structure_Phenotype"
}

def save_figure(fig, filename):
    """Generate or save figure."""
    output_base = OUTPUT_ROOT / filename
    for fmt in FIG_FORMATS:
        path = output_base.with_suffix(f".{fmt}")
        kwargs = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.08}
        if fmt not in ["pdf", "svg"]: kwargs["dpi"] = FIG_DPI
        fig.savefig(path, **kwargs)
    plt.close(fig)

def clean_axes(ax):
    """Perform clean axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", direction="out")

def metric_dict(y_true, y_pred):
    """Perform metric dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"Test_R2": r2_score(y_true, y_pred), "Test_RMSE": rmse, "Test_rRMSE": rmse / np.mean(np.abs(y_true)) * 100, "Test_MAE": mean_absolute_error(y_true, y_pred)}

def serialized_size_mb(obj):
    """Perform serialized size mb."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer, compress=0)
    return buffer.tell() / (1024 ** 2)

# ================================================================
# Data Loading
# ================================================================

def load_data():
    """Retrieve or load data for subsequent analysis."""
    train_df = pd.read_excel(TRAIN_VAL_FILE)
    test_df = pd.read_excel(TEST_FILE)
    split_df = pd.read_csv(TEST_SPLIT_FILE)
    train_df[ID_COL] = train_df[ID_COL].astype(str)
    test_df[ID_COL] = test_df[ID_COL].astype(str)
    split_df[ID_COL] = split_df[ID_COL].astype(str)
    for column in [ID_COL, TARGET]:
        if column not in train_df.columns: raise ValueError(f"{column} not found in training file.")
        if column not in test_df.columns: raise ValueError(f"{column} not found in test file.")
    train_df = train_df.drop_duplicates(ID_COL).copy()
    test_df = test_df.drop_duplicates(ID_COL).copy()
    overlap = set(train_df[ID_COL]) & set(test_df[ID_COL])
    if overlap: raise ValueError(f"Train/Test overlap detected: {len(overlap)} samples.")
    test_ids = split_df[ID_COL].tolist()
    missing_test_ids = set(test_ids) - set(test_df[ID_COL])
    if missing_test_ids: raise ValueError(f"{len(missing_test_ids)} test IDs are missing from test feature file.")
    test_df = test_df.set_index(ID_COL).loc[test_ids].reset_index()
    feature_cols = [c for c in train_df.columns if c not in [ID_COL, TARGET]]
    missing_features = set(feature_cols) - set(test_df.columns)
    if missing_features: raise ValueError(f"Missing test features: {missing_features}")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    test_df = test_df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET].to_numpy(dtype=float)
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET].to_numpy(dtype=float)
    return X_train, y_train, X_test, y_test, feature_cols, train_df, test_df

# ================================================================
# Load Optimal Base Models
# ================================================================

def find_model_file(model_name):
    """Retrieve or load model file for subsequent analysis."""
    candidates = [MODEL_ROOT / model_name / "best_model.joblib", MODEL_ROOT / model_name / "best_model.pkl", MODEL_ROOT / model_name / "model.joblib", MODEL_ROOT / model_name / "model.pkl"]
    for path in candidates:
        if path.exists(): return path
    raise FileNotFoundError(f"No saved best model found for {model_name}.\nSearched:\n" + "\n".join(str(x) for x in candidates))

def load_best_ensemble():
    """Retrieve or load best ensemble for subsequent analysis."""
    models = {}
    sources = {}
    print("\n" + "=" * 80)
    print("Loading Fixed Ensemble Base Models")
    print("=" * 80)
    print(f"Fixed strategy     : {ENSEMBLE_STRATEGY}")
    print(f"Anchored model     : ExtraTrees")
    print(f"Complementary     : GBR + XGB")
    print(f"Residual model     : Ridge")
    print(f"Ridge alpha        : {RIDGE_ALPHA}")
    for name in BASE_MODEL_NAMES:
        path = find_model_file(name)
        model = joblib.load(path)
        if isinstance(model, dict):
            if "model" in model: model = model["model"]
            elif "estimator" in model: model = model["estimator"]
            elif "best_model" in model: model = model["best_model"]
            else: raise TypeError(f"{name} file contains a dictionary but no recognized model key: {list(model.keys())}")
        if not hasattr(model, "fit") or not hasattr(model, "predict"): raise TypeError(f"{name} file does not contain a valid sklearn estimator: {type(model)}")
        models[name] = model
        sources[name] = str(path)
        print(f"  {name:<20}: {type(model).__name__} <- {path}")
    return models, sources

# ================================================================
# Feature Groups
# ================================================================

def define_feature_groups(feature_cols):
    """Perform define feature groups."""
    groups = {}
    for feature in feature_cols:
        if feature not in FEATURE_VARIETY: raise ValueError(f"Feature '{feature}' has no defined variety.")
        groups.setdefault(FEATURE_VARIETY[feature], []).append(feature)
    expected_features = set(FEATURE_SOURCE)
    actual_features = set(feature_cols)
    missing_defined = expected_features - actual_features
    unknown_features = actual_features - expected_features
    if missing_defined: raise ValueError(f"Defined features missing from selected feature file: {sorted(missing_defined)}")
    if unknown_features: raise ValueError(f"Unknown selected features not included in feature definition: {sorted(unknown_features)}")
    if len(feature_cols) != EXPECTED_FEATURE_NUMBER: raise ValueError(f"Expected {EXPECTED_FEATURE_NUMBER} selected features, got {len(feature_cols)}.")
    assigned = [f for fs in groups.values() for f in fs]
    duplicated = pd.Series(assigned).value_counts()
    duplicated = duplicated[duplicated > 1].index.tolist()
    if duplicated: raise ValueError(f"Features assigned to multiple groups: {duplicated}")
    return groups

# ================================================================
# Experiment Definition
# ================================================================

def define_experiments(feature_cols):
    """Perform define experiments."""
    ms = [f for f in feature_cols if FEATURE_SOURCE[f] == "Multispectral"]
    thermal = [f for f in feature_cols if FEATURE_SOURCE[f] == "Thermal Infrared"]
    structure = [f for f in feature_cols if FEATURE_SOURCE[f] == "3D Point Cloud"]
    experiments = {"MS": ms, "Thermal": thermal, "3D": structure, "MS+Thermal": ms + thermal, "MS+3D": ms + structure, "Thermal+3D": thermal + structure, "All features (20)": feature_cols.copy()}
    return {name: list(dict.fromkeys(features)) for name, features in experiments.items()}

def validate_experiments(experiments, feature_cols):
    """Validate experiments."""
    feature_set = set(feature_cols)
    for name, features in experiments.items():
        if not features: raise ValueError(f"Experiment '{name}' contains no features.")
        unknown = set(features) - feature_set
        if unknown: raise ValueError(f"Experiment '{name}' contains unknown features: {unknown}")
    if len(experiments["All features (20)"]) != EXPECTED_FEATURE_NUMBER: raise ValueError("All features experiment does not contain 20 features.")
    for name in ["MS", "Thermal", "3D"]: print(f"{name} feature number: {len(experiments[name])}")

# ================================================================
# OOF Base Predictions
# ================================================================

def generate_oof_predictions(X, y, features, models):
    """Perform generate oof predictions."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.zeros(len(X), dtype=float) for name in BASE_MODEL_NAMES}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        print(f"    OOF fold {fold}/{N_SPLITS}")
        X_tr, X_va, y_tr = X.iloc[tr_idx][features], X.iloc[va_idx][features], y[tr_idx]
        for name in BASE_MODEL_NAMES:
            model = clone(models[name])
            model.fit(X_tr, y_tr)
            oof[name][va_idx] = model.predict(X_va)
    return pd.DataFrame(oof)

# ================================================================
# Full Development Fit
# ================================================================

def fit_full_base_models(X, y, X_test, features, models):
    """Perform fit full base models."""
    fitted = {}
    test_predictions = {}
    for name in BASE_MODEL_NAMES:
        print(f"    Full-development fit: {name}")
        model = clone(models[name])
        model.fit(X[features], y)
        fitted[name] = model
        test_predictions[name] = model.predict(X_test[features])
    return fitted, pd.DataFrame(test_predictions)

# ================================================================
# Fixed ET + Ridge Residual + GBR/XGB
# ================================================================

def fit_fixed_ensemble(y, oof_df, test_pred_df):
    """Perform fit fixed ensemble."""
    et_oof = oof_df["ExtraTrees"].to_numpy()
    gbr_oof = oof_df["GBR"].to_numpy()
    xgb_oof = oof_df["XGB"].to_numpy()
    et_test = test_pred_df["ExtraTrees"].to_numpy()
    gbr_test = test_pred_df["GBR"].to_numpy()
    xgb_test = test_pred_df["XGB"].to_numpy()
    X_meta = np.column_stack([gbr_oof - et_oof, xgb_oof - et_oof])
    X_meta_test = np.column_stack([gbr_test - et_test, xgb_test - et_test])
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_meta, y - et_oof)
    pred = et_test + ridge.predict(X_meta_test)
    return pred, ridge, X_meta

# ================================================================
# Run Experiments
# ================================================================

def run_experiments(X_train, y_train, X_test, y_test, experiments, base_models, model_sources):
    """Perform run experiments."""
    results = []
    predictions = {}
    model_records = []
    for name, features in experiments.items():
        print("\n" + "=" * 80)
        print(f"Experiment: {name} | Features: {len(features)}")
        print("=" * 80)
        print("  Generating OOF base predictions...")
        oof_df = generate_oof_predictions(X_train, y_train, features, base_models)
        print("  Fitting full-development base models...")
        fitted_models, test_pred_df = fit_full_base_models(X_train, y_train, X_test, features, base_models)
        print(f"  Fitting fixed {ENSEMBLE_STRATEGY} ensemble...")
        pred, ridge_model, X_meta = fit_fixed_ensemble(y_train, oof_df, test_pred_df)
        metrics = metric_dict(y_test, pred)
        predictions[name] = pred
        results.append({"Feature_Group": name, "Feature_Number": len(features), "Features": ", ".join(features), "Ensemble_Strategy": ENSEMBLE_STRATEGY, "Ridge_Alpha": RIDGE_ALPHA, **metrics})
        model_bundle = {"Model_Type": "ExtraTrees_Anchored_Ensemble", "Strategy": ENSEMBLE_STRATEGY, "Method": "Ridge_Residual", "Selected": ("GBR", "XGB"), "Features": tuple(features), "ExtraTrees": fitted_models["ExtraTrees"], "GBR": fitted_models["GBR"], "XGB": fitted_models["XGB"], "Meta_Model": ridge_model}
        model_path = OUTPUT_ROOT / f"ensemble_model_{name.replace('+', '_').replace(' ', '_')}.joblib"
        joblib.dump(model_bundle, model_path, compress=3)
        model_records.append({"Feature_Group": name, "Model_Path": str(model_path), "Model_Size_MB": serialized_size_mb(model_bundle), "Ridge_Alpha": RIDGE_ALPHA, "ExtraTrees_Type": type(fitted_models["ExtraTrees"]).__name__, "GBR_Type": type(fitted_models["GBR"]).__name__, "XGB_Type": type(fitted_models["XGB"]).__name__})
        print(f"R²={metrics['Test_R2']:.6f} | RMSE={metrics['Test_RMSE']:.6f} | rRMSE={metrics['Test_rRMSE']:.4f}% | MAE={metrics['Test_MAE']:.6f}")
    return pd.DataFrame(results), pd.DataFrame(predictions), pd.DataFrame(model_records)

# ================================================================
# Delta Analysis
# ================================================================

def calculate_deltas(result_df):
    """Compute deltas."""
    result_df = result_df.copy()
    full = result_df[result_df["Feature_Group"] == "All features (20)"].iloc[0]
    ms = result_df[result_df["Feature_Group"] == "MS"].iloc[0]
    result_df["Delta_R2_vs_Full"] = result_df["Test_R2"] - full["Test_R2"]
    result_df["Delta_RMSE_vs_Full"] = result_df["Test_RMSE"] - full["Test_RMSE"]
    result_df["Delta_rRMSE_vs_Full"] = result_df["Test_rRMSE"] - full["Test_rRMSE"]
    result_df["Delta_MAE_vs_Full"] = result_df["Test_MAE"] - full["Test_MAE"]
    result_df["Delta_R2_vs_MS"] = result_df["Test_R2"] - ms["Test_R2"]
    return result_df

# ================================================================
# Incremental Contribution
# ================================================================

def calculate_incremental_contribution(result_df):
    """Compute incremental contribution."""
    r2 = dict(zip(result_df["Feature_Group"], result_df["Test_R2"]))
    comparisons = [("Thermal | MS", "MS", "Thermal", "MS+Thermal"), ("3D | MS", "MS", "3D", "MS+3D"), ("3D | MS+Thermal", "MS+Thermal", "3D", "All features (20)"), ("Thermal | MS+3D", "MS+3D", "Thermal", "All features (20)"), ("MS | Thermal+3D", "Thermal+3D", "MS", "All features (20)")]
    records = [{"Comparison": comparison, "Base": base, "Added": added, "Combined": combined, "Base_R2": r2[base], "Combined_R2": r2[combined], "Incremental_Delta_R2": r2[combined] - r2[base]} for comparison, base, added, combined in comparisons]
    df = pd.DataFrame(records)
    df.to_excel(OUTPUT_ROOT / "incremental_contribution.xlsx", index=False)
    return df

# ================================================================
# Pairwise Gain
# ================================================================

def calculate_pairwise_gain(result_df):
    """Compute pairwise gain."""
    r2 = dict(zip(result_df["Feature_Group"], result_df["Test_R2"]))
    rmse = dict(zip(result_df["Feature_Group"], result_df["Test_RMSE"]))
    comparisons = [("MS + Thermal vs MS", "MS", "MS+Thermal"), ("MS + 3D vs MS", "MS", "MS+3D"), ("Thermal + 3D vs Thermal", "Thermal", "Thermal+3D"), ("All features vs MS", "MS", "All features (20)"), ("All features vs MS + Thermal", "MS+Thermal", "All features (20)"), ("All features vs MS + 3D", "MS+3D", "All features (20)"), ("All features vs Thermal + 3D", "Thermal+3D", "All features (20)")]
    records = [{"Comparison": comparison, "Base": base, "Combined": combined, "Base_R2": r2[base], "Combined_R2": r2[combined], "Delta_R2": r2[combined] - r2[base], "Base_RMSE": rmse[base], "Combined_RMSE": rmse[combined], "Delta_RMSE": rmse[combined] - rmse[base]} for comparison, base, combined in comparisons]
    df = pd.DataFrame(records)
    df.to_excel(OUTPUT_ROOT / "pairwise_information_gain.xlsx", index=False)
    return df

# ================================================================
# Feature Group Summary
# ================================================================

def create_group_summary(groups):
    """Perform create group summary."""
    group_order = ["LiDAR_Statistics", "Multispectral_Statistics", "Multispectral_Texture", "3D_Structure_Phenotype", "Temperature_Index", "Vertical_Structure_Phenotype", "Thermal_Infrared_Texture", "Thermal_Infrared_Statistics", "Vegetation_Index"]
    records = []
    for group_name in group_order:
        if group_name not in groups: continue
        features = groups[group_name]
        sources = list(dict.fromkeys(FEATURE_SOURCE[f] for f in features))
        records.append({"Feature_Group": group_name, "Display_Name": GROUP_DISPLAY_NAMES[group_name], "Information_Source": " + ".join(sources), "Feature_Number": len(features), "Features": ", ".join(features)})
    df = pd.DataFrame(records)
    df.to_excel(OUTPUT_ROOT / "feature_group_definition.xlsx", index=False)
    return df

# ================================================================
# Save Predictions
# ================================================================

def save_predictions(predictions, test_df):
    """Generate or save predictions."""
    output = predictions.copy()
    output.insert(0, ID_COL, test_df[ID_COL].values)
    output.insert(1, TARGET, test_df[TARGET].values)
    output.to_excel(OUTPUT_ROOT / "feature_group_test_predictions.xlsx", index=False)

# ================================================================
# Ranked Results
# ================================================================

def save_ranked_results(result_df):
    """Generate or save ranked results."""
    ranked = result_df.sort_values("Test_R2", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    ranked.to_excel(OUTPUT_ROOT / "feature_group_ablation_ranked.xlsx", index=False)
    return ranked

# ================================================================
# Figure 1
# ================================================================

def plot_r2_comparison(result_df):
    """Generate or save r2 comparison."""
    order = ["MS", "Thermal", "3D", "MS+Thermal", "MS+3D", "Thermal+3D", "All features (20)"]
    df = result_df.set_index("Feature_Group").loc[order].reset_index()
    values = df["Test_R2"].values
    labels = [DISPLAY_NAMES[x] for x in df["Feature_Group"]]
    colors = [GROUP_COLORS[x] for x in df["Feature_Group"]]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    bars = ax.bar(np.arange(len(df)), values, width=0.68, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"Independent test $R^2$")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values): ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(values[-1], linewidth=0.8, linestyle="--", alpha=0.45)
    clean_axes(ax)
    ax.set_title(f"Feature-group ablation using {ENSEMBLE_STRATEGY}", pad=10)
    fig.tight_layout()
    save_figure(fig, "01_test_r2_comparison")

# ================================================================
# Figure 2
# ================================================================

def plot_metric_panel(result_df):
    """Generate or save metric panel."""
    order = ["MS", "Thermal", "3D", "MS+Thermal", "MS+3D", "Thermal+3D", "All features (20)"]
    df = result_df.set_index("Feature_Group").loc[order].reset_index()
    labels = [DISPLAY_NAMES[x] for x in df["Feature_Group"]]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    metric_info = [("Test_R2", r"$R^2$", ".3f"), ("Test_RMSE", "RMSE", ".3f"), ("Test_rRMSE", "rRMSE (%)", ".2f"), ("Test_MAE", "MAE", ".3f")]
    for ax, (metric, ylabel, fmt) in zip(axes.ravel(), metric_info):
        values = df[metric].values
        bars = ax.bar(x, values, width=0.68, color=[GROUP_COLORS[name] for name in df["Feature_Group"]], edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if metric == "Test_R2":
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
        else:
            ax.set_ylim(0, np.max(values) * 1.18)
        ax.grid(axis="y", linewidth=0.5, alpha=0.22)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values): ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025 * np.max(values), format(value, fmt), ha="center", va="bottom", fontsize=7.5)
        clean_axes(ax)
    fig.suptitle(f"Performance comparison using {ENSEMBLE_STRATEGY}", fontsize=13, y=1.01)
    fig.tight_layout()
    save_figure(fig, "02_test_metrics_comparison")

# ================================================================
# Figure 3
# ================================================================

def plot_incremental_contribution(incremental_df):
    """Generate or save incremental contribution."""
    df = incremental_df.sort_values("Incremental_Delta_R2").reset_index(drop=True)
    values = df["Incremental_Delta_R2"].values
    labels = df["Comparison"].values
    colors = [COLOR_3D if "+3D" in label else COLOR_THERMAL for label in labels]
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    y = np.arange(len(df))
    bars = ax.barh(y, values, height=0.58, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Incremental $\Delta R^2$")
    span = max(np.max(np.abs(values)) * 1.35, 0.05)
    ax.set_xlim(min(-0.02, np.min(values) - 0.08 * span), np.max(values) + 0.18 * span)
    ax.grid(axis="x", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        if value >= 0: ax.text(value + 0.015 * span, bar.get_y() + bar.get_height() / 2, f"+{value:.3f}", va="center", ha="left", fontsize=8.5)
        else: ax.text(value - 0.015 * span, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="right", fontsize=8.5)
    clean_axes(ax)
    ax.set_title("Incremental contribution of additional information sources", pad=10)
    fig.tight_layout()
    save_figure(fig, "03_incremental_delta_r2")

# ================================================================
# Figure 4
# ================================================================

def plot_pairwise_gain(pairwise_df):
    """Generate or save pairwise gain."""
    df = pairwise_df.sort_values("Delta_R2").reset_index(drop=True)
    values = df["Delta_R2"].values
    labels = df["Comparison"].values
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    y = np.arange(len(df))
    colors = [COLOR_3D if "3D" in label else COLOR_THERMAL for label in labels]
    bars = ax.barh(y, values, height=0.56, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Change in independent-test $R^2$")
    ax.grid(axis="x", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    max_abs = max(np.max(np.abs(values)), 0.01)
    ax.set_xlim(min(-0.05, np.min(values) - 0.12 * max_abs), np.max(values) + 0.18 * max_abs)
    for bar, value in zip(bars, values):
        if value >= 0: ax.text(value + 0.018 * max_abs, bar.get_y() + bar.get_height() / 2, f"+{value:.3f}", va="center", ha="left", fontsize=8.5)
        else: ax.text(value - 0.018 * max_abs, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="right", fontsize=8.5)
    clean_axes(ax)
    ax.set_title("Pairwise information gain", pad=10)
    fig.tight_layout()
    save_figure(fig, "04_pairwise_information_gain")

# ================================================================
# Figure 5
# ================================================================

def plot_feature_number_vs_r2(result_df):
    """Generate or save feature number vs r2."""
    order = ["MS", "Thermal", "3D", "MS+Thermal", "MS+3D", "Thermal+3D", "All features (20)"]
    df = result_df.set_index("Feature_Group").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for _, row in df.iterrows():
        name = row["Feature_Group"]
        ax.scatter(row["Feature_Number"], row["Test_R2"], s=72, color=GROUP_COLORS[name], edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(DISPLAY_NAMES[name], (row["Feature_Number"], row["Test_R2"]), xytext=(5, 7), textcoords="offset points", fontsize=8.5)
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel(r"Independent test $R^2$")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.set_title("Relationship between feature-set size and Pn prediction", pad=10)
    fig.tight_layout()
    save_figure(fig, "05_feature_number_vs_r2")

# ================================================================
# Figure 6
# ================================================================

def plot_combination_matrix(result_df):
    """Generate or save combination matrix."""
    order = ["MS", "Thermal", "3D", "MS+Thermal", "MS+3D", "Thermal+3D", "All features (20)"]
    df = result_df.set_index("Feature_Group").loc[order]
    values = df["Test_R2"].values
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    image = ax.imshow(values[np.newaxis, :], aspect="auto", cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([DISPLAY_NAMES[x] for x in order], rotation=35, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["Test R²"])
    for i, value in enumerate(values): ax.text(i, 0, f"{value:.3f}", ha="center", va="center", fontsize=9, color="black" if value < 0.65 else "white")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(r"$R^2$", rotation=0, labelpad=8)
    ax.set_title("Independent-test performance across feature combinations", pad=12)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "06_feature_combination_r2_heatmap")

# ================================================================
# Figure 7
# ================================================================

def plot_feature_group_composition(groups):
    """Generate or save feature group composition."""
    order = ["LiDAR_Statistics", "Multispectral_Statistics", "Multispectral_Texture", "3D_Structure_Phenotype", "Temperature_Index", "Vertical_Structure_Phenotype", "Thermal_Infrared_Texture", "Thermal_Infrared_Statistics", "Vegetation_Index"]
    labels = [GROUP_DISPLAY_NAMES[x] for x in order if x in groups]
    counts = [len(groups[x]) for x in order if x in groups]
    colors = [COLOR_3D, COLOR_MS, COLOR_MS, COLOR_3D, COLOR_THERMAL, COLOR_3D, COLOR_THERMAL, COLOR_THERMAL, COLOR_NEUTRAL][:len(labels)]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, counts, height=0.58, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of selected features")
    ax.set_xlim(0, max(counts) + 1)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="x", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    for bar, count in zip(bars, counts): ax.text(count + 0.12, bar.get_y() + bar.get_height() / 2, str(count), va="center", ha="left", fontsize=9)
    clean_axes(ax)
    ax.set_title("Composition of the 20 selected features", pad=10)
    fig.tight_layout()
    save_figure(fig, "07_selected_feature_group_composition")

# ================================================================
# Figure 8
# ================================================================

def plot_r2_loss_vs_full(result_df):
    """Generate or save r2 loss vs full."""
    order = ["MS", "Thermal", "3D", "MS+Thermal", "MS+3D", "Thermal+3D"]
    full_r2 = result_df.loc[result_df["Feature_Group"] == "All features (20)", "Test_R2"].iloc[0]
    df = result_df.set_index("Feature_Group").loc[order].reset_index()
    df["R2_Loss"] = full_r2 - df["Test_R2"]
    df = df.sort_values("R2_Loss", ascending=True)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    y = np.arange(len(df))
    values = df["R2_Loss"].values
    bars = ax.barh(y, values, height=0.58, color=[GROUP_COLORS[x] for x in df["Feature_Group"]], edgecolor="white", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY_NAMES[x] for x in df["Feature_Group"]])
    ax.set_xlabel(r"$R^2$ loss relative to all selected features")
    ax.grid(axis="x", linewidth=0.5, alpha=0.2)
    ax.set_axisbelow(True)
    max_value = max(np.max(values), 0.01)
    ax.set_xlim(0, max_value * 1.18)
    for bar, value in zip(bars, values): ax.text(value + max_value * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=8.5)
    clean_axes(ax)
    ax.set_title("Performance loss caused by removing information sources", pad=10)
    fig.tight_layout()
    save_figure(fig, "08_r2_loss_vs_full")

# ================================================================
# Save Summary
# ================================================================

def save_summary(result_df, incremental_df, pairwise_df, ranked_df, groups, experiments, model_sources, model_records):
    """Generate or save summary."""
    full_r2 = result_df.loc[result_df["Feature_Group"] == "All features (20)", "Test_R2"].iloc[0]
    summary = {
        "experiment": {"name": "Strict Feature Group Ablation", "model": ENSEMBLE_STRATEGY, "evaluation": "Independent held-out test set", "random_state": RANDOM_STATE, "n_splits": N_SPLITS, "selected_feature_number": EXPECTED_FEATURE_NUMBER, "ridge_alpha": RIDGE_ALPHA},
        "ensemble": {"strategy": ENSEMBLE_STRATEGY, "method": "Ridge_Residual", "anchored_model": "ExtraTrees", "complementary_models": ["GBR", "XGB"], "residual_model": "Ridge", "ridge_alpha": RIDGE_ALPHA, "base_model_sources": model_sources},
        "feature_sources": {source: [f for f in FEATURE_SOURCE if FEATURE_SOURCE[f] == source] for source in ["Multispectral", "Thermal Infrared", "3D Point Cloud"]},
        "feature_varieties": groups,
        "experiments": experiments,
        "performance": {"full_r2": float(full_r2)},
        "incremental_contribution": incremental_df.to_dict(orient="records"),
        "pairwise_information_gain": pairwise_df.to_dict(orient="records"),
        "ranked_results": ranked_df.to_dict(orient="records"),
        "model_files": model_records.to_dict(orient="records")
    }
    with open(OUTPUT_ROOT / "feature_group_ablation_summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, indent=4, ensure_ascii=False)

# ================================================================
# Main
# ================================================================

def main():
    """Run the main analysis workflow."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Publication-quality Feature Group Ablation 2.0")
    print(f"Fixed Ensemble: {ENSEMBLE_STRATEGY}")
    print("=" * 80)
    X_train, y_train, X_test, y_test, feature_cols, train_df, test_df = load_data()
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}")
    print(f"Features      : {len(feature_cols)}")
    if len(feature_cols) != EXPECTED_FEATURE_NUMBER: raise ValueError(f"Expected {EXPECTED_FEATURE_NUMBER} selected features, got {len(feature_cols)}.")
    base_models, model_sources = load_best_ensemble()
    groups = define_feature_groups(feature_cols)
    experiments = define_experiments(feature_cols)
    validate_experiments(experiments, feature_cols)
    create_group_summary(groups)
    print("\n" + "=" * 80)
    print("Feature Group Definition")
    print("=" * 80)
    for name, features in groups.items():
        sources = list(dict.fromkeys(FEATURE_SOURCE[f] for f in features))
        print(f"{name:<32}{len(features):>3} features | {' + '.join(sources)}")
    print("\n" + "=" * 80)
    print("Experiment Definition")
    print("=" * 80)
    for name, features in experiments.items(): print(f"{name:<28}{len(features):>3} features")
    result_df, predictions, model_records = run_experiments(X_train, y_train, X_test, y_test, experiments, base_models, model_sources)
    result_df = calculate_deltas(result_df)
    incremental_df = calculate_incremental_contribution(result_df)
    pairwise_df = calculate_pairwise_gain(result_df)
    ranked_df = save_ranked_results(result_df)
    result_df.to_excel(OUTPUT_ROOT / "feature_group_ablation.xlsx", index=False)
    save_predictions(predictions, test_df)
    model_records.to_excel(OUTPUT_ROOT / "ablation_model_sizes.xlsx", index=False)
    print("\n" + "=" * 80)
    print("Generating Publication-quality Figures")
    print("=" * 80)
    plot_r2_comparison(result_df)
    plot_metric_panel(result_df)
    plot_incremental_contribution(incremental_df)
    plot_pairwise_gain(pairwise_df)
    plot_feature_number_vs_r2(result_df)
    plot_combination_matrix(result_df)
    plot_feature_group_composition(groups)
    plot_r2_loss_vs_full(result_df)
    save_summary(result_df, incremental_df, pairwise_df, ranked_df, groups, experiments, model_sources, model_records)
    print("\n" + "=" * 80)
    print("Final Feature Group Ablation Results")
    print("=" * 80)
    print(ranked_df[["Rank", "Feature_Group", "Feature_Number", "Test_R2", "Test_RMSE", "Test_rRMSE", "Test_MAE", "Delta_R2_vs_Full"]].to_string(index=False))
    print("\n" + "=" * 80)
    print("Incremental Contribution")
    print("=" * 80)
    print(incremental_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("Pairwise Information Gain")
    print("=" * 80)
    print(pairwise_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("Results saved to:")
    print(OUTPUT_ROOT)
    print("=" * 80)

if __name__ == "__main__":
    main()