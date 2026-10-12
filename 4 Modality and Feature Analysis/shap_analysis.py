"""Explain model predictions with SHAP feature-attribution analysis."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import shap
import joblib

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap


# ================================================================
# Configuration
# ================================================================

DATA_ROOT = Path(r"G:\PhyMaize")

FEATURE_FILE = DATA_ROOT / "features" / "feature_selection_results" / "20_features" / "selected_features.xlsx"
TEST_FILE = DATA_ROOT / "features" / "features-Pn (test).xlsx"
TEST_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "test.csv"
MODEL_FILE = DATA_ROOT / "traditional_ml_results" / "20_features" / "ExtraTrees" / "best_model.joblib"
OUTPUT_DIR = DATA_ROOT / "traditional_ml_results" / "20_features" / "SHAP"

ID_COL = "Sample_ID"
TARGET = "Pn"

TOP_N = 20
TOP_INTERACTION_N = 10
INTERACTION_MATRIX_N = 20
DECISION_SAMPLES = 100
RANDOM_STATE = 42

MATRIX_POINT_SIZE = 3.5
MATRIX_ALPHA = 0.35
MATRIX_JITTER = 0.22
MATRIX_X_PERCENTILE = 99.5


# ================================================================
# Figure switches
# ================================================================

PLOT_SHAP_BEESWARM = False
PLOT_INTERACTION_MATRIX = False
PLOT_TOP10_INTERACTIONS = True
PLOT_DECISION_PLOT = False


# ================================================================
# Plot settings
# ================================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ================================================================
# Utility
# ================================================================

def save_figure(fig, name):
    """Generate or save figure."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def find_target_column(df):
    """Retrieve or load target column for subsequent analysis."""
    candidates = [TARGET, "Ground_truth Pn", "Ground_truth_Pn", "Pn_ground_truth"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find Pn target column. Available columns: {df.columns.tolist()}")


# ================================================================
# Load selected feature names
# ================================================================

def load_selected_features():
    """Retrieve or load selected features for subsequent analysis."""
    df = pd.read_excel(FEATURE_FILE)
    target_col = find_target_column(df)
    feature_names = [col for col in df.columns if col not in [ID_COL, target_col]]
    if len(feature_names) != 20:
        raise ValueError(f"Expected 20 selected features, but found {len(feature_names)}:\n{feature_names}")
    print("\nSelected features:")
    for i, feature in enumerate(feature_names, start=1):
        print(f"{i:02d}. {feature}")
    return feature_names


# ================================================================
# Load independent test set
# ================================================================

def load_test_data(feature_names):
    """Retrieve or load test data for subsequent analysis."""
    test_df = pd.read_excel(TEST_FILE)
    if ID_COL not in test_df.columns:
        raise ValueError(f"{ID_COL} not found in test file.")
    target_col = find_target_column(test_df)
    missing_features = [feature for feature in feature_names if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"The following selected features are missing from test file:\n{missing_features}")
    if test_df[ID_COL].duplicated().any():
        duplicate_ids = test_df.loc[test_df[ID_COL].duplicated(), ID_COL].astype(str).tolist()
        raise ValueError(f"Duplicate Sample_ID detected in test file: {duplicate_ids[:10]}")
    split_df = pd.read_csv(TEST_SPLIT_FILE)
    if ID_COL not in split_df.columns:
        raise ValueError(f"{ID_COL} not found in test split.")
    test_ids = split_df[ID_COL].astype(str).tolist()
    if len(test_ids) == 0:
        raise ValueError("Official test split contains no Sample_ID.")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate Sample_ID detected in official test split.")
    test_df[ID_COL] = test_df[ID_COL].astype(str)
    test_id_set = set(test_ids)
    available_id_set = set(test_df[ID_COL])
    missing_ids = test_id_set - available_id_set
    if missing_ids:
        raise ValueError(f"Some official test Sample_IDs are missing from test feature file:\n{list(missing_ids)[:20]}")
    test_df = test_df[test_df[ID_COL].isin(test_id_set)].copy()
    test_order = {sample_id: i for i, sample_id in enumerate(test_ids)}
    test_df["_split_order"] = test_df[ID_COL].map(test_order)
    test_df = test_df.sort_values("_split_order").drop(columns="_split_order").reset_index(drop=True)
    X_test = test_df[feature_names].copy()
    y_test = test_df[target_col].copy()
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(y_test, errors="coerce")
    valid = X_test.notna().all(axis=1) & y_test.notna()
    if not valid.all():
        removed = (~valid).sum()
        print(f"\nWarning: {removed} test samples contain missing/non-numeric values and will be removed.")
    X_test = X_test.loc[valid].reset_index(drop=True)
    y_test = y_test.loc[valid].reset_index(drop=True)
    test_ids_final = test_df.loc[valid, ID_COL].reset_index(drop=True)
    if len(X_test) == 0:
        raise ValueError("No valid test samples remain after data cleaning.")
    print(f"\nOfficial test split  : {len(test_ids)} samples")
    print(f"Matched test samples : {len(X_test)} samples")
    print(f"Feature number       : {len(feature_names)}")
    return X_test, y_test, test_ids_final


# ================================================================
# Extract ExtraTrees from model
# ================================================================

def extract_estimator(model):
    """Perform extract estimator."""
    if hasattr(model, "named_steps"):
        print("\nPipeline steps:")
        for name, step in model.named_steps.items():
            print(f"  {name}: {step.__class__.__name__}")
            if hasattr(step, "feature_importances_"):
                return step
        raise ValueError("Cannot find tree-based estimator in pipeline.")
    if hasattr(model, "feature_importances_"):
        return model
    raise ValueError("Loaded model does not contain a tree-based estimator.")


# ================================================================
# Metrics
# ================================================================

def calculate_metrics(y_true, y_pred):
    """Compute metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rrmse = rmse / np.mean(np.abs(y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, rrmse, mae


# ================================================================
# SHAP interaction analysis
# ================================================================

def calculate_interactions(explainer, X_test, feature_names):
    """Compute interactions."""
    print("\nCalculating SHAP interaction values...")
    interaction_values = explainer.shap_interaction_values(X_test)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]
    interaction_values = np.asarray(interaction_values)
    if interaction_values.ndim != 3:
        raise ValueError(f"Unexpected interaction shape: {interaction_values.shape}")
    expected_shape = (len(X_test), len(feature_names), len(feature_names))
    if interaction_values.shape != expected_shape:
        raise ValueError(f"Interaction shape {interaction_values.shape} does not match expected {expected_shape}")
    if not np.isfinite(interaction_values).all():
        raise ValueError("SHAP interaction matrix contains NaN or infinite values.")
    print(f"Interaction matrix: {interaction_values.shape}")
    mean_interaction = interaction_values.mean(axis=0)
    mean_abs_interaction = np.abs(interaction_values).mean(axis=0)
    return interaction_values, mean_interaction, mean_abs_interaction


# ================================================================
# Build interaction pair table
# ================================================================

def build_interaction_table(interaction_values, feature_names):
    """Perform build interaction table."""
    records = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            values = interaction_values[:, i, j]
            records.append({
                "Feature_1": feature_names[i],
                "Feature_2": feature_names[j],
                "Mean_Interaction": np.mean(values),
                "Mean_Abs_Interaction": np.mean(np.abs(values)),
                "Median_Interaction": np.median(values),
                "Std_Interaction": np.std(values)
            })
    df = pd.DataFrame(records).sort_values("Mean_Abs_Interaction", ascending=False).reset_index(drop=True)
    df["Interaction_Rank"] = np.arange(1, len(df) + 1)
    return df


# ================================================================
# SHAP beeswarm
# ================================================================

def plot_shap_beeswarm(shap_values, X_test, feature_names):
    """Generate or save shap beeswarm."""
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=TOP_N,
        show=False,
        plot_size=None
    )
    ax = plt.gca()
    ax.set_xlabel("SHAP value", fontsize=20)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "Fig_SHAP_Beeswarm")


# ================================================================
# SHAP interaction summary matrix
# ================================================================

def plot_interaction_summary_matrix(interaction_values, X_test, feature_names, importance_df):
    """Generate or save interaction summary matrix."""
    n_features = min(INTERACTION_MATRIX_N, len(feature_names))
    selected_features = importance_df.head(n_features)["Feature"].tolist()
    indices = [feature_names.index(f) for f in selected_features]
    interaction_sub = interaction_values[:, indices][:, :, indices]
    X_sub = X_test[selected_features].copy()
    finite_values = interaction_sub[np.isfinite(interaction_sub)]
    x_lim = np.percentile(np.abs(finite_values), MATRIX_X_PERCENTILE)
    x_lim = max(float(x_lim), 1e-6)
    print(f"\nGenerating {n_features} × {n_features} SHAP interaction summary matrix...")
    print(f"Interaction x-axis limit: ±{x_lim:.4f}")
    fig, axes = plt.subplots(n_features, n_features, figsize=(24, 24), squeeze=False)
    rng = np.random.default_rng(RANDOM_STATE)
    cmap = plt.cm.coolwarm
    feature_norms = {}
    for feature in selected_features:
        values = X_sub[feature].to_numpy(dtype=float)
        vmin, vmax = np.nanpercentile(values, [2, 98])
        if np.isclose(vmin, vmax):
            vmin, vmax = np.nanmin(values), np.nanmax(values)
        feature_norms[feature] = Normalize(vmin=vmin, vmax=vmax)
    for row in range(n_features):
        for col in range(n_features):
            ax = axes[row, col]
            values = interaction_sub[:, row, col]
            color_values = X_sub.iloc[:, col].to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(color_values)
            values = values[valid]
            color_values = color_values[valid]
            if len(values):
                jitter = rng.uniform(-MATRIX_JITTER, MATRIX_JITTER, len(values))
                norm = feature_norms[selected_features[col]]
                ax.scatter(values, jitter, s=MATRIX_POINT_SIZE, alpha=MATRIX_ALPHA, c=cmap(norm(color_values)), linewidths=0)
            ax.axvline(0, linewidth=0.6)
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            if row < n_features - 1:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            else:
                ax.tick_params(axis="x", labelsize=9, length=3, pad=2)
            if row == 0:
                ax.set_title(selected_features[col], fontsize=11, pad=7)
            if col == 0:
                ax.set_ylabel(selected_features[row], fontsize=11, rotation=0, labelpad=58, va="center")
    fig.supxlabel("SHAP interaction value", fontsize=20, y=0.035)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.012, aspect=40)
    cbar.set_label("Feature value", fontsize=18)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=13)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.08, top=0.96, wspace=0.05, hspace=0.05)
    save_figure(fig, "Fig_SHAP_Interaction_Summary_Matrix")


# ================================================================
# TOP10 SHAP interactions
# ================================================================

def plot_top10_interaction_summary(interaction_values, X_test, interaction_df):
    """Generate or save top10 interaction summary."""
    top_df = interaction_df.head(TOP_INTERACTION_N).copy()
    rng = np.random.default_rng(RANDOM_STATE)
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#008BFB", "#FF0051"])
    pairs = list(zip(top_df["Feature_1"], top_df["Feature_2"]))
    labels = [f"{f1} × {f2}" for f1, f2 in pairs]
    y_pos = np.arange(len(pairs))[::-1]
    fig, ax = plt.subplots(figsize=(11, 8))
    for rank, ((f1, f2), y) in enumerate(zip(pairs, y_pos), start=1):
        i = X_test.columns.get_loc(f1)
        j = X_test.columns.get_loc(f2)
        interaction = interaction_values[:, i, j].astype(float)
        feature_value = X_test[f2].to_numpy(dtype=float)
        valid = np.isfinite(interaction) & np.isfinite(feature_value)
        interaction = interaction[valid]
        feature_value = feature_value[valid]
        if len(interaction) == 0:
            continue
        vmin, vmax = np.nanpercentile(feature_value, [2, 98])
        if np.isclose(vmin, vmax):
            vmin, vmax = np.nanmin(feature_value), np.nanmax(feature_value)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(feature_value))
        jitter = rng.uniform(-0.18, 0.18, len(interaction))
        ax.scatter(interaction, y + jitter, c=colors, s=28, alpha=0.55, edgecolors="none")
    ax.axvline(0, linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlabel("SHAP interaction value", fontsize=20)
    ax.set_ylabel("Feature interaction", fontsize=20)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_values = interaction_values[:, :, :]
    finite_values = x_values[np.isfinite(x_values)]
    x_lim = np.percentile(np.abs(finite_values), MATRIX_X_PERCENTILE)
    x_lim = max(float(x_lim), 1e-6)
    ax.set_xlim(-x_lim, x_lim)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=35)
    cbar.set_label("Feature value", fontsize=18)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=13)
    fig.tight_layout()
    save_figure(fig, "Fig_SHAP_Top10_Interactions")

# ================================================================
# Decision plot
# ================================================================

def plot_decision_plot(explainer, shap_values, X_test):
    """Generate or save decision plot."""
    print("\nGenerating SHAP decision plot...")
    decision_sample_num = min(DECISION_SAMPLES, len(X_test))
    expected_value = np.asarray(explainer.expected_value).reshape(-1)[0]
    fig = plt.figure(figsize=(12, 8))
    shap.decision_plot(
        expected_value,
        shap_values[:decision_sample_num],
        X_test.iloc[:decision_sample_num],
        show=False
    )
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("Model output", fontsize=20)
    ax.set_ylabel("Features", fontsize=20)
    fig.tight_layout()
    save_figure(fig, "Fig_SHAP_Decision_Plot")
    return expected_value


# ================================================================
# Main
# ================================================================

def main():
    """Run the main analysis workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("SHAP Analysis - ExtraTrees 20 Features")
    print("=" * 70)
    print("\nFigure configuration:")
    print(f"SHAP beeswarm          : {PLOT_SHAP_BEESWARM}")
    print(f"Interaction matrix     : {PLOT_INTERACTION_MATRIX}")
    print(f"TOP10 interactions     : {PLOT_TOP10_INTERACTIONS}")
    print(f"Decision plot          : {PLOT_DECISION_PLOT}")
    feature_names = load_selected_features()
    X_test, y_test, test_ids = load_test_data(feature_names)
    print("\nFinal analysis dataset:")
    print(f"Samples  : {len(X_test)}")
    print(f"Features : {X_test.shape[1]}")
    model = joblib.load(MODEL_FILE)
    print(f"\nModel loaded:\n{MODEL_FILE}")
    estimator = extract_estimator(model)
    print("\nEstimator:")
    print(estimator.__class__.__name__)
    y_pred = model.predict(X_test)
    r2, rmse, rrmse, mae = calculate_metrics(y_test, y_pred)
    print("\nModel performance:")
    print(f"R²    : {r2:.6f}")
    print(f"RMSE  : {rmse:.6f}")
    print(f"rRMSE : {rrmse:.6f}%")
    print(f"MAE   : {mae:.6f}")
    # ------------------------------------------------------------
    # SHAP values
    # ------------------------------------------------------------
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")
    if shap_values.shape != X_test.shape:
        raise ValueError(f"SHAP shape {shap_values.shape} does not match X_test shape {X_test.shape}")
    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP matrix contains NaN or infinite values.")
    print(f"SHAP matrix : {shap_values.shape}")
    print(f"SHAP range  : [{shap_values.min():.6f}, {shap_values.max():.6f}]")
    # ------------------------------------------------------------
    # SHAP importance
    # ------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    median_abs_shap = np.median(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Mean_SHAP": mean_shap,
        "Median_Abs_SHAP": median_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    importance_df["SHAP_Rank"] = np.arange(1, len(importance_df) + 1)
    importance_df.to_excel(OUTPUT_DIR / "SHAP_feature_importance.xlsx", index=False)
    # ------------------------------------------------------------
    # Raw SHAP values
    # ------------------------------------------------------------
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.insert(0, ID_COL, test_ids)
    shap_df.to_excel(OUTPUT_DIR / "SHAP_values.xlsx", index=False)
    summary_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Mean_SHAP": mean_shap,
        "Median_Abs_SHAP": median_abs_shap
    })
    summary_df.to_excel(OUTPUT_DIR / "SHAP_summary_data.xlsx", index=False)
    # ------------------------------------------------------------
    # SHAP beeswarm
    # ------------------------------------------------------------
    if PLOT_SHAP_BEESWARM:
        plot_shap_beeswarm(shap_values, X_test, feature_names)
    # ------------------------------------------------------------
    # SHAP interaction analysis
    # ------------------------------------------------------------
    interaction_needed = PLOT_INTERACTION_MATRIX or PLOT_TOP10_INTERACTIONS
    if interaction_needed:
        interaction_values, mean_interaction, mean_abs_interaction = calculate_interactions(
            explainer,
            X_test,
            feature_names
        )
        interaction_df = build_interaction_table(interaction_values, feature_names)
        interaction_df.to_excel(OUTPUT_DIR / "SHAP_interaction_pairs.xlsx", index=False)
        pd.DataFrame(
            mean_interaction,
            index=feature_names,
            columns=feature_names
        ).to_excel(OUTPUT_DIR / "SHAP_interaction_mean_matrix.xlsx")
        pd.DataFrame(
            mean_abs_interaction,
            index=feature_names,
            columns=feature_names
        ).to_excel(OUTPUT_DIR / "SHAP_interaction_mean_abs_matrix.xlsx")
        if PLOT_INTERACTION_MATRIX:
            plot_interaction_summary_matrix(
                interaction_values,
                X_test,
                feature_names,
                importance_df
            )
        if PLOT_TOP10_INTERACTIONS:
            plot_top10_interaction_summary(interaction_values, X_test, interaction_df)
    # ------------------------------------------------------------
    # Decision plot
    # ------------------------------------------------------------
    expected_value = np.asarray(explainer.expected_value).reshape(-1)[0]
    if PLOT_DECISION_PLOT:
        expected_value = plot_decision_plot(explainer, shap_values, X_test)
    # ------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------
    model_summary = pd.DataFrame({
        "Metric": [
            "Test samples",
            "Feature number",
            "Test R2",
            "Test RMSE",
            "Test rRMSE",
            "Test MAE",
            "Expected value"
        ],
        "Value": [
            len(X_test),
            len(feature_names),
            r2,
            rmse,
            rrmse,
            mae,
            expected_value
        ]
    })
    model_summary.to_excel(OUTPUT_DIR / "SHAP_model_summary.xlsx", index=False)
    # ------------------------------------------------------------
    # Save prediction results
    # ------------------------------------------------------------
    prediction_df = pd.DataFrame({
        ID_COL: test_ids,
        "Observed_Pn": y_test.values,
        "Predicted_Pn": y_pred,
        "Residual": y_test.values - y_pred
    })
    prediction_df.to_excel(OUTPUT_DIR / "SHAP_test_predictions.xlsx", index=False)
    # ------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SHAP analysis completed.")
    print("=" * 70)
    print(f"\nOutput directory:\n{OUTPUT_DIR}")
    print("\nTop 10 SHAP features:")
    print(importance_df[["SHAP_Rank", "Feature", "Mean_Abs_SHAP", "Mean_SHAP"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()