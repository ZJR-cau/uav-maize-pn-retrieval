"""Analyze feature importance using leave-one-feature-out evaluation."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ================================================================
# Configuration
# ================================================================

DATA_ROOT = Path(r"G:\PhyMaize")

FEATURE_FILE = DATA_ROOT / "features" / "feature_selection_results" / "20_features" / "selected_features.xlsx"
ALL_FEATURE_FILE = DATA_ROOT / "features" / "features.csv"
LABEL_FILE = DATA_ROOT / "annotations" / "lables.csv"
TRAIN_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "train.csv"
VALIDATION_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "val.csv"
TEST_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "test.csv"
MODEL_FILE = DATA_ROOT / "traditional_ml_results" / "20_features" / "ExtraTrees" / "best_model.joblib"
OUTPUT_DIR = DATA_ROOT / "traditional_ml_results" / "20_features" / "LOFO"

ID_COL = "Sample_ID"
TARGET = "Pn"
RANDOM_STATE = 42


# ================================================================
# Plot settings
# ================================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
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
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in selected feature file.")
    target_col = None
    for col in [TARGET, "Ground_truth Pn", "Ground_truth_Pn", "Pn_ground_truth"]:
        if col in df.columns:
            target_col = col
            break
    feature_names = [col for col in df.columns if col not in [ID_COL, target_col]]
    if len(feature_names) != 20:
        raise ValueError(f"Expected 20 selected features, but found {len(feature_names)}:\n{feature_names}")
    print("\nSelected features:")
    for i, feature in enumerate(feature_names, start=1):
        print(f"{i:02d}. {feature}")
    return feature_names


# ================================================================
# Load split IDs
# ================================================================

def load_split_ids(split_file, split_name):
    """Retrieve or load split ids for subsequent analysis."""
    df = pd.read_csv(split_file)
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in {split_name} split file.")
    ids = df[ID_COL].astype(str).str.strip().tolist()
    if len(ids) == 0:
        raise ValueError(f"{split_name} split contains no Sample_ID.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate Sample_ID detected in {split_name} split.")
    return ids


# ================================================================
# Load all feature data
# ================================================================

def load_all_features():
    """Retrieve or load all features for subsequent analysis."""
    df = pd.read_csv(ALL_FEATURE_FILE)
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in feature file.")
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    if df[ID_COL].duplicated().any():
        duplicate_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate Sample_ID detected in feature file: {duplicate_ids[:10]}")
    return df


# ================================================================
# Load labels
# ================================================================

def load_labels():
    """Retrieve or load labels for subsequent analysis."""
    df = pd.read_csv(LABEL_FILE)
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in label file.")
    target_col = find_target_column(df)
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    if df[ID_COL].duplicated().any():
        duplicate_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate Sample_ID detected in label file: {duplicate_ids[:10]}")
    return df[[ID_COL, target_col]].copy(), target_col


# ================================================================
# Prepare dataset
# ================================================================

def prepare_dataset(feature_df, label_df, target_col, sample_ids, feature_names, dataset_name):
    """Perform prepare dataset."""
    feature_id_set = set(feature_df[ID_COL])
    label_id_set = set(label_df[ID_COL])
    sample_id_set = set(sample_ids)
    missing_features = sample_id_set - feature_id_set
    if missing_features:
        raise ValueError(f"{dataset_name}: {len(missing_features)} Sample_IDs are missing from feature file:\n{list(missing_features)[:20]}")
    missing_labels = sample_id_set - label_id_set
    if missing_labels:
        raise ValueError(f"{dataset_name}: {len(missing_labels)} Sample_IDs are missing from label file:\n{list(missing_labels)[:20]}")
    missing_features_columns = [feature for feature in feature_names if feature not in feature_df.columns]
    if missing_features_columns:
        raise ValueError(f"The following selected features are missing from feature file:\n{missing_features_columns}")
    feature_indexed = feature_df.set_index(ID_COL)
    label_indexed = label_df.set_index(ID_COL)
    X = feature_indexed.loc[sample_ids, feature_names].copy()
    y = label_indexed.loc[sample_ids, target_col].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    removed = (~valid).sum()
    if removed > 0:
        print(f"\nWarning: {removed} {dataset_name} samples contain missing/non-numeric values and will be removed.")
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    final_ids = pd.Series(sample_ids).loc[valid.values].reset_index(drop=True)
    if len(X) == 0:
        raise ValueError(f"No valid {dataset_name} samples remain after data cleaning.")
    return X, y, final_ids


# ================================================================
# Load development data
# ================================================================

def load_development_data(feature_names):
    """Retrieve or load development data for subsequent analysis."""
    train_ids = load_split_ids(TRAIN_SPLIT_FILE, "training")
    validation_ids = load_split_ids(VALIDATION_SPLIT_FILE, "validation")
    overlap = set(train_ids).intersection(validation_ids)
    if overlap:
        raise ValueError(f"Training and validation splits contain overlapping Sample_IDs:\n{list(overlap)[:20]}")
    development_ids = train_ids + validation_ids
    feature_df = load_all_features()
    label_df, target_col = load_labels()
    X_dev, y_dev, dev_ids = prepare_dataset(feature_df, label_df, target_col, development_ids, feature_names, "development")
    print("\nDevelopment dataset:")
    print(f"Training split   : {len(train_ids)}")
    print(f"Validation split : {len(validation_ids)}")
    print(f"Development      : {len(X_dev)}")
    return X_dev, y_dev, dev_ids


# ================================================================
# Load independent test data
# ================================================================

def load_test_data(feature_names):
    """Retrieve or load test data for subsequent analysis."""
    test_ids = load_split_ids(TEST_SPLIT_FILE, "test")
    feature_df = load_all_features()
    label_df, target_col = load_labels()
    X_test, y_test, test_ids_final = prepare_dataset(feature_df, label_df, target_col, test_ids, feature_names, "test")
    print("\nIndependent test dataset:")
    print(f"Official test split : {len(test_ids)}")
    print(f"Valid test samples  : {len(X_test)}")
    return X_test, y_test, test_ids_final


# ================================================================
# Extract ExtraTrees estimator
# ================================================================

def extract_estimator(model):
    """Perform extract estimator."""
    if hasattr(model, "named_steps"):
        print("\nPipeline steps:")
        for name, step in model.named_steps.items():
            print(f"  {name}: {step.__class__.__name__}")
            if isinstance(step, ExtraTreesRegressor):
                return step
            if hasattr(step, "feature_importances_"):
                return step
        raise ValueError("Cannot find ExtraTrees estimator in pipeline.")
    if isinstance(model, ExtraTreesRegressor):
        return model
    if hasattr(model, "feature_importances_"):
        return model
    raise ValueError("Loaded model does not contain a tree-based estimator.")


# ================================================================
# Get ExtraTrees parameters
# ================================================================

def get_model_parameters():
    """Retrieve or load model parameters for subsequent analysis."""
    model = joblib.load(MODEL_FILE)
    estimator = extract_estimator(model)
    params = estimator.get_params()
    print("\nExtraTrees parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    return params


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
# Train and evaluate
# ================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """Perform train and evaluate."""
    model = ExtraTreesRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, pred)
    return model, pred, metrics


# ================================================================
# Main
# ================================================================

def main():
    """Run the main analysis workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("LOFO Analysis - ExtraTrees 21 Features")
    print("=" * 70)
    # ------------------------------------------------------------
    # Load selected feature names
    # ------------------------------------------------------------
    feature_names = load_selected_features()
    # ------------------------------------------------------------
    # Load development data
    # ------------------------------------------------------------
    X_dev, y_dev, dev_ids = load_development_data(feature_names)
    # ------------------------------------------------------------
    # Load independent test data
    # ------------------------------------------------------------
    X_test, y_test, test_ids = load_test_data(feature_names)
    print("\nFinal analysis datasets:")
    print(f"Development samples : {len(X_dev)}")
    print(f"Test samples        : {len(X_test)}")
    print(f"Features            : {len(feature_names)}")
    # ------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------
    params = get_model_parameters()
    # ------------------------------------------------------------
    # Full model
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training full 21-feature model...")
    full_model, full_pred, full_metrics = train_and_evaluate(X_dev, y_dev, X_test, y_test, params)
    full_r2, full_rmse, full_rrmse, full_mae = full_metrics
    print("\nFull model:")
    print(f"R²    : {full_r2:.6f}")
    print(f"RMSE  : {full_rmse:.6f}")
    print(f"rRMSE : {full_rrmse:.6f}%")
    print(f"MAE   : {full_mae:.6f}")
    # ------------------------------------------------------------
    # Save full model predictions
    # ------------------------------------------------------------
    full_prediction_df = pd.DataFrame({
        ID_COL: test_ids,
        "Observed_Pn": y_test.values,
        "Predicted_Pn": full_pred,
        "Residual": y_test.values - full_pred
    })
    full_prediction_df.to_excel(OUTPUT_DIR / "LOFO_full_model_predictions.xlsx", index=False)
    # ------------------------------------------------------------
    # LOFO
    # ------------------------------------------------------------
    results = []
    for i, feature in enumerate(feature_names, start=1):
        print("\n" + "-" * 70)
        print(f"[{i:02d}/{len(feature_names)}] Removing: {feature}")
        remaining_features = [f for f in feature_names if f != feature]
        model, pred, metrics = train_and_evaluate(X_dev[remaining_features], y_dev, X_test[remaining_features], y_test, params)
        r2, rmse, rrmse, mae = metrics
        delta_r2 = full_r2 - r2
        delta_rmse = rmse - full_rmse
        delta_rrmse = rrmse - full_rrmse
        delta_mae = mae - full_mae
        print(f"LOFO R²    : {r2:.6f}")
        print(f"LOFO RMSE  : {rmse:.6f}")
        print(f"LOFO rRMSE : {rrmse:.6f}%")
        print(f"LOFO MAE   : {mae:.6f}")
        print(f"ΔR²        : {delta_r2:.6f}")
        print(f"ΔRMSE      : {delta_rmse:.6f}")
        print(f"ΔrRMSE     : {delta_rrmse:.6f}%")
        print(f"ΔMAE       : {delta_mae:.6f}")
        results.append({
            "Feature": feature,
            "Feature_Number": len(remaining_features),
            "Full_R2": full_r2,
            "LOFO_R2": r2,
            "Delta_R2": delta_r2,
            "Full_RMSE": full_rmse,
            "LOFO_RMSE": rmse,
            "Delta_RMSE": delta_rmse,
            "Full_rRMSE": full_rrmse,
            "LOFO_rRMSE": rrmse,
            "Delta_rRMSE": delta_rrmse,
            "Full_MAE": full_mae,
            "LOFO_MAE": mae,
            "Delta_MAE": delta_mae
        })
    # ------------------------------------------------------------
    # Results dataframe
    # ------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Delta_R2", ascending=False).reset_index(drop=True)
    results_df["LOFO_Rank"] = np.arange(1, len(results_df) + 1)
    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    results_df.to_excel(OUTPUT_DIR / "LOFO_feature_importance.xlsx", index=False)
    # ------------------------------------------------------------
    # Rank plot: Delta R2
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.barh(results_df["Feature"][::-1], results_df["Delta_R2"][::-1], edgecolor="black", linewidth=0.6)
    ax.axvline(0, linewidth=1.0)
    ax.set_xlabel("Decrease in test R² after feature removal")
    ax.set_ylabel("Feature")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "Fig_LOFO_Delta_R2")
    # ------------------------------------------------------------
    # RMSE increase
    # ------------------------------------------------------------
    rmse_df = results_df.sort_values("Delta_RMSE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.barh(rmse_df["Feature"][::-1], rmse_df["Delta_RMSE"][::-1], edgecolor="black", linewidth=0.6)
    ax.axvline(0, linewidth=1.0)
    ax.set_xlabel("Increase in test RMSE after feature removal")
    ax.set_ylabel("Feature")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "Fig_LOFO_Delta_RMSE")
    # ------------------------------------------------------------
    # R2 comparison
    # ------------------------------------------------------------
    compare_df = results_df.sort_values("LOFO_R2", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.barh(compare_df["Feature"], compare_df["LOFO_R2"], edgecolor="black", linewidth=0.6)
    ax.axvline(full_r2, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Test R² after feature removal")
    ax.set_ylabel("Removed feature")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "Fig_LOFO_R2_After_Removal")
    # ------------------------------------------------------------
    # MAE increase
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # MAE increase
    # ------------------------------------------------------------
    mae_df = results_df.sort_values("Delta_MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.barh(mae_df["Feature"][::-1], mae_df["Delta_MAE"][::-1], edgecolor="black", linewidth=0.6)
    ax.axvline(0, linewidth=1.0)
    ax.set_xlabel("Increase in test MAE after feature removal")
    ax.set_ylabel("Feature")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "Fig_LOFO_Delta_MAE")
    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    summary = pd.DataFrame({
        "Metric": [
            "Training samples",
            "Validation samples",
            "Development samples",
            "Test samples",
            "Feature number",
            "Full R2",
            "Full RMSE",
            "Full rRMSE",
            "Full MAE"
        ],
        "Value": [
            len(load_split_ids(TRAIN_SPLIT_FILE, "training")),
            len(load_split_ids(VALIDATION_SPLIT_FILE, "validation")),
            len(X_dev),
            len(X_test),
            len(feature_names),
            full_r2,
            full_rmse,
            full_rrmse,
            full_mae
        ]
    })
    summary.to_excel(OUTPUT_DIR / "LOFO_model_summary.xlsx", index=False)
    # ------------------------------------------------------------
    # Top 10
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOFO analysis completed.")
    print("=" * 70)
    print(f"\nOutput directory:\n{OUTPUT_DIR}")
    print("\nTop 10 LOFO features:")
    print(
        results_df[
            ["LOFO_Rank", "Feature", "LOFO_R2", "Delta_R2", "Delta_RMSE", "Delta_MAE"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()