"""Analyze model performance across different retained feature counts."""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor


path = r"G:\PhyMaize\features\features-Pn (train and val).xlsx"
sheet = "Sheet1"
OUTPUT_ROOT = Path(r"G:\PhyMaize\features\feature_selection_results")

MIN_FEATURE_NUM = 1
MAX_FEATURE_NUM = 25
CORRELATION_THRESHOLD = 0.95
VIF_THRESHOLD = 10
RUN_SHAP = False

target = "Pn"
id_col = "Sample_ID"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def print_stage(title, function, input_num, output_num, removed_num=None):
    """Perform print stage."""
    print("\n========================")
    print(title)
    print("========================")
    print("Function:", function)
    print("Input features:", input_num)
    if removed_num is not None:
        print("Removed features:", removed_num)
    print("Remaining features:", output_num)


def load_data(path, sheet, target, id_col):
    """Retrieve or load data for subsequent analysis."""
    df = pd.read_excel(path, sheet_name=sheet)
    if df.columns.duplicated().any():
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names detected: {duplicate_cols}")
    if df[id_col].duplicated().any():
        duplicate_ids = df.loc[df[id_col].duplicated(), id_col].tolist()
        raise ValueError(f"Duplicate Sample_ID detected: {duplicate_ids[:10]}")
    feature_cols = [c for c in df.columns if c not in [target, id_col]]
    mask = ~df[feature_cols].isna().any(axis=1) & ~df[target].isna()
    df = df.loc[mask].reset_index(drop=True)
    sample_ids = df[id_col].copy()
    X_raw = df[feature_cols].copy()
    y = df[target].copy()
    return sample_ids, X_raw, y, feature_cols


def correlation_filter(X, threshold):
    """Perform correlation filter."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    corr = X_scaled.corr().abs()
    plt.figure(figsize=(18, 15))
    ax = sns.heatmap(corr, cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "corr_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    X_after_corr = X_scaled.drop(columns=drop_cols).copy()
    return X_after_corr, drop_cols


def calc_vif(data):
    """Compute vif."""
    return pd.DataFrame({"Feature": data.columns, "VIF": [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]})


def vif_filter(X, threshold):
    """Perform vif filter."""
    X_vif = X.copy()
    removed_vif_features = []
    while X_vif.shape[1] > 1:
        vif_df = calc_vif(X_vif)
        max_vif = vif_df["VIF"].max()
        if max_vif < threshold:
            break
        remove_feature = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
        print("Remove VIF:", remove_feature, "| VIF:", max_vif)
        removed_vif_features.append(remove_feature)
        X_vif = X_vif.drop(columns=[remove_feature])
    return X_vif.copy(), removed_vif_features


def boruta_filter(X, y, target_num):
    """Perform boruta filter."""
    rf_boruta = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    boruta = BorutaPy(estimator=rf_boruta, n_estimators="auto", verbose=2, random_state=42)
    boruta.fit(X.values, y.values)
    confirmed_features = X.columns[boruta.support_].tolist()
    tentative_features = X.columns[boruta.support_weak_].tolist()
    rejected_features = X.columns[~(boruta.support_ | boruta.support_weak_)].tolist()
    selected_mask = boruta.support_ | boruta.support_weak_
    selected_features = X.columns[selected_mask].tolist()
    additional_features = []
    if len(selected_features) < target_num:
        boruta_ranking = pd.DataFrame({
            "Feature": X.columns,
            "Boruta_Rank": boruta.ranking_,
            "Confirmed": boruta.support_,
            "Tentative": boruta.support_weak_
        }).sort_values("Boruta_Rank", ascending=True).reset_index(drop=True)
        for feature in boruta_ranking["Feature"]:
            if feature not in selected_features:
                additional_features.append(feature)
            if len(selected_features) + len(additional_features) >= target_num:
                break
        selected_features += additional_features
    X_boruta = X[selected_features].copy()
    return X_boruta, boruta, confirmed_features, tentative_features, rejected_features, additional_features


def rfe_selection(X, y):
    """Perform rfe selection."""
    svr = SVR(kernel="linear")
    rfe = RFE(estimator=svr, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    rank_df = pd.DataFrame({"Feature": X.columns, "RFE_Rank": rfe.ranking_}).sort_values(["RFE_Rank", "Feature"]).reset_index(drop=True)
    return rank_df, rfe

def save_selection_results(output_dir, feature_cols, drop_cols, removed_vif_features, confirmed_features, tentative_features, additional_features, selected_features_boruta, final_features):
    """Generate or save selection results."""
    selection_summary = []
    for feature in feature_cols:
        if feature in drop_cols:
            status = "Removed_Correlation"
        elif feature in removed_vif_features:
            status = "Removed_VIF"
        elif feature in confirmed_features:
            status = "Final_Selected_Confirmed" if feature in final_features else "Boruta_Confirmed_Not_Selected"
        elif feature in tentative_features:
            if feature in final_features:
                status = "Final_Selected_Tentative"
            elif feature in selected_features_boruta:
                status = "Boruta_Tentative_Not_Selected"
            else:
                status = "Boruta_Tentative"
        elif feature in additional_features:
            status = "Final_Selected_Boruta_Supplemented" if feature in final_features else "Boruta_Supplemented_Not_Selected"
        else:
            status = "Rejected_Boruta"
        selection_summary.append({"Feature": feature, "Status": status})
    pd.DataFrame(selection_summary).to_excel(output_dir / "feature_selection_summary.xlsx", index=False)


def save_final_features(output_dir, sample_ids, X_final_raw, y):
    """Generate or save final features."""
    sample_ids = sample_ids.reset_index(drop=True)
    X_final_raw = X_final_raw.reset_index(drop=True)
    y = y.reset_index(drop=True)
    if not (len(sample_ids) == len(X_final_raw) == len(y)):
        raise ValueError(f"Row number mismatch: Sample_ID={len(sample_ids)}, X={len(X_final_raw)}, Pn={len(y)}")
    if sample_ids.isna().any():
        raise ValueError("Sample_ID contains missing values.")
    if sample_ids.duplicated().any():
        duplicate_ids = sample_ids[sample_ids.duplicated()].tolist()
        raise ValueError(f"Duplicate Sample_ID detected: {duplicate_ids[:10]}")
    final_df = X_final_raw.copy()
    final_df.insert(0, "Sample_ID", sample_ids.values)
    final_df["Pn"] = y.values
    final_df.to_excel(output_dir / "selected_features.xlsx", index=False)
    final_df.to_csv(output_dir / "final_selected_features.csv", index=False, encoding="utf-8-sig")


def random_forest_analysis(output_dir, X_final, y):
    """Perform random forest analysis."""
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_final, y)
    imp = pd.DataFrame({"Feature": X_final.columns, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False).reset_index(drop=True)
    imp.to_csv(output_dir / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp, x="Importance", y="Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "rf_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    return rf, imp


def shap_analysis(output_dir, rf, X_final, imp):
    """Perform shap analysis."""
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_final)
    feature_names = X_final.columns.tolist()
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_final, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_final, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    top5_features = imp["Feature"].head(min(5, len(imp))).tolist()
    for feat in top5_features:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feat, shap_values, X_final, interaction_index="auto", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_dependence_{feat}.png", dpi=300, bbox_inches="tight")
        plt.close()
    print("\nComputing SHAP interaction values...")
    shap_interaction_values = explainer.shap_interaction_values(X_final)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_interaction_values, X_final, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_interaction_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    sample_id = min(10, len(X_final) - 1)
    shap_exp = shap.Explanation(
        values=shap_values[sample_id],
        base_values=explainer.expected_value,
        data=X_final.iloc[sample_id],
        feature_names=feature_names
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_waterfall_sample{sample_id}.png", dpi=300, bbox_inches="tight")
    plt.close()
    shap.force_plot(explainer.expected_value, shap_values[sample_id], X_final.iloc[sample_id], matplotlib=True, show=False)
    plt.savefig(output_dir / f"shap_force_sample{sample_id}.png", dpi=300, bbox_inches="tight")
    plt.close()
    decision_sample_num = min(100, len(X_final))
    plt.figure(figsize=(10, 6))
    shap.decision_plot(explainer.expected_value, shap_values[:decision_sample_num], X_final.iloc[:decision_sample_num], show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_decision_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(output_dir / "shap_values.csv", index=False, encoding="utf-8-sig")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs_shap}).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    shap_importance.to_csv(output_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")


def main():
    """Run the main analysis workflow."""
    sample_ids, X_raw, y, feature_cols = load_data(path, sheet, target, id_col)
    X = X_raw.copy()
    original_feature_num = X.shape[1]
    print("\n============================================================")
    print("Four-Stage Feature Selection 2.0")
    print("============================================================")
    print("Input file:", path)
    print("Target:", target)
    print("Original features:", original_feature_num)
    print("Feature range:", f"{MIN_FEATURE_NUM}-{MAX_FEATURE_NUM}")
    print("Correlation threshold:", CORRELATION_THRESHOLD)
    print("VIF threshold:", VIF_THRESHOLD)
    print("SHAP analysis:", "Enabled" if RUN_SHAP else "Disabled")
    if original_feature_num < MAX_FEATURE_NUM:
        raise ValueError(f"Only {original_feature_num} original features are available, which is fewer than MAX_FEATURE_NUM={MAX_FEATURE_NUM}.")
    X_after_corr, drop_cols = correlation_filter(X, CORRELATION_THRESHOLD)
    correlation_feature_num = X_after_corr.shape[1]
    print_stage("Stage 1: Correlation Filter", "Remove highly redundant features.", original_feature_num, correlation_feature_num, len(drop_cols))
    print("Threshold:", CORRELATION_THRESHOLD)
    print("Removed correlation features:", drop_cols)
    X_corr, removed_vif_features = vif_filter(X_after_corr, VIF_THRESHOLD)
    vif_feature_num = X_corr.shape[1]
    print_stage("Stage 2: VIF Filter", "Remove multicollinearity.", correlation_feature_num, vif_feature_num, len(removed_vif_features))
    print("Threshold:", VIF_THRESHOLD)
    print("Removed VIF features:", removed_vif_features)
    if vif_feature_num < MAX_FEATURE_NUM:
        raise ValueError(f"Only {vif_feature_num} features remain after VIF, which is fewer than MAX_FEATURE_NUM={MAX_FEATURE_NUM}.")
    X_boruta, boruta, confirmed_features, tentative_features, rejected_features, additional_features = boruta_filter(X_corr, y, MAX_FEATURE_NUM)
    confirmed_num = len(confirmed_features)
    tentative_num = len(tentative_features)
    rejected_num = len(rejected_features)
    boruta_retained_num = confirmed_num + tentative_num
    rfe_candidate_num = len(X_boruta.columns)
    selected_features_boruta = X_boruta.columns.tolist()
    print_stage("Stage 3: Boruta Filter", "Identify features associated with Pn.", vif_feature_num, boruta_retained_num, rejected_num)
    print("Confirmed:", confirmed_num)
    print("Tentative:", tentative_num)
    print("Rejected:", rejected_num)
    print("Confirmed + Tentative:", boruta_retained_num)
    print("\nConfirmed features:")
    print(confirmed_features)
    print("\nTentative features:")
    print(tentative_features)
    print("\nRejected features:")
    print(rejected_features)
    if len(additional_features) > 0:
        print("\nBoruta retained fewer than", MAX_FEATURE_NUM, "features.")
        print("Supplemented according to Boruta ranking:", additional_features)
        print("Features entering RFE:", rfe_candidate_num)
    else:
        print("\nBoruta retained enough features.")
        print("No supplementary features required.")
        print("Features entering RFE:", rfe_candidate_num)
    boruta_rank_df = pd.DataFrame({
        "Feature": X_corr.columns,
        "Boruta_Rank": boruta.ranking_,
        "Confirmed": boruta.support_,
        "Tentative": boruta.support_weak_
    }).sort_values(["Boruta_Rank", "Feature"])
    boruta_rank_df.to_excel(OUTPUT_ROOT / "boruta_feature_ranking.xlsx", index=False)
    print("\n============================================================")
    print("Stage 4: RFE Feature Number Evaluation")
    print("============================================================")
    print("RFE candidate features:", rfe_candidate_num)
    print("Testing feature numbers:", f"{MIN_FEATURE_NUM}-{MAX_FEATURE_NUM}")
    all_results = []
    rfe_rank_df, rfe = rfe_selection(X_boruta, y)
    rfe_rank_df.to_excel(OUTPUT_ROOT / "rfe_feature_ranking.xlsx", index=False)
    print("\nFull RFE ranking:")
    print(rfe_rank_df.to_string(index=False))
    for target_num in range(MIN_FEATURE_NUM, MAX_FEATURE_NUM + 1):
        output_dir = OUTPUT_ROOT / f"{target_num}_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_features = rfe_rank_df.head(target_num)["Feature"].tolist()
        X_final = X_boruta[selected_features].copy()
        X_final_raw = X_raw[selected_features].copy()
        final_feature_num = X_final.shape[1]
        rank_df = rfe_rank_df.copy()
        rank_df["Selected"] = rank_df["RFE_Rank"] <= target_num
        print("\n------------------------------------------------------------")
        print(f"RFE Target: {target_num} features")
        print("------------------------------------------------------------")
        print("Input features:", rfe_candidate_num)
        print("Remaining features:", final_feature_num)
        print("Final selected features:", selected_features)
        rank_df.to_excel(output_dir / "feature_rank.xlsx", index=False)
        boruta_rank_df.to_excel(output_dir / "boruta_feature_ranking.xlsx", index=False)
        save_selection_results(
            output_dir,
            feature_cols,
            drop_cols,
            removed_vif_features,
            confirmed_features,
            tentative_features,
            additional_features,
            selected_features_boruta,
            selected_features
        )
        save_final_features(output_dir, sample_ids, X_final_raw, y)
        rf, imp = random_forest_analysis(output_dir, X_final, y)
        if RUN_SHAP:
            print("SHAP analysis enabled.")
            shap_analysis(output_dir, rf, X_final, imp)
            print("SHAP analysis completed.")
        all_results.append({
            "Feature_Number": target_num,
            "RFE_Candidates": rfe_candidate_num,
            "Final_Feature_Number": final_feature_num,
            "Selected_Features": ", ".join(selected_features)
        })
    result_df = pd.DataFrame(all_results)
    result_df.to_excel(OUTPUT_ROOT / "feature_number_comparison.xlsx", index=False)
    result_df.to_csv(OUTPUT_ROOT / "feature_number_comparison.csv", index=False, encoding="utf-8-sig")
    stage_summary = pd.DataFrame({
        "Stage": [
            "Original",
            "After_Correlation",
            "After_VIF",
            "Boruta_Confirmed",
            "Boruta_Tentative",
            "Boruta_Confirmed_Tentative",
            "RFE_Candidates"
        ],
        "Feature_Count": [
            original_feature_num,
            correlation_feature_num,
            vif_feature_num,
            confirmed_num,
            tentative_num,
            boruta_retained_num,
            rfe_candidate_num
        ]
    })
    stage_summary.to_excel(OUTPUT_ROOT / "stage_summary.xlsx", index=False)
    print("\n============================================================")
    print("Feature Selection 2.0 Summary")
    print("============================================================")
    print("Stage 0 - Original:", original_feature_num)
    print("Stage 1 - Correlation:", original_feature_num, "->", correlation_feature_num)
    print("Stage 2 - VIF:", correlation_feature_num, "->", vif_feature_num)
    print("Stage 3 - Boruta:", vif_feature_num, "->", boruta_retained_num)
    print("  Confirmed:", confirmed_num)
    print("  Tentative:", tentative_num)
    print("  Rejected:", rejected_num)
    print("  Supplemented:", len(additional_features))
    print("  RFE candidates:", rfe_candidate_num)
    print("Stage 4 - RFE:", f"{MIN_FEATURE_NUM}-{MAX_FEATURE_NUM} features")
    print("\n============================================================")
    print("Output Directories")
    print("============================================================")
    for target_num in range(MIN_FEATURE_NUM, MAX_FEATURE_NUM + 1):
        print(f"{target_num} features:", OUTPUT_ROOT / f"{target_num}_features")
    print("\nCommon files:")
    print("- corr_heatmap.png")
    print("- boruta_feature_ranking.xlsx")
    print("- stage_summary.xlsx")
    print("- feature_number_comparison.xlsx")
    print("- feature_number_comparison.csv")
    print("\nEach feature-number folder contains:")
    print("- boruta_feature_ranking.xlsx")
    print("- feature_rank.xlsx")
    print("- feature_selection_summary.xlsx")
    print("- selected_features.xlsx")
    print("- final_selected_features.csv")
    print("- rf_feature_importance.csv")
    print("- rf_feature_importance.png")
    if RUN_SHAP:
        print("- shap_summary.png")
        print("- shap_bar.png")
        print("- shap_dependence_*.png")
        print("- shap_interaction_summary.png")
        print("- shap_waterfall_sample*.png")
        print("- shap_force_sample*.png")
        print("- shap_decision_plot.png")
        print("- shap_values.csv")
        print("- shap_importance.csv")
    print("============================================================")


if __name__ == "__main__":
    main()