"""Select informative training features using statistical and model-based methods."""
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

TARGET_FEATURE_RANGE = range(1, 26)
CORRELATION_THRESHOLD = 0.95
VIF_THRESHOLD = 10
RUN_SHAP = False

target = "Pn"
id_col = "Sample_ID"


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
    feature_cols = [c for c in df.columns if c not in [target, id_col]]
    X = df[feature_cols].copy()
    y = df[target].copy()
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y, feature_cols


def correlation_filter(X, threshold, output_dir):
    """Perform correlation filter."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    corr = X_scaled.corr().abs()
    plt.figure(figsize=(18, 15))
    ax = sns.heatmap(
        corr,
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(
        output_dir / "corr_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )
    drop_cols = [
        col
        for col in upper.columns
        if any(upper[col] > threshold)
    ]
    X_after_corr = X_scaled.drop(
        columns=drop_cols
    ).copy()
    return X_after_corr, drop_cols


def calc_vif(data):
    """Compute vif."""
    return pd.DataFrame({
        "Feature": data.columns,
        "VIF": [
            variance_inflation_factor(
                data.values,
                i
            )
            for i in range(data.shape[1])
        ]
    })


def vif_filter(X, threshold):
    """Perform vif filter."""
    X_vif = X.copy()
    removed_vif_features = []
    while X_vif.shape[1] > 1:
        vif_df = calc_vif(X_vif)
        max_vif = vif_df["VIF"].max()
        if max_vif < threshold:
            break
        remove_feature = vif_df.loc[
            vif_df["VIF"].idxmax(),
            "Feature"
        ]
        print(
            "Remove VIF:",
            remove_feature,
            "| VIF:",
            max_vif
        )
        removed_vif_features.append(
            remove_feature
        )
        X_vif = X_vif.drop(
            columns=[remove_feature]
        )
    return X_vif.copy(), removed_vif_features


def boruta_filter(X, y, target_num):
    """Perform boruta filter."""
    rf_boruta = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    boruta = BorutaPy(
        estimator=rf_boruta,
        n_estimators="auto",
        verbose=2,
        random_state=42
    )
    boruta.fit(
        X.values,
        y.values
    )
    confirmed_features = X.columns[
        boruta.support_
    ].tolist()
    tentative_features = X.columns[
        boruta.support_weak_
    ].tolist()
    rejected_features = X.columns[
        ~(boruta.support_ | boruta.support_weak_)
    ].tolist()
    selected_mask = (
        boruta.support_ |
        boruta.support_weak_
    )
    selected_features = X.columns[
        selected_mask
    ].tolist()
    additional_features = []
    if len(selected_features) < target_num:
        boruta_ranking = pd.DataFrame({
            "Feature": X.columns,
            "Boruta_Rank": boruta.ranking_,
            "Confirmed": boruta.support_,
            "Tentative": boruta.support_weak_
        }).sort_values(
            "Boruta_Rank",
            ascending=True
        ).reset_index(drop=True)
        for feature in boruta_ranking["Feature"]:
            if feature not in selected_features:
                additional_features.append(
                    feature
                )
            if (
                len(selected_features) +
                len(additional_features)
                >= target_num
            ):
                break
        selected_features += additional_features
    X_boruta = X[
        selected_features
    ].copy()
    return (
        X_boruta,
        boruta,
        confirmed_features,
        tentative_features,
        rejected_features,
        additional_features
    )


def rfe_selection(X, y, target_num):
    """Perform rfe selection."""
    svr = SVR(kernel="linear")
    rfe = RFE(
        estimator=svr,
        n_features_to_select=target_num
    )
    rfe.fit(X, y)
    final_features = X.columns[
        rfe.support_
    ].tolist()
    X_final = X[
        final_features
    ].copy()
    return X_final, final_features, rfe


def save_selection_results(
    feature_cols,
    drop_cols,
    removed_vif_features,
    confirmed_features,
    tentative_features,
    additional_features,
    selected_features_boruta,
    final_features,
    output_dir
):
    """Generate or save selection results."""
    selection_summary = []
    for feature in feature_cols:
        if feature in drop_cols:
            status = "Removed_Correlation"
        elif feature in removed_vif_features:
            status = "Removed_VIF"
        elif feature in confirmed_features:
            status = (
                "Final_Selected_Confirmed"
                if feature in final_features
                else "Boruta_Confirmed_Not_Selected"
            )
        elif feature in tentative_features:
            if feature in final_features:
                status = "Final_Selected_Tentative"
            elif feature in selected_features_boruta:
                status = "Boruta_Tentative_Not_Selected"
            else:
                status = "Boruta_Tentative"
        elif feature in additional_features:
            status = (
                "Final_Selected_Boruta_Supplemented"
                if feature in final_features
                else "Boruta_Supplemented_Not_Selected"
            )
        else:
            status = "Rejected_Boruta"
        selection_summary.append({
            "Feature": feature,
            "Status": status
        })
    pd.DataFrame(
        selection_summary
    ).to_excel(
        output_dir / "feature_selection_summary.xlsx",
        index=False
    )


def save_final_features(X_final, y, output_dir):
    """Generate or save final features."""
    final_df = pd.concat(
        [
            X_final.reset_index(drop=True),
            y.reset_index(drop=True)
        ],
        axis=1
    )
    final_df.to_excel(
        output_dir / "selected_features.xlsx",
        index=False
    )
    final_df.to_csv(
        output_dir / "final_selected_features.csv",
        index=False,
        encoding="utf-8-sig"
    )


def random_forest_analysis(X_final, y, output_dir):
    """Perform random forest analysis."""
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(
        X_final,
        y
    )
    imp = pd.DataFrame({
        "Feature": X_final.columns,
        "Importance": rf.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)
    imp.to_csv(
        output_dir / "rf_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=imp,
        x="Importance",
        y="Feature"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "rf_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    return rf, imp


def shap_analysis(rf, X_final, imp, output_dir):
    """Perform shap analysis."""
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(
        X_final
    )
    feature_names = X_final.columns.tolist()
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_final,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_final,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    top5_features = imp[
        "Feature"
    ].head(5).tolist()
    for feat in top5_features:
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            feat,
            shap_values,
            X_final,
            interaction_index="auto",
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"shap_dependence_{feat}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
    print(
        "\nComputing SHAP interaction values..."
    )
    shap_interaction_values = (
        explainer.shap_interaction_values(
            X_final
        )
    )
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_interaction_values,
        X_final,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_interaction_summary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    sample_id = min(
        10,
        len(X_final) - 1
    )
    shap_exp = shap.Explanation(
        values=shap_values[sample_id],
        base_values=explainer.expected_value,
        data=X_final.iloc[sample_id],
        feature_names=feature_names
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(
        shap_exp,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"shap_waterfall_sample{sample_id}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    shap.force_plot(
        explainer.expected_value,
        shap_values[sample_id],
        X_final.iloc[sample_id],
        matplotlib=True,
        show=False
    )
    plt.savefig(
        output_dir / f"shap_force_sample{sample_id}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    decision_sample_num = min(
        100,
        len(X_final)
    )
    plt.figure(figsize=(10, 6))
    shap.decision_plot(
        explainer.expected_value,
        shap_values[:decision_sample_num],
        X_final.iloc[:decision_sample_num],
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "shap_decision_plot.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )
    shap_df.to_csv(
        output_dir / "shap_values.csv",
        index=False,
        encoding="utf-8-sig"
    )
    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": mean_abs_shap
    }).sort_values(
        "MeanAbsSHAP",
        ascending=False
    ).reset_index(drop=True)
    shap_importance.to_csv(
        output_dir / "shap_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )


def run_feature_selection(target_feature_num, output_dir):
    """Perform run feature selection."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    X, y, feature_cols = load_data(
        path,
        sheet,
        target,
        id_col
    )
    original_feature_num = X.shape[1]
    print("\n")
    print("#" * 70)
    print(
        f"开始特征筛选：TARGET_FEATURE_NUM = "
        f"{target_feature_num}"
    )
    print(
        f"输出目录：{output_dir}"
    )
    print("#" * 70)
    X_after_corr, drop_cols = correlation_filter(
        X,
        CORRELATION_THRESHOLD,
        output_dir
    )
    correlation_feature_num = (
        X_after_corr.shape[1]
    )
    print_stage(
        "Stage 1: Correlation Filter",
        "Remove highly redundant features.",
        original_feature_num,
        correlation_feature_num,
        len(drop_cols)
    )
    X_corr, removed_vif_features = vif_filter(
        X_after_corr,
        VIF_THRESHOLD
    )
    vif_feature_num = X_corr.shape[1]
    print_stage(
        "Stage 2: VIF Filter",
        "Remove multicollinearity.",
        correlation_feature_num,
        vif_feature_num,
        len(removed_vif_features)
    )
    X_boruta, boruta, confirmed_features, tentative_features, rejected_features, additional_features = boruta_filter(
        X_corr,
        y,
        target_feature_num
    )
    confirmed_num = len(
        confirmed_features
    )
    tentative_num = len(
        tentative_features
    )
    rejected_num = len(
        rejected_features
    )
    boruta_retained_num = (
        confirmed_num +
        tentative_num
    )
    selected_features_boruta = (
        X_boruta.columns.tolist()
    )
    print_stage(
        "Stage 3: Boruta Filter",
        "Identify features associated with Pn.",
        vif_feature_num,
        boruta_retained_num,
        rejected_num
    )
    print(
        "Confirmed:",
        confirmed_num
    )
    print(
        "Tentative:",
        tentative_num
    )
    print(
        "Rejected:",
        rejected_num
    )
    print(
        "Confirmed + Tentative:",
        boruta_retained_num
    )
    if len(additional_features) > 0:
        print(
            "\nBoruta retained fewer than",
            target_feature_num,
            "features."
        )
        print(
            "Supplemented according to Boruta ranking:",
            additional_features
        )
    rfe_candidate_num = len(
        selected_features_boruta
    )
    if rfe_candidate_num < target_feature_num:
        raise ValueError(
            f"Only {rfe_candidate_num} features remain "
            f"after Boruta, which is fewer than "
            f"TARGET_FEATURE_NUM={target_feature_num}."
        )
    boruta_rank_df = pd.DataFrame({
        "Feature": X_corr.columns,
        "Boruta_Rank": boruta.ranking_,
        "Confirmed": boruta.support_,
        "Tentative": boruta.support_weak_
    }).sort_values(
        "Boruta_Rank",
        ascending=True
    )
    boruta_rank_df.to_excel(
        output_dir / "boruta_feature_ranking.xlsx",
        index=False
    )
    X_final, final_features, rfe = rfe_selection(
        X_boruta,
        y,
        target_feature_num
    )
    final_feature_num = X_final.shape[1]
    print_stage(
        "Stage 4: RFE",
        "Recursively eliminate less informative features.",
        rfe_candidate_num,
        final_feature_num,
        rfe_candidate_num - final_feature_num
    )
    print(
        "Final selected features:",
        final_features
    )
    rank_df = pd.DataFrame({
        "Feature": X_boruta.columns,
        "RFE_Rank": rfe.ranking_,
        "Selected": rfe.support_
    }).sort_values(
        ["RFE_Rank", "Feature"]
    )
    rank_df.to_excel(
        output_dir / "feature_rank.xlsx",
        index=False
    )
    save_selection_results(
        feature_cols,
        drop_cols,
        removed_vif_features,
        confirmed_features,
        tentative_features,
        additional_features,
        selected_features_boruta,
        final_features,
        output_dir
    )
    save_final_features(
        X_final,
        y,
        output_dir
    )
    rf, imp = random_forest_analysis(
        X_final,
        y,
        output_dir
    )
    if RUN_SHAP:
        shap_analysis(
            rf,
            X_final,
            imp,
            output_dir
        )
    stage_summary = pd.DataFrame({
        "Stage": [
            "Original",
            "After_Correlation",
            "After_VIF",
            "Boruta_Confirmed",
            "Boruta_Tentative",
            "Boruta_Confirmed_Tentative",
            "RFE_Candidates",
            "Final_RFE"
        ],
        "Feature_Count": [
            original_feature_num,
            correlation_feature_num,
            vif_feature_num,
            confirmed_num,
            tentative_num,
            boruta_retained_num,
            rfe_candidate_num,
            final_feature_num
        ]
    })
    stage_summary.to_excel(
        output_dir / "stage_summary.xlsx",
        index=False
    )
    print("\n最终特征：")
    for i, feature in enumerate(
        final_features,
        start=1
    ):
        print(
            f"{i}. {feature}"
        )
    print(
        f"\n{target_feature_num} features "
        f"筛选完成。"
    )
    return final_features


def main():
    """Run the main analysis workflow."""
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )
    print("=" * 70)
    print("PhyMaize 批量特征筛选")
    print("=" * 70)
    print(
        f"目标特征数量："
        f"{min(TARGET_FEATURE_RANGE)} ~ "
        f"{max(TARGET_FEATURE_RANGE)}"
    )
    print(
        f"输入文件：{path}"
    )
    print(
        f"输出根目录：{OUTPUT_ROOT}"
    )
    all_results = {}
    for target_feature_num in TARGET_FEATURE_RANGE:
        output_dir = (
            OUTPUT_ROOT /
            f"{target_feature_num}_features"
        )
        try:
            final_features = run_feature_selection(
                target_feature_num,
                output_dir
            )
            all_results[
                target_feature_num
            ] = final_features
        except Exception as e:
            print("\n" + "!" * 70)
            print(
                f"{target_feature_num} features "
                f"运行失败：{e}"
            )
            print("!" * 70)
    # ========================================================
    # ========================================================
    summary_rows = []
    for n, features in all_results.items():
        row = {
            "Feature_Number": n
        }
        for i, feature in enumerate(
            features,
            start=1
        ):
            row[f"Feature_{i}"] = feature
        summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(
            summary_rows
        ).sort_values(
            "Feature_Number"
        )
        summary_df.to_excel(
            OUTPUT_ROOT /
            "all_feature_selection_summary.xlsx",
            index=False
        )
        summary_df.to_csv(
            OUTPUT_ROOT /
            "all_feature_selection_summary.csv",
            index=False,
            encoding="utf-8-sig"
        )
    print("\n")
    print("=" * 70)
    print("全部批量特征筛选完成")
    print("=" * 70)
    print(
        f"成功完成："
        f"{len(all_results)} / "
        f"{len(TARGET_FEATURE_RANGE)}"
    )
    print(
        f"结果目录：{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()