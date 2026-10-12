"""Analyze associations between agronomic variables and photosynthetic outcomes."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

ROOT = Path(r"G:\PhyMaize")
FEATURE_FILE = ROOT / r"features\feature_selection_results\20_features\selected_features.xlsx"
LABEL_FILE = ROOT / r"annotations\lables.csv"
META_FILE = ROOT / r"meta data\samples_description.xlsx"
SPLIT_DIR = ROOT / r"features\splits"
OUT_DIR = ROOT / r"agronomic_analysis_results"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

ID_COL = "Sample_ID"
TARGET_COL = "__TARGET_PN__"

def norm_id(x):
    """Perform norm id."""
    x = str(x).strip()
    if x.startswith("PM_"):
        return x
    try:
        return f"PM_{int(float(x)):04d}"
    except:
        return x

def find_col(df, candidates):
    """Retrieve or load col for subsequent analysis."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def numeric_extract(x):
    """Perform numeric extract."""
    return pd.to_numeric(x.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")

def sample_group(s):
    """Perform sample group."""
    try:
        n = int(str(s).replace("PM_", ""))
    except:
        return "Unknown"
    if 1 <= n <= 972:
        return "2025_Fangshan"
    if 973 <= n <= 2316:
        return "2026_Gongzhuling"
    if 2317 <= n <= 2376:
        return "2025_Gongzhuling"
    return "Unknown"

def safe_formula_col(c):
    """Perform safe formula col."""
    return f'Q("{c}")'

def cluster_ols(formula, df, cluster_col="PlotKey"):
    """Perform cluster ols."""
    x = df.dropna(subset=[cluster_col]).copy()
    if len(x) < 20 or x[cluster_col].nunique() < 5:
        return None
    try:
        return smf.ols(formula, data=x).fit(cov_type="cluster", cov_kwds={"groups": x[cluster_col]})
    except:
        try:
            return smf.ols(formula, data=x).fit()
        except:
            return None

def extract_interaction(model, term):
    """Perform extract interaction."""
    if model is None or term not in model.params.index:
        return np.nan, np.nan
    return model.params[term], model.pvalues[term]

def adjusted_r2(model):
    """Perform adjusted r2."""
    return np.nan if model is None else model.rsquared_adj

def high_pn_factor_table(df, factor):
    """Perform high pn factor table."""
    x = df.groupby(factor).agg(Sample_N=("High_Pn", "size"), HighPn_N=("High_Pn", "sum"), HighPn_Rate=("High_Pn", "mean")).reset_index()
    x["HighPn_Rate"] *= 100
    return x

print("\n================ DATA LOADING ================\n")

features = pd.read_excel(FEATURE_FILE)
labels = pd.read_csv(LABEL_FILE)
meta = pd.read_excel(META_FILE)

for df in [features, labels, meta]:
    if ID_COL not in df.columns:
        raise ValueError(f"Missing required column: {ID_COL}")

features[ID_COL] = features[ID_COL].map(norm_id)
labels[ID_COL] = labels[ID_COL].map(norm_id)
meta[ID_COL] = meta[ID_COL].map(norm_id)

pn_col = find_col(labels, ["Ground_truth Pn", "Ground_truth_Pn", "Pn", "GroundTruth_Pn"])
if pn_col is None:
    raise ValueError("Pn column not found in label file.")

labels = labels[[ID_COL, pn_col]].copy().rename(columns={pn_col: TARGET_COL})
labels[TARGET_COL] = pd.to_numeric(labels[TARGET_COL], errors="coerce")

exclude_feature_cols = {ID_COL, TARGET_COL, "Pn", "Ground_truth Pn", "Ground_truth_Pn", "GroundTruth_Pn"}
feature_cols = [c for c in features.columns if c not in exclude_feature_cols and pd.api.types.is_numeric_dtype(features[c])]

print(f"Detected numeric remote-sensing features: {len(feature_cols)}")

if len(feature_cols) != 20:
    print("\nDetected feature columns:")
    for i, c in enumerate(feature_cols, 1):
        print(f"{i:02d}. {c}")
    raise ValueError(f"Expected 20 selected remote-sensing features, but detected {len(feature_cols)}.")

val_file = SPLIT_DIR / "val.csv"
if not val_file.exists():
    val_file = SPLIT_DIR / "validation.csv"

train_file = SPLIT_DIR / "train.csv"
test_file = SPLIT_DIR / "test.csv"

if not train_file.exists() or not val_file.exists() or not test_file.exists():
    raise FileNotFoundError("train.csv / val.csv(or validation.csv) / test.csv not all found.")

train_ids = pd.read_csv(train_file)[ID_COL].map(norm_id)
val_ids = pd.read_csv(val_file)[ID_COL].map(norm_id)
test_ids = pd.read_csv(test_file)[ID_COL].map(norm_id)

analysis_ids = pd.Index(pd.concat([train_ids, val_ids]).drop_duplicates())
test_ids = pd.Index(test_ids.drop_duplicates())

print(f"Total samples in full dataset definition: {len(set(train_ids) | set(val_ids) | set(test_ids))}")
print(f"Train samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Independent test samples excluded: {len(test_ids)}")
print(f"Samples available for agronomic analysis: {len(analysis_ids)}")

features = features[features[ID_COL].isin(analysis_ids)].copy()
labels = labels[labels[ID_COL].isin(analysis_ids)].copy()
meta = meta[meta[ID_COL].isin(analysis_ids)].copy()

features = features.drop_duplicates(ID_COL)
labels = labels.drop_duplicates(ID_COL)
meta = meta.drop_duplicates(ID_COL)

data = features.merge(labels[[ID_COL, TARGET_COL]], on=ID_COL, how="inner").merge(meta, on=ID_COL, how="left")
data["Pn"] = data[TARGET_COL]
data["Dataset"] = data[ID_COL].map(sample_group)

density_col = find_col(data, ["Plant_Density", "Density", "Plant density", "PlantDensity"])
n_col = find_col(data, ["Nitrogen", "N", "Nitrogen_Rate", "N_rate"])
variety_col = find_col(data, ["Variety", "variety"])
stage_col = find_col(data, ["Stage", "Growth_Stage", "growth_stage"])
year_col = find_col(data, ["Year", "year"])
site_col = find_col(data, ["Site", "site"])
plot_col = find_col(data, ["Plot_ID", "PlotID", "plot_id"])
subplot_col = find_col(data, ["Subplot_ID", "SubplotID", "subplot_id"])

data["Density"] = numeric_extract(data[density_col]) if density_col else np.nan
data["N"] = numeric_extract(data[n_col]) if n_col else np.nan
data["Variety"] = data[variety_col].astype(str) if variety_col else "Unknown"
data["Stage"] = data[stage_col].astype(str) if stage_col else "Unknown"
data["Year"] = pd.to_numeric(data[year_col], errors="coerce") if year_col else np.nan
data["Site"] = data[site_col].astype(str) if site_col else "Unknown"

data["Environment"] = data["Year"].astype(str) + "_" + data["Site"].astype(str) + "_" + data["Stage"].astype(str)

if plot_col:
    data["Plot_ID"] = data[plot_col]
    data["PlotKey"] = data["Year"].astype(str) + "_" + data["Site"].astype(str) + "_" + data[plot_col].astype(str)
else:
    data["Plot_ID"] = np.nan
    data["PlotKey"] = data[ID_COL]

print(f"\nTotal merged samples: {len(data)}")
print(f"Remote-sensing features: {len(feature_cols)}")

print("\nDataset distribution:")
print(data["Dataset"].value_counts().sort_index())

print("\n================ SAMPLE COVERAGE CHECK ================\n")

full_expected = pd.Index([f"PM_{i:04d}" for i in range(1, 2377)], name=ID_COL)
feature_ids = set(pd.read_excel(FEATURE_FILE)[ID_COL].map(norm_id))
label_ids = set(pd.read_csv(LABEL_FILE)[ID_COL].map(norm_id))
meta_ids = set(pd.read_excel(META_FILE)[ID_COL].map(norm_id))
train_ids_set = set(train_ids)
val_ids_set = set(val_ids)
test_ids_set = set(test_ids)
analysis_ids_set = set(analysis_ids)
merged_ids_set = set(data[ID_COL])

coverage = pd.DataFrame({ID_COL: full_expected})
coverage["Dataset"] = coverage[ID_COL].map(sample_group)
coverage["Feature"] = coverage[ID_COL].isin(feature_ids)
coverage["Label"] = coverage[ID_COL].isin(label_ids)
coverage["Metadata"] = coverage[ID_COL].isin(meta_ids)
coverage["Train"] = coverage[ID_COL].isin(train_ids_set)
coverage["Validation"] = coverage[ID_COL].isin(val_ids_set)
coverage["Independent_Test"] = coverage[ID_COL].isin(test_ids_set)
coverage["Agronomic_Analysis"] = coverage[ID_COL].isin(analysis_ids_set)
coverage["Merged"] = coverage[ID_COL].isin(merged_ids_set)

coverage.to_excel(TABLE_DIR / "sample_coverage.xlsx", index=False)

coverage_summary = coverage.groupby("Dataset").agg(
    Total=("Sample_ID", "size"),
    Feature=("Feature", "sum"),
    Label=("Label", "sum"),
    Metadata=("Metadata", "sum"),
    Train=("Train", "sum"),
    Validation=("Validation", "sum"),
    Independent_Test=("Independent_Test", "sum"),
    Agronomic_Analysis=("Agronomic_Analysis", "sum"),
    Merged=("Merged", "sum")
).reset_index()

coverage_summary.to_excel(TABLE_DIR / "sample_coverage_summary.xlsx", index=False)
print(coverage_summary.to_string(index=False))

print("\n================ DATA QUALITY ================\n")

quality_cols = ["Pn"] + feature_cols + ["N", "Density", "Variety", "Stage"]
quality = pd.DataFrame({
    "Variable": quality_cols,
    "N": [data[c].notna().sum() for c in quality_cols],
    "Missing": [data[c].isna().sum() for c in quality_cols]
})
quality["Missing_Rate"] = quality["Missing"] / len(data)
quality.to_excel(TABLE_DIR / "data_quality.xlsx", index=False)
print(quality[quality["Missing"] > 0].to_string(index=False))

print(f"\nPn-complete samples: {data['Pn'].notna().sum()}")

print("\n================ DESCRIPTIVE STATISTICS ================\n")

desc_cols = feature_cols + ["Pn", "N", "Density"]
desc = data[desc_cols].describe().T
desc["Missing"] = data[desc_cols].isna().sum()
desc.to_excel(TABLE_DIR / "descriptive_statistics.xlsx")

dataset_desc = data.groupby("Dataset")[["Pn"] + feature_cols].agg(["count", "mean", "std", "min", "max"])
dataset_desc.to_excel(TABLE_DIR / "dataset_descriptive_statistics.xlsx")

print("\n================ PART I: AGRONOMIC EFFECTS ON 20 FEATURES ================\n")

part1 = []

for feature in feature_cols:
    f = safe_formula_col(feature)
    model = cluster_ols(f"{f} ~ N + Density + C(Variety) + C(Environment)", data)
    if model is None:
        continue
    for factor, term in [("Nitrogen", "N"), ("Density", "Density")]:
        if term in model.params.index:
            part1.append({"Feature": feature, "Factor": factor, "Coefficient": model.params.get(term, np.nan), "P": model.pvalues.get(term, np.nan), "Adjusted_R2": model.rsquared_adj, "N": int(model.nobs)})

part1 = pd.DataFrame(part1)

if not part1.empty:
    part1["FDR_P"] = np.nan
    for factor in part1["Factor"].unique():
        idx = part1["Factor"] == factor
        part1.loc[idx, "FDR_P"] = multipletests(part1.loc[idx, "P"].fillna(1), method="fdr_bh")[1]
    part1.to_excel(TABLE_DIR / "part1_agronomic_effects.xlsx", index=False)

print(f"Part I completed: {len(part1)} results")

print("\n================ PART II: FEATURE × NITROGEN → Pn ================\n")

part2 = []

for feature in feature_cols:
    f = safe_formula_col(feature)
    d = data[["Pn", feature, "N", "Density", "Variety", "Environment", "PlotKey"]].dropna().copy()
    if len(d) < 30 or d["N"].nunique() < 2 or d[feature].std() == 0:
        continue
    d[feature] = (d[feature] - d[feature].mean()) / d[feature].std()
    full = cluster_ols(f"Pn ~ {f} * N + Density + C(Variety) + C(Environment)", d)
    reduced = cluster_ols(f"Pn ~ {f} + N + Density + C(Variety) + C(Environment)", d)
    if full is None or reduced is None:
        continue
    term = f"{f}:N"
    coef, p = extract_interaction(full, term)
    part2.append({"Feature": feature, "Interaction_Coefficient": coef, "P": p, "FDR_P": np.nan, "Full_Adjusted_R2": adjusted_r2(full), "Reduced_Adjusted_R2": adjusted_r2(reduced), "Delta_Adjusted_R2": adjusted_r2(full)-adjusted_r2(reduced), "N": len(d)})

part2 = pd.DataFrame(part2)

if not part2.empty:
    part2["FDR_P"] = multipletests(part2["P"].fillna(1), method="fdr_bh")[1]
    part2.to_excel(TABLE_DIR / "part2_feature_N_interactions.xlsx", index=False)

print(f"Part II completed: {len(part2)} results")

print("\n================ PART III: FEATURE × DENSITY → Pn ================\n")

part3 = []

for feature in feature_cols:
    f = safe_formula_col(feature)
    d = data[["Pn", feature, "N", "Density", "Variety", "Environment", "PlotKey"]].dropna().copy()
    if len(d) < 30 or d["Density"].nunique() < 2 or d[feature].std() == 0:
        continue
    d[feature] = (d[feature] - d[feature].mean()) / d[feature].std()
    full = cluster_ols(f"Pn ~ {f} * Density + N + C(Variety) + C(Environment)", d)
    reduced = cluster_ols(f"Pn ~ {f} + Density + N + C(Variety) + C(Environment)", d)
    if full is None or reduced is None:
        continue
    term = f"{f}:Density"
    coef, p = extract_interaction(full, term)
    part3.append({"Feature": feature, "Interaction_Coefficient": coef, "P": p, "FDR_P": np.nan, "Full_Adjusted_R2": adjusted_r2(full), "Reduced_Adjusted_R2": adjusted_r2(reduced), "Delta_Adjusted_R2": adjusted_r2(full)-adjusted_r2(reduced), "N": len(d)})

part3 = pd.DataFrame(part3)

if not part3.empty:
    part3["FDR_P"] = multipletests(part3["P"].fillna(1), method="fdr_bh")[1]
    part3.to_excel(TABLE_DIR / "part3_feature_density_interactions.xlsx", index=False)

print(f"Part III completed: {len(part3)} results")

print("\n================ PART IV: FEATURE × VARIETY → Pn ================\n")

part4 = []

for feature in feature_cols:
    f = safe_formula_col(feature)
    d = data[["Pn", feature, "N", "Density", "Variety", "Environment", "PlotKey"]].dropna().copy()
    if len(d) < 30 or d["Variety"].nunique() < 2 or d[feature].std() == 0:
        continue
    d[feature] = (d[feature] - d[feature].mean()) / d[feature].std()
    full = cluster_ols(f"Pn ~ {f} * C(Variety) + N + Density + C(Environment)", d)
    reduced = cluster_ols(f"Pn ~ {f} + N + Density + C(Variety) + C(Environment)", d)
    if full is None or reduced is None:
        continue
    interaction_terms = [x for x in full.params.index if x.startswith(f"{f}:C(Variety)")]
    pvals = full.pvalues[interaction_terms].dropna()
    if len(pvals) == 0:
        continue
    part4.append({"Feature": feature, "Interaction_P_Min": pvals.min(), "FDR_P": np.nan, "Full_Adjusted_R2": adjusted_r2(full), "Reduced_Adjusted_R2": adjusted_r2(reduced), "Delta_Adjusted_R2": adjusted_r2(full)-adjusted_r2(reduced), "N": len(d), "Variety_Levels": d["Variety"].nunique()})

part4 = pd.DataFrame(part4)

if not part4.empty:
    part4["FDR_P"] = multipletests(part4["Interaction_P_Min"].fillna(1), method="fdr_bh")[1]
    part4.to_excel(TABLE_DIR / "part4_feature_variety_interactions.xlsx", index=False)

print(f"Part IV completed: {len(part4)} results")

print("\n================ HIGH-Pn PHENOTYPIC SIGNATURE ================\n")

high_base = data.dropna(subset=["Pn"] + feature_cols).copy()
high_base["High_Pn"] = high_base.groupby("Environment")["Pn"].transform(lambda x: x >= x.quantile(0.75)).astype(int)

print("\nHigh-Pn distribution:")
print(high_base["High_Pn"].value_counts().sort_index())

high_base[["Sample_ID", "Dataset", "Environment", "Pn", "High_Pn"]].to_excel(TABLE_DIR / "high_pn_samples.xlsx", index=False)

signature = []

for feature in feature_cols:
    a = high_base.loc[high_base["High_Pn"] == 1, feature].values
    b = high_base.loc[high_base["High_Pn"] == 0, feature].values
    if len(a) < 5 or len(b) < 5:
        continue
    t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    pooled_sd = np.sqrt(((len(a)-1)*np.var(a, ddof=1)+(len(b)-1)*np.var(b, ddof=1))/(len(a)+len(b)-2))
    d = (np.mean(a)-np.mean(b))/pooled_sd if pooled_sd > 0 else np.nan
    signature.append({"Feature": feature, "HighPn_Mean": np.mean(a), "Other_Mean": np.mean(b), "Difference": np.mean(a)-np.mean(b), "Cohens_d": d, "T": t, "P": p, "FDR_P": np.nan})

signature = pd.DataFrame(signature)

if not signature.empty:
    signature["FDR_P"] = multipletests(signature["P"].fillna(1), method="fdr_bh")[1]
    signature["Abs_Cohens_d"] = signature["Cohens_d"].abs()
    signature = signature.sort_values(["FDR_P", "Abs_Cohens_d"])
    signature.to_excel(TABLE_DIR / "high_pn_feature_signature.xlsx", index=False)

print(f"High-Pn signature completed: {len(signature)} features")

print("\n================ HIGH-Pn DISTRIBUTION BY AGRONOMIC FACTORS ================\n")

for factor in ["Dataset", "Environment", "N", "Density", "Variety", "Stage"]:
    d = high_base.dropna(subset=[factor]).copy()
    if d.empty:
        continue
    x = high_pn_factor_table(d, factor)
    x.to_excel(TABLE_DIR / f"high_pn_by_{factor}.xlsx", index=False)
    print(f"\n{factor}:")
    print(x.to_string(index=False))

print("\n================ ENVIRONMENT-SPECIFIC FEATURE-Pn CORRELATIONS ================\n")

corr_results = []

for env, d in high_base.groupby("Environment"):
    if len(d) < 15:
        continue
    for feature in feature_cols:
        r = d[[feature, "Pn"]].corr().iloc[0, 1]
        corr_results.append({"Environment": env, "Feature": feature, "Pearson_r": r, "N": d[[feature, "Pn"]].dropna().shape[0]})

corr_results = pd.DataFrame(corr_results)
corr_results.to_excel(TABLE_DIR / "environment_feature_Pn_correlations.xlsx", index=False)

print(f"Correlation results: {len(corr_results)}")

print("\n================ OVERALL ADJUSTED FEATURE-Pn ASSOCIATION ================\n")

overall = []

for feature in feature_cols:
    f = safe_formula_col(feature)
    d = data[["Pn", feature, "N", "Density", "Variety", "Environment", "PlotKey"]].dropna().copy()
    if len(d) < 30 or d[feature].std() == 0:
        continue
    d[feature] = (d[feature]-d[feature].mean())/d[feature].std()
    model = cluster_ols(f"Pn ~ {f} + N + Density + C(Variety) + C(Environment)", d)
    if model is None:
        continue
    overall.append({"Feature": feature, "Standardized_Coefficient": model.params.get(f, np.nan), "P": model.pvalues.get(f, np.nan), "Adjusted_R2": model.rsquared_adj, "N": int(model.nobs)})

overall = pd.DataFrame(overall)

if not overall.empty:
    overall["FDR_P"] = multipletests(overall["P"].fillna(1), method="fdr_bh")[1]
    overall["Abs_Standardized_Coefficient"] = overall["Standardized_Coefficient"].abs()
    overall = overall.sort_values("Abs_Standardized_Coefficient", ascending=False)
    overall.to_excel(TABLE_DIR / "overall_adjusted_feature_Pn_association.xlsx", index=False)

print(f"Overall adjusted analysis completed: {len(overall)} features")

print("\n================ AGRONOMIC SUMMARY ================\n")

summary = []

for factor in ["N", "Density", "Variety", "Stage"]:
    d = data.dropna(subset=["Pn", factor]).copy()
    if len(d) == 0:
        continue
    summary.append({"Factor": factor, "Available_Samples": len(d), "Levels": d[factor].nunique(), "Pn_Mean": d["Pn"].mean(), "Pn_SD": d["Pn"].std()})

summary = pd.DataFrame(summary)
summary.to_excel(TABLE_DIR / "agronomic_analysis_summary.xlsx", index=False)
print(summary.to_string(index=False))

print("\n================ FINAL DATASET SUMMARY ================\n")

final_summary = pd.DataFrame({
    "Item": ["Full dataset samples", "Independent test samples excluded", "Train + validation samples", "Merged analysis samples", "Pn-complete samples", "Number of selected features", "2025 Fangshan", "2026 Gongzhuling", "2025 Gongzhuling"],
    "Value": [len(full_expected), len(test_ids), len(analysis_ids), len(data), data["Pn"].notna().sum(), len(feature_cols), (data["Dataset"]=="2025_Fangshan").sum(), (data["Dataset"]=="2026_Gongzhuling").sum(), (data["Dataset"]=="2025_Gongzhuling").sum()]
})

final_summary.to_excel(TABLE_DIR / "final_dataset_summary.xlsx", index=False)
print(final_summary.to_string(index=False))

print("\n========================================================")
print("Agronomic association analysis completed.")
print(f"Results saved to: {OUT_DIR}")
print("========================================================")

print("\n================ FIGURE GENERATION ================\n")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("white")

feature_order = feature_cols.copy()

CMAP_RDBU = "RdBu_r"
CMAP_RDYLGN = "RdYlGn_r"
CMAP_RDPU = "RdPu"
CMAP_YLGN = "YlGn"
CMAP_PUBU = "PuBu"

def savefig(name):
    """Perform savefig."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()

def clean_feature_name(x):
    """Perform clean feature name."""
    return str(x).replace("_mean", "").replace("_std", "").replace("_median", "").replace("_min", "").replace("_max", "")

def add_sig(ax, p):
    """Perform add sig."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def style_axis(ax):
    """Perform style axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.6, length=4, colors="black")
    ax.xaxis.label.set_fontfamily("Arial")
    ax.yaxis.label.set_fontfamily("Arial")
    ax.title.set_fontfamily("Arial")

print("\nFigure 1: Agronomic effects heatmap")

if not part1.empty:
    p1 = part1.pivot(index="Feature", columns="Factor", values="Coefficient").reindex(feature_order)

    fig, ax = plt.subplots(figsize=(7.2, 8.5))

    sns.heatmap(
        p1,
        cmap=CMAP_RDBU,
        center=0,
        linewidths=.4,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 9, "fontfamily": "Arial"},
        cbar_kws={"label": "Effect coefficient"},
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([clean_feature_name(x) for x in ax.get_yticklabels()], rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.set_title(
        "Effects of agronomic factors on UAV-derived features",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    savefig("Fig1_Agronomic_Effects_Heatmap.png")

print("\nFigure 2: High-Pn phenotypic signature")

if not signature.empty:
    s = signature.set_index("Feature").reindex(feature_order).dropna(subset=["Cohens_d"])
    s = s.sort_values("Cohens_d")

    fig, ax = plt.subplots(figsize=(7.2, 8.5))

    y = np.arange(len(s))
    values = s["Cohens_d"].values

    cmap = plt.get_cmap(CMAP_RDYLGN)
    norm = plt.Normalize(
        vmin=min(values.min(), -0.01),
        vmax=max(values.max(), 0.01)
    )

    bar_colors = cmap(norm(values))

    ax.barh(
        y,
        values,
        height=.65,
        color=bar_colors,
        edgecolor="none"
    )

    ax.axvline(0, lw=0.9, color="black")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [clean_feature_name(x) for x in s.index],
        fontsize=10
    )

    ax.set_xlabel("Cohen's d", fontsize=13, fontweight="bold")
    ax.set_ylabel("")

    ax.set_title(
        "UAV phenotypic signature associated with high Pn",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    for i, (_, row) in enumerate(s.iterrows()):
        if row["FDR_P"] < 0.05:
            offset = 0.025 if row["Cohens_d"] >= 0 else -0.025
            ha = "left" if row["Cohens_d"] >= 0 else "right"

            ax.text(
                row["Cohens_d"] + offset,
                i,
                "*",
                va="center",
                ha=ha,
                fontsize=13,
                fontweight="bold"
            )

    style_axis(ax)
    savefig("Fig2_HighPn_Phenotypic_Signature.png")

print("\nFigure 3: High-Pn standardized feature profile")

if not high_base.empty:
    z = high_base[feature_cols].copy()
    z = pd.DataFrame(
        (z - z.mean()) / z.std(),
        columns=feature_cols,
        index=z.index
    )

    z["High_Pn"] = high_base["High_Pn"].values

    profile = z.groupby("High_Pn")[feature_cols].mean().T
    profile.columns = ["Other Pn", "High Pn"]
    profile = profile.reindex(feature_order)

    fig, ax = plt.subplots(figsize=(6.5, 8.5))

    sns.heatmap(
        profile,
        cmap=CMAP_RDBU,
        center=0,
        linewidths=.3,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Standardized feature value"},
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(
        [clean_feature_name(x) for x in ax.get_yticklabels()],
        rotation=0
    )

    ax.set_title(
        "UAV phenotypic profile of high-Pn samples",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    savefig("Fig3_HighPn_Feature_Profile_Heatmap.png")

print("\nFigure 4: Key feature responses to nitrogen")

n_key = []

if not signature.empty:
    n_key = signature.sort_values(
        ["FDR_P", "Abs_Cohens_d"]
    ).head(4)["Feature"].tolist()

if len(n_key) == 0:
    n_key = feature_cols[:4]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7)
)

axes = axes.flatten()

n_colors = sns.color_palette(CMAP_YLGN, 6)

for ax, feature, color in zip(axes, n_key, [n_colors[3]] * len(n_key)):

    d = data[["N", feature, "Dataset"]].dropna().copy()

    if d.empty or d["N"].nunique() < 2:
        ax.axis("off")
        continue

    d["N"] = pd.to_numeric(d["N"], errors="coerce")
    d = d.dropna(subset=["N"])

    sns.pointplot(
        data=d,
        x="N",
        y=feature,
        errorbar="se",
        color=color,
        linewidth=1.4,
        markers="o",
        ax=ax
    )

    ax.set_xlabel("Nitrogen rate", fontsize=11)
    ax.set_ylabel(clean_feature_name(feature), fontsize=11)
    ax.set_title(
        clean_feature_name(feature),
        fontsize=11,
        fontweight="bold"
    )

    style_axis(ax)

fig.suptitle(
    "Responses of key UAV features to nitrogen",
    y=1.01,
    fontsize=13,
    fontweight="bold"
)

savefig("Fig4_Key_Features_Nitrogen_Response.png")

print("\nFigure 5: Key feature responses to planting density")

d_key = []

if not signature.empty:
    d_key = signature.sort_values(
        ["FDR_P", "Abs_Cohens_d"]
    ).head(4)["Feature"].tolist()

if len(d_key) == 0:
    d_key = feature_cols[:4]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7)
)

axes = axes.flatten()

d_colors = sns.color_palette(CMAP_PUBU, 6)

for ax, feature, color in zip(axes, d_key, [d_colors[3]] * len(d_key)):

    d = data[["Density", feature]].dropna().copy()

    if d.empty or d["Density"].nunique() < 2:
        ax.axis("off")
        continue

    d["Density"] = pd.to_numeric(d["Density"], errors="coerce")
    d = d.dropna(subset=["Density"])

    sns.pointplot(
        data=d,
        x="Density",
        y=feature,
        errorbar="se",
        color=color,
        linewidth=1.4,
        markers="o",
        ax=ax
    )

    ax.set_xlabel("Plant density", fontsize=11)
    ax.set_ylabel(clean_feature_name(feature), fontsize=11)
    ax.set_title(
        clean_feature_name(feature),
        fontsize=11,
        fontweight="bold"
    )

    style_axis(ax)

fig.suptitle(
    "Responses of key UAV features to planting density",
    y=1.01,
    fontsize=13,
    fontweight="bold"
)

savefig("Fig5_Key_Features_Density_Response.png")

print("\nFigure 6: Key feature-Pn relationships")

corr_rank = overall.sort_values(
    ["FDR_P", "Abs_Standardized_Coefficient"]
) if not overall.empty else pd.DataFrame()

p_key = (
    corr_rank.head(4)["Feature"].tolist()
    if not corr_rank.empty
    else feature_cols[:4]
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7)
)

axes = axes.flatten()

p_color = sns.color_palette(CMAP_RDPU, 6)[4]

for ax, feature in zip(axes, p_key):

    d = data[[feature, "Pn"]].dropna().copy()

    if len(d) < 10:
        ax.axis("off")
        continue

    x = d[feature].values
    y = d["Pn"].values

    r, p = pearsonr(x, y)

    coef = np.polyfit(x, y, 1)

    xx = np.linspace(
        x.min(),
        x.max(),
        100
    )

    yy = coef[0] * xx + coef[1]

    ax.scatter(
        x,
        y,
        s=15,
        alpha=.35,
        color=p_color,
        edgecolors="none"
    )

    ax.plot(
        xx,
        yy,
        lw=1.6,
        color=p_color
    )

    ax.text(
        .05,
        .94,
        f"r = {r:.2f}\nP = {p:.2e}",
        transform=ax.transAxes,
        va="top",
        fontsize=9
    )

    ax.set_xlabel(
        clean_feature_name(feature),
        fontsize=11
    )

    ax.set_ylabel(
        "Leaf Pn (μmol m⁻² s⁻¹)",
        fontsize=11
    )

    ax.set_title(
        clean_feature_name(feature),
        fontsize=11,
        fontweight="bold"
    )

    style_axis(ax)

fig.suptitle(
    "Relationships between UAV-derived features and leaf Pn",
    y=1.01,
    fontsize=13,
    fontweight="bold"
)

savefig("Fig6_Key_Feature_Pn_Relationships.png")

print("\nFigure 7: High-Pn rate under nitrogen")

d = high_base.dropna(subset=["N"]).copy()

if not d.empty:

    rate = d.groupby("N")["High_Pn"].agg(
        ["mean", "count"]
    ).reset_index()

    rate["Rate"] = rate["mean"] * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    sns.barplot(
        data=rate,
        x="N",
        y="Rate",
        errorbar=None,
        color=sns.color_palette(CMAP_YLGN, 6)[3],
        ax=ax
    )

    ax.set_xlabel(
        "Nitrogen rate",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel(
        "High-Pn samples (%)",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_title(
        "High-Pn proportion across nitrogen treatments",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    for i, row in rate.iterrows():
        ax.text(
            i,
            row["Rate"] + 1,
            f"{row['Rate']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    style_axis(ax)
    savefig("Fig7_HighPn_Rate_Nitrogen.png")

print("\nFigure 8: High-Pn rate under planting density")

d = high_base.dropna(subset=["Density"]).copy()

if not d.empty:

    rate = d.groupby("Density")["High_Pn"].agg(
        ["mean", "count"]
    ).reset_index()

    rate["Rate"] = rate["mean"] * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    sns.barplot(
        data=rate,
        x="Density",
        y="Rate",
        errorbar=None,
        color=sns.color_palette(CMAP_PUBU, 6)[3],
        ax=ax
    )

    ax.set_xlabel(
        "Plant density",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel(
        "High-Pn samples (%)",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_title(
        "High-Pn proportion across planting densities",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    for i, row in rate.iterrows():
        ax.text(
            i,
            row["Rate"] + 1,
            f"{row['Rate']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    style_axis(ax)
    savefig("Fig8_HighPn_Rate_Density.png")

print("\nFigure 9: Variety phenotypic profile")

v = high_base.dropna(subset=["Variety"]).copy()

if not v.empty and v["Variety"].nunique() >= 2:

    means = v.groupby("Variety")[feature_cols].mean()

    means = (
        means - means.mean()
    ) / means.std()

    means = means.T

    fig, ax = plt.subplots(figsize=(9, 8))

    sns.heatmap(
        means,
        cmap=CMAP_RDYLGN,
        center=0,
        linewidths=.3,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Standardized feature value"},
        ax=ax
    )

    ax.set_xlabel(
        "Variety",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylabel("")

    ax.set_yticklabels(
        [clean_feature_name(x) for x in ax.get_yticklabels()],
        rotation=0
    )

    ax.set_title(
        "Variety-specific UAV phenotypic profiles",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    savefig("Fig9_Variety_Phenotypic_Profile.png")

print("\nFigure 10: Environment-specific feature-Pn correlation heatmap")

if not corr_results.empty:

    cr = corr_results.pivot(
        index="Feature",
        columns="Environment",
        values="Pearson_r"
    ).reindex(feature_order)

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        cr,
        cmap=CMAP_RDBU,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=.3,
        linecolor="white",
        annot=False,
        cbar_kws={"label": "Pearson's r"},
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_yticklabels(
        [clean_feature_name(x) for x in ax.get_yticklabels()],
        rotation=0
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    ax.set_title(
        "Environment-specific associations between UAV features and leaf Pn",
        pad=12,
        fontsize=13,
        fontweight="bold"
    )

    savefig("Fig10_Environment_Feature_Pn_Correlation.png")

print("\n================ FIGURE GENERATION COMPLETED ================\n")
print(f"Figures saved to: {FIG_DIR}")