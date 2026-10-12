"""Build and visualize phenotypic fingerprints of agronomic observations."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"G:\PhyMaize")
FEATURE_FILE = ROOT / r"features\feature_selection_results\20_features\selected_features.xlsx"
LABEL_FILE = ROOT / r"annotations\lables.csv"
META_FILE = ROOT / r"meta data\samples_description.xlsx"
SPLIT_DIR = ROOT / r"features\splits"
OUT_DIR = ROOT / r"agronomic_analysis_results\phenotypic_fingerprint"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "Sample_ID"
PN_COL = "Pn"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

FEATURE_ORDER = [
    "LiDAR_intensity_mean", "Green_mean", "Green_std", "Green_homogeneity", "COV", "CWSI",
    "Green_correlation", "Canopy_S", "Canopy_Volume", "LiDAR_intensity_std", "Thermal_energy",
    "Thermal_dissimilarity", "NIR_contrast", "Green_contrast", "Thermal_correlation",
    "Canopy_LAI", "Temp_min", "MTCI_std", "CIrededge_std", "Canopy_K"
]

FEATURE_LABELS = {
    "LiDAR_intensity_mean": r"$I_{\mathrm{mean}}$",
    "Green_mean": r"$Green_{\mathrm{mean}}$",
    "Green_std": r"$Green_{\mathrm{std}}$",
    "Green_homogeneity": r"$Green_{\mathrm{homo}}$",
    "COV": "COV",
    "CWSI": "CWSI",
    "Green_correlation": r"$Green_{\mathrm{corr}}$",
    "Canopy_S": "S",
    "Canopy_Volume": "CV",
    "LiDAR_intensity_std": r"$I_{\mathrm{std}}$",
    "Thermal_energy": r"$T_{\mathrm{energy}}$",
    "Thermal_dissimilarity": r"$T_{\mathrm{diss}}$",
    "NIR_contrast": r"$NIR_{\mathrm{cont}}$",
    "Green_contrast": r"$Green_{\mathrm{cont}}$",
    "Thermal_correlation": r"$T_{\mathrm{corr}}$",
    "Canopy_LAI": "LAI",
    "Temp_min": r"$T_{\mathrm{min}}$",
    "MTCI_std": r"$MTCI_{\mathrm{std}}$",
    "CIrededge_std": r"$CI_{\mathrm{rededge,std}}$",
    "Canopy_K": "K"
}


def norm_id(x):
    """Perform norm id."""
    x = str(x).strip()
    if x.startswith("PM_"):
        return x
    try:
        return f"PM_{int(float(x)):04d}"
    except Exception:
        return x


def find_col(df, candidates):
    """Retrieve or load col for subsequent analysis."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


print("\n================ DATA LOADING ================\n")

features = pd.read_excel(FEATURE_FILE)
labels = pd.read_csv(LABEL_FILE)
meta = pd.read_excel(META_FILE)

for name, df in [("features", features), ("labels", labels), ("meta", meta)]:
    if ID_COL not in df.columns:
        raise ValueError(f"{name} is missing required column: {ID_COL}")

features[ID_COL] = features[ID_COL].map(norm_id)
labels[ID_COL] = labels[ID_COL].map(norm_id)
meta[ID_COL] = meta[ID_COL].map(norm_id)

pn_col = find_col(labels, ["Ground_truth Pn", "Ground_truth_Pn", "Pn", "GroundTruth_Pn"])

if pn_col is None:
    raise ValueError(
        "Pn column not found in labels file. "
        "Expected one of: Ground_truth Pn, Ground_truth_Pn, Pn, GroundTruth_Pn."
    )

labels = labels[[ID_COL, pn_col]].copy()
labels = labels.rename(columns={pn_col: PN_COL})
labels[PN_COL] = pd.to_numeric(labels[PN_COL], errors="coerce")

missing_features = [f for f in FEATURE_ORDER if f not in features.columns]
if missing_features:
    raise ValueError(f"Missing required features: {missing_features}")

feature_cols = FEATURE_ORDER.copy()

print(f"Feature samples: {len(features)}")
print(f"Pn labels: {len(labels)}")
print(f"Metadata samples: {len(meta)}")
print(f"Selected features: {len(feature_cols)}")

print("\n================ SAMPLE SPLITS ================\n")

train_file = SPLIT_DIR / "train.csv"
val_file = SPLIT_DIR / "val.csv"
if not val_file.exists():
    val_file = SPLIT_DIR / "validation.csv"
test_file = SPLIT_DIR / "test.csv"

for file in [train_file, val_file, test_file]:
    if not file.exists():
        raise FileNotFoundError(f"Missing: {file}")

train_ids = pd.read_csv(train_file)[ID_COL].map(norm_id)
val_ids = pd.read_csv(val_file)[ID_COL].map(norm_id)
test_ids = pd.read_csv(test_file)[ID_COL].map(norm_id)

analysis_ids = pd.Index(pd.concat([train_ids, val_ids]).drop_duplicates())
test_ids = pd.Index(test_ids.drop_duplicates())

print(f"Train samples: {len(train_ids)}")
print(f"Validation samples: {len(val_ids)}")
print(f"Independent test samples: {len(test_ids)}")
print(f"Train + validation unique samples: {len(analysis_ids)}")

overlap_test = analysis_ids.intersection(test_ids)

if len(overlap_test) > 0:
    raise ValueError(
        f"Detected {len(overlap_test)} samples overlapping between "
        f"train/validation and independent test sets."
    )

print("\n================ DATA MERGING ================\n")

features = features[features[ID_COL].isin(analysis_ids)].copy()
labels = labels[labels[ID_COL].isin(analysis_ids)].copy()
meta = meta[meta[ID_COL].isin(analysis_ids)].copy()

features = features.drop_duplicates(ID_COL)
labels = labels.drop_duplicates(ID_COL)
meta = meta.drop_duplicates(ID_COL)

data = (
    features[[ID_COL] + feature_cols]
    .merge(labels[[ID_COL, PN_COL]], on=ID_COL, how="inner")
    .merge(meta, on=ID_COL, how="left")
)

required_meta = ["Year", "Site", "Stage"]
missing_meta = [c for c in required_meta if c not in data.columns]

if missing_meta:
    raise ValueError(f"Missing required metadata columns: {missing_meta}")

data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data["Site"] = data["Site"].astype("string").str.strip()
data["Stage"] = data["Stage"].astype("string").str.strip()

data["Environment"] = (
    data["Year"].astype("Int64").astype(str) + "_"
    + data["Site"].astype(str) + "_"
    + data["Stage"].astype(str)
)

data = data.replace(["<NA>", "nan", "None"], np.nan)

before_drop = len(data)
data = data.dropna(subset=[PN_COL] + feature_cols + ["Year", "Site", "Stage"]).copy()

print(f"Merged samples: {before_drop}")
print(f"Complete analysis samples: {len(data)}")
print(f"Number of environments: {data['Environment'].nunique()}")

print("\nEnvironment distribution:")
print(data.groupby("Environment")[ID_COL].count().sort_index().to_string())

if data["Environment"].nunique() < 2:
    raise ValueError(
        "Fewer than two environments were detected. "
        "Please check Year, Site, and Stage in samples_description.xlsx."
    )

print("\n================ HIGH-Pn DEFINITION ================\n")

data["Environment_Pn_Q75"] = data.groupby("Environment")[PN_COL].transform("quantile", 0.75)
data["High_Pn"] = (data[PN_COL] >= data["Environment_Pn_Q75"]).astype(int)

group_counts = data["High_Pn"].value_counts().rename(index={0: "Other Pn", 1: "High Pn"})
print(group_counts)

print("\nHigh-Pn proportion:")
print(f"{data['High_Pn'].mean() * 100:.2f}%")

environment_summary = (
    data.groupby("Environment")
    .agg(
        N=(ID_COL, "size"),
        Pn_mean=(PN_COL, "mean"),
        Pn_median=(PN_COL, "median"),
        Pn_Q75=("Environment_Pn_Q75", "first"),
        High_Pn_N=("High_Pn", "sum")
    )
    .reset_index()
)

environment_summary["High_Pn_proportion"] = (
    environment_summary["High_Pn_N"] / environment_summary["N"]
)

environment_summary.to_excel(TABLE_DIR / "environment_high_pn_definition.xlsx", index=False)

data[
    [ID_COL, PN_COL, "Year", "Site", "Stage", "Environment", "Environment_Pn_Q75", "High_Pn"]
].to_excel(TABLE_DIR / "phenotypic_fingerprint_sample_groups.xlsx", index=False)

print("\n================ STANDARDIZATION ================\n")

X = data[feature_cols].copy()
feature_mean = X.mean()
feature_std = X.std(ddof=1)

zero_sd_features = feature_std[feature_std <= 0].index.tolist()

if zero_sd_features:
    raise ValueError(f"Features with zero standard deviation: {zero_sd_features}")

Z = (X - feature_mean) / feature_std
Z = Z.replace([np.inf, -np.inf], np.nan)

valid = Z.notna().all(axis=1)

if (~valid).sum() > 0:
    print(f"Removing {(~valid).sum()} samples with invalid standardized values.")

data = data.loc[valid].copy()
Z = Z.loc[valid].copy()

Z["High_Pn"] = data["High_Pn"].values
Z[ID_COL] = data[ID_COL].values
Z[PN_COL] = data[PN_COL].values
Z["Environment"] = data["Environment"].values

standardization_table = pd.DataFrame({
    "Feature": feature_cols,
    "Mean": feature_mean.values,
    "SD": feature_std.values
})

standardization_table.to_excel(
    TABLE_DIR / "fingerprint_standardization_parameters.xlsx",
    index=False
)

Z[[ID_COL, PN_COL, "Environment", "High_Pn"] + feature_cols].to_excel(
    TABLE_DIR / "sample_phenotypic_fingerprint_vectors.xlsx",
    index=False
)

print(f"Valid fingerprint samples: {len(Z)}")

print("\n================ GROUP FINGERPRINT STATISTICS ================\n")

groups = {
    "High Pn": Z.loc[Z["High_Pn"] == 1, feature_cols],
    "Other Pn": Z.loc[Z["High_Pn"] == 0, feature_cols]
}

if len(groups["High Pn"]) == 0 or len(groups["Other Pn"]) == 0:
    raise ValueError("High-Pn or Other-Pn group contains no samples.")

summary = []

for group, d in groups.items():
    for feature in feature_cols:
        x = d[feature].dropna().values
        summary.append({
            "Group": group,
            "Feature": feature,
            "N": len(x),
            "Mean": np.mean(x),
            "Median": np.median(x),
            "SD": np.std(x, ddof=1),
            "Q25": np.quantile(x, 0.25),
            "Q75": np.quantile(x, 0.75),
            "IQR": np.quantile(x, 0.75) - np.quantile(x, 0.25),
            "Min": np.min(x),
            "Max": np.max(x)
        })

summary = pd.DataFrame(summary)

summary.to_excel(
    TABLE_DIR / "phenotypic_fingerprint_group_statistics.xlsx",
    index=False
)

print(
    summary[["Group", "Feature", "N", "Median", "Q25", "Q75", "IQR"]]
    .to_string(index=False)
)

print("\n================ FINGERPRINT DISTANCE ================\n")

high_profile = groups["High Pn"].median().values
other_profile = groups["Other Pn"].median().values

centroid_euclidean = np.linalg.norm(high_profile - other_profile)
centroid_mahalanobis = np.nan

try:
    covariance = np.cov(Z[feature_cols].values, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * 1e-6
    inv_covariance = np.linalg.pinv(covariance)
    delta = high_profile - other_profile
    centroid_mahalanobis = np.sqrt(delta.T @ inv_covariance @ delta)
except Exception as e:
    print(f"Mahalanobis distance calculation failed: {e}")

distance_table = pd.DataFrame({
    "Metric": [
        "Euclidean distance between group median fingerprints",
        "Mahalanobis distance between group median fingerprints"
    ],
    "Value": [centroid_euclidean, centroid_mahalanobis]
})

distance_table.to_excel(
    TABLE_DIR / "phenotypic_fingerprint_distance.xlsx",
    index=False
)

print(distance_table.to_string(index=False))

print("\n================ RADAR FINGERPRINT ================\n")

N_FEATURES = len(feature_cols)

angles = np.linspace(0, 2 * np.pi, N_FEATURES, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])

high = groups["High Pn"]
other = groups["Other Pn"]

high_median = high.median().values
other_median = other.median().values

high_q25 = high.quantile(0.25).values
high_q75 = high.quantile(0.75).values
other_q25 = other.quantile(0.25).values
other_q75 = other.quantile(0.75).values

high_median_closed = np.concatenate([high_median, [high_median[0]]])
high_q25_closed = np.concatenate([high_q25, [high_q25[0]]])
high_q75_closed = np.concatenate([high_q75, [high_q75[0]]])
other_q25_closed = np.concatenate([other_q25, [other_q25[0]]])
other_q75_closed = np.concatenate([other_q75, [other_q75[0]]])

from matplotlib.patches import Patch

fig = plt.figure(figsize=(9.5, 9.5))
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles)
ax.set_xticklabels([FEATURE_LABELS[f] for f in feature_cols], fontsize=15)
ax.tick_params(axis="x", pad=24)

ax.set_ylim(-3, 3)
ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_yticklabels(["-3", "−2", "−1", "0", "1", "2", "3"], fontsize=15)

ax.grid(linewidth=0.6, alpha=0.35)
ax.spines["polar"].set_linewidth(0.9)

sample_high = high.sample(min(180, len(high)), random_state=42)
sample_other = other.sample(min(180, len(other)), random_state=42)

for _, row in sample_other.iterrows():
    values = row[feature_cols].values.astype(float)
    values = np.concatenate([values, [values[0]]])
    ax.plot(
        angles_closed,
        values,
        linewidth=1.20,
        alpha=0.10,
        color="#00A087",
        zorder=1
    )

for _, row in sample_high.iterrows():
    values = row[feature_cols].values.astype(float)
    values = np.concatenate([values, [values[0]]])
    ax.plot(
        angles_closed,
        values,
        linewidth=2.00,
        alpha=0.11,
        color="#E64B35",
        zorder=2
    )

ax.fill_between(
    angles_closed,
    other_q25_closed,
    other_q75_closed,
    color="#00A087",
    alpha=0.12,
    zorder=3
)

ax.fill_between(
    angles_closed,
    high_q25_closed,
    high_q75_closed,
    color="#E64B35",
    alpha=0.20,
    zorder=4
)

ax.plot(
    angles_closed,
    high_q25_closed,
    color="#E64B35",
    linewidth=1.2,
    alpha=0.75,
    zorder=5
)

ax.plot(
    angles_closed,
    high_q75_closed,
    color="#E64B35",
    linewidth=1.2,
    alpha=0.75,
    zorder=5
)

ax.plot(
    angles_closed,
    high_median_closed,
    linewidth=3.0,
    color="#FF00FF",
    label="High Pn",
    zorder=6
)

ax.scatter(
    angles,
    high_median,
    s=30,
    color="#FF00FF",
    zorder=7
)

legend_handles = [
    plt.Line2D(
        [0], [0],
        color="#00A087",
        linewidth=1.5,
        alpha=0.55,
        label="Other Pn samples"
    ),
    plt.Line2D(
        [0], [0],
        color="#E64B35",
        linewidth=1.5,
        alpha=0.55,
        label="High Pn samples (≥75th percentile)"
    ),
    plt.Line2D(
        [0], [0],
        color="#FF00FF",
        linewidth=3.0,
        label="High Pn median"
    ),
    Patch(
        facecolor="#E64B35",
        alpha=0.20,
        edgecolor="none",
        label="High-Pn region (IQR)"
    )
]

ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.19),
    ncol=2,
    frameon=False,
    fontsize=15,
    handlelength=2.5,
    columnspacing=1.8,
    handletextpad=0.6
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "HighPn_Phenotypic_Fingerprint_Radar.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    FIG_DIR / "HighPn_Phenotypic_Fingerprint_Radar.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print(FIG_DIR / "HighPn_Phenotypic_Fingerprint_Radar.png")
print(FIG_DIR / "HighPn_Phenotypic_Fingerprint_Radar.pdf")

print("\n================ ANALYSIS COMPLETED ================\n")
print(f"Output directory: {OUT_DIR}")