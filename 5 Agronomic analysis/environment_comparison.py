"""Compare agronomic and photosynthesis results across environments."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# Paths
# ================================================================

ROOT = Path(r"G:\PhyMaize")

INPUT_FILE = ROOT / r"agronomic_analysis_results\tables\environment_feature_Pn_correlations.xlsx"
OUT_DIR = ROOT / r"agronomic_analysis_results\figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUT_DIR / "Fig10_Environment_Feature_Pn_Correlation.png"
OUTPUT_PDF = OUT_DIR / "Fig10_Environment_Feature_Pn_Correlation.pdf"


# ================================================================
# Parameters
# ================================================================

FONT_SIZE = 20
FIGSIZE = (15, 12)

FEATURE_ORDER = [
    "LiDAR_intensity_mean",
    "Green_mean",
    "Green_std",
    "Green_homogeneity",
    "COV",
    "CWSI",
    "Green_correlation",
    "Canopy_S",
    "Canopy_Volume",
    "LiDAR_intensity_std",
    "Thermal_energy",
    "Thermal_dissimilarity",
    "NIR_contrast",
    "Green_contrast",
    "Thermal_correlation",
    "Canopy_LAI",
    "Temp_min",
    "MTCI_std",
    "CIrededge_std",
    "Canopy_K"
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

ENVIRONMENT_ORDER = [
    "2025_Fangshan_V6",
    "2025_Fangshan_V9",
    "2025_Fangshan_V12",
    "2025_Gongzhuling_V6",
    "2025_Gongzhuling_V9",
    "2025_Gongzhuling_V12",
    "2026_Gongzhuling_V6",
    "2026_Gongzhuling_V9",
    "2026_Gongzhuling_V12"
]


# ================================================================
# Plot settings
# ================================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = FONT_SIZE
plt.rcParams["axes.unicode_minus"] = False

sns.set_style("white")


# ================================================================
# Load correlation results
# ================================================================

print("\n================ LOAD DATA ================\n")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_excel(INPUT_FILE)

required_cols = [
    "Environment",
    "Feature",
    "Pearson_r"
]

missing = [
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print(f"Input file: {INPUT_FILE}")
print(f"Rows: {len(df)}")
print(f"Environments: {df['Environment'].nunique()}")
print(f"Features: {df['Feature'].nunique()}")


# ================================================================
# Check feature coverage
# ================================================================

missing_features = [
    f for f in FEATURE_ORDER
    if f not in df["Feature"].unique()
]

if missing_features:
    raise ValueError(
        f"Missing features:\n{missing_features}"
    )


# ================================================================
# Check environment coverage
# ================================================================

missing_envs = [
    e for e in ENVIRONMENT_ORDER
    if e not in df["Environment"].unique()
]

if missing_envs:
    raise ValueError(
        f"Missing environments:\n{missing_envs}"
    )


# ================================================================
# Construct 20 × 9 correlation matrix
# ================================================================

print("\n================ CONSTRUCT MATRIX ================\n")

corr_matrix = df.pivot(
    index="Feature",
    columns="Environment",
    values="Pearson_r"
)

corr_matrix = corr_matrix.reindex(
    index=FEATURE_ORDER,
    columns=ENVIRONMENT_ORDER
)

if corr_matrix.shape != (20, 9):
    raise ValueError(
        f"Unexpected matrix shape: {corr_matrix.shape}. "
        f"Expected (20, 9)."
    )

if corr_matrix.isna().any().any():
    missing_cells = np.argwhere(
        corr_matrix.isna().values
    )

    print("Missing correlation cells:")

    for r, c in missing_cells:
        print(
            FEATURE_ORDER[r],
            ENVIRONMENT_ORDER[c]
        )

    raise ValueError(
        "Correlation matrix contains missing values."
    )

print(corr_matrix.to_string())


# ================================================================
# Display labels
# ================================================================

y_labels = [
    FEATURE_LABELS.get(
        feature,
        feature
    )
    for feature in FEATURE_ORDER
]

x_labels = [
    environment.replace("_", " ")
    for environment in ENVIRONMENT_ORDER
]


# ================================================================
# Plot
# ================================================================

print("\n================ PLOT FIGURE ================\n")

fig, ax = plt.subplots(
    figsize=FIGSIZE
)

sns.heatmap(
    corr_matrix,
    cmap="RdYlGn_r",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5,
    linecolor="white",
    annot=False,
    xticklabels=False,
    yticklabels=False,
    cbar=True,
    cbar_kws={
        "label": "Pearson's r",
        "shrink": 0.82,
        "pad": 0.025
    },
    ax=ax
)


# ================================================================
# Manually set y-axis ticks and labels
# ================================================================

ax.set_yticks(
    np.arange(len(y_labels)) + 0.5
)

ax.set_yticklabels(
    y_labels,
    rotation=0,
    fontsize=FONT_SIZE
)


# ================================================================
# Manually set x-axis ticks and labels
# ================================================================

ax.set_xticks(
    np.arange(len(x_labels)) + 0.5
)

ax.set_xticklabels(
    x_labels,
    rotation=30,
    ha="right",
    fontsize=FONT_SIZE
)


# ================================================================
# Axis labels
# ================================================================

ax.set_xlabel("")
ax.set_ylabel("")


# ================================================================
# Colorbar
# ================================================================

cbar = ax.collections[0].colorbar

cbar.set_label(
    "Pearson's r",
    fontsize=FONT_SIZE,
    labelpad=12
)

cbar.ax.tick_params(
    labelsize=FONT_SIZE,
    width=0.8,
    length=4
)


# ================================================================
# Axis appearance
# ================================================================

ax.tick_params(
    axis="both",
    width=0.8,
    length=4,
    colors="black"
)

ax.tick_params(
    axis="y",
    length=0
)


# ================================================================
# Title
# ================================================================

ax.set_title(
    "Environment-specific associations between UAV-derived features and leaf Pn",
    fontsize=FONT_SIZE,
    fontweight="bold",
    pad=18
)


# ================================================================
# Save
# ================================================================

plt.tight_layout()

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("\n================ DONE ================\n")
print(f"Matrix shape: {corr_matrix.shape}")
print(f"PNG: {OUTPUT_PNG}")
print(f"PDF: {OUTPUT_PDF}")