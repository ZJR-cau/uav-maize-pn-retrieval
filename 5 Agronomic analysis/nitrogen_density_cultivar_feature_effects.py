"""Evaluate nitrogen, planting density, and cultivar effects on features."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(r"G:\PhyMaize")
TABLE_DIR = ROOT / r"agronomic_analysis_results\tables"
FIG_DIR = ROOT / r"agronomic_analysis_results\figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.unicode_minus"] = False

PART2_FILE = TABLE_DIR / "part2_feature_N_interactions.xlsx"
PART3_FILE = TABLE_DIR / "part3_feature_density_interactions.xlsx"
PART4_FILE = TABLE_DIR / "part4_feature_variety_interactions.xlsx"

FEATURE_ORDER = [
    "I_mean", "Green_mean", "Green_std", "Green_homo", "COV", "CWSI",
    "Green_corr", "S", "CV", "I_std", "T_energy", "T_diss", "NIR_cont",
    "Green_cont", "T_corr", "LAI", "T_min", "MTCI_std", "CIrededge_std", "K"
]

FEATURE_LABELS = {
    "I_mean": r"$I_{\mathrm{mean}}$",
    "Green_mean": r"$Green_{\mathrm{mean}}$",
    "Green_std": r"$Green_{\mathrm{std}}$",
    "Green_homo": r"$Green_{\mathrm{homo}}$",
    "COV": "COV",
    "CWSI": "CWSI",
    "Green_corr": r"$Green_{\mathrm{corr}}$",
    "S": "S",
    "CV": "CV",
    "I_std": r"$I_{\mathrm{std}}$",
    "T_energy": r"$T_{\mathrm{energy}}$",
    "T_diss": r"$T_{\mathrm{diss}}$",
    "NIR_cont": r"$NIR_{\mathrm{cont}}$",
    "Green_cont": r"$Green_{\mathrm{cont}}$",
    "T_corr": r"$T_{\mathrm{corr}}$",
    "LAI": "LAI",
    "T_min": r"$T_{\mathrm{min}}$",
    "MTCI_std": r"$MTCI_{\mathrm{std}}$",
    "CIrededge_std": r"$CI_{\mathrm{rededge,std}}$",
    "K": "K"
}

FACTOR_COLORS = {
    "Nitrogen": "#8E8BFE",
    "Plant density": "#FEA3A2",
    "Variety": "#86D5B8"
}

FACTORS = ["Nitrogen", "Plant density", "Variety"]


def sig_marker(p):
    """Perform sig marker."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def read_result(file, factor):
    """Retrieve or load result for subsequent analysis."""
    d = pd.read_excel(file)
    d["Factor"] = factor
    return d[["Feature", "FDR_P", "Delta_Adjusted_R2", "Factor"]]


part2 = read_result(PART2_FILE, "Nitrogen")
part3 = read_result(PART3_FILE, "Plant density")
part4 = read_result(PART4_FILE, "Variety")

results = pd.concat([part2, part3, part4], ignore_index=True)
results = results[results["Feature"].isin(FEATURE_ORDER)].copy()

results["FDR_P"] = pd.to_numeric(results["FDR_P"], errors="coerce")
results["Delta_Adjusted_R2"] = pd.to_numeric(results["Delta_Adjusted_R2"], errors="coerce")
results["Significant"] = results["FDR_P"] < 0.05

sig_features = results.groupby("Feature")["Significant"].any()

feature_order = [f for f in FEATURE_ORDER if sig_features.get(f, False)]

if not feature_order:
    raise ValueError("No feature reached FDR < 0.05 in the moderation analysis.")

results = results[results["Feature"].isin(feature_order)].copy()
results["Feature"] = pd.Categorical(results["Feature"], categories=feature_order, ordered=True)
results = results.sort_values("Feature")

fig_height = max(4.5, 0.58 * len(feature_order))
fig, ax = plt.subplots(figsize=(8.2, fig_height))

y = np.arange(len(feature_order))
bar_height = 0.20
offsets = [-bar_height, 0, bar_height]

for factor, offset in zip(FACTORS, offsets):

    d = results[results["Factor"] == factor].copy()
    values, pvalues = [], []

    for feature in feature_order:
        row = d[d["Feature"] == feature]
        if row.empty:
            values.append(np.nan)
            pvalues.append(np.nan)
        else:
            values.append(row["Delta_Adjusted_R2"].iloc[0])
            pvalues.append(row["FDR_P"].iloc[0])

    values = np.asarray(values, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)

    for i, (feature, value, p) in enumerate(zip(feature_order, values, pvalues)):

        if np.isnan(value):
            continue

        ax.barh(
            y[i] + offset,
            value,
            height=bar_height * 0.82,
            color=FACTOR_COLORS[factor],
            edgecolor="black",
            linewidth=1.0,
            zorder=3
        )

        marker = sig_marker(p)

        if marker:
            x_range = np.nanmax(np.abs(values))
            x_offset = max(x_range * 0.018, 0.0015)
            x_text = value + x_offset if value >= 0 else value - x_offset

            ax.text(
                x_text,
                y[i] + offset + 0.13,
                marker,
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="black"
            )

ax.axvline(
    0,
    color="black",
    linewidth=0.8,
    zorder=2
)

ax.set_yticks(y)
ax.set_yticklabels(
    [FEATURE_LABELS[f] for f in feature_order],
    fontsize=12
)
ax.invert_yaxis()

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.45,
    alpha=0.3,
    zorder=0
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

ax.tick_params(
    axis="both",
    width=0.6,
    length=4,
    colors="black"
)

ax.tick_params(
    axis="y",
    length=0
)

factor_handles = [
    Patch(
        facecolor=FACTOR_COLORS[f],
        edgecolor="black",
        linewidth=1.0,
        label=f
    )
    for f in FACTORS
]

ax.legend(
    handles=factor_handles,
    title="Agronomic factor",
    loc="lower right",
    frameon=False,
    fontsize=14,
    title_fontsize=14
)

plt.subplots_adjust(
    left=0.24,
    right=0.96,
    top=0.91,
    bottom=0.12
)

plt.savefig(
    FIG_DIR / "Fig2_Moderation_Agronomic_Factors_Combined.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    FIG_DIR / "Fig2_Moderation_Agronomic_Factors_Combined.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Combined moderation figure saved:")
print(FIG_DIR / "Fig_Moderation_Agronomic_Factors_Combined.png")
print(FIG_DIR / "Fig_Moderation_Agronomic_Factors_Combined.pdf")
print(f"Significant features shown: {len(feature_order)}")
print("Features:", feature_order)