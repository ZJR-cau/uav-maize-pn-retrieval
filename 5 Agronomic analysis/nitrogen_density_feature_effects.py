"""Analyze nitrogen and planting-density effects on measured features."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"G:\PhyMaize")
TABLE_FILE = ROOT / r"agronomic_analysis_results\tables\part1_agronomic_effects.xlsx"
FIG_DIR = ROOT / r"agronomic_analysis_results\figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 15
plt.rcParams["axes.unicode_minus"] = False

FEATURE_ORDER = ["I_mean","Green_mean","Green_std","Green_homo","COV","CWSI","Green_corr",
                 "S","CV","I_std","T_energy","T_diss","NIR_cont","Green_cont","T_corr","LAI","T_min","MTCI_std","CIrededge_std","K"]

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

COLORS = {
    "Nitrogen": "#8E8BFE",
    "Plant density": "#FEA3A2"
}

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

def bubble_size(p):
    """Perform bubble size."""
    if pd.isna(p) or p >= 0.05:
        return 32
    if p < 0.001:
        return 230
    if p < 0.01:
        return 180
    return 135

part1 = pd.read_excel(TABLE_FILE)

part1["Factor"] = part1["Factor"].replace({
    "Nitrogen": "Nitrogen",
    "Density": "Plant density"
})

part1 = part1[part1["Feature"].isin(FEATURE_ORDER)].copy()

part1["Feature"] = pd.Categorical(
    part1["Feature"],
    categories=FEATURE_ORDER,
    ordered=True
)

part1["y"] = part1["Feature"].cat.codes

factor_order = ["Nitrogen", "Plant density"]
x_pos = {factor: i for i, factor in enumerate(factor_order)}

fig, ax = plt.subplots(figsize=(5.2, 8.6))

for factor in factor_order:

    d = part1[part1["Factor"] == factor].copy()
    color = COLORS[factor]
    x = x_pos[factor]

    for _, row in d.iterrows():

        y = row["y"]
        p = row["FDR_P"]
        coef = row["Coefficient"]
        sig = p < 0.05 if not pd.isna(p) else False

        ax.scatter(
            x,
            y,
            s=bubble_size(p),
            facecolor=color if sig else "white",
            edgecolor=color,
            linewidth=1.2 if sig else 0.9,
            alpha=1.0 if sig else 0.55,
            zorder=3
        )

        if sig:

            direction = "↑" if coef > 0 else "↓"
            marker = sig_marker(p)

            ax.text(
                x + 0.13,
                y,
                f"{direction} {marker}",
                ha="left",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=color
            )

ax.set_xlim(-0.35, 1.35)
ax.set_ylim(len(FEATURE_ORDER) - 0.6, -0.6)

ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["Nitrogen", "Plant density"],
    fontsize=12,
    fontweight="bold"
)

ax.set_yticks(np.arange(len(FEATURE_ORDER)))
ax.set_yticklabels(
    [FEATURE_LABELS[x] for x in FEATURE_ORDER],
    fontsize=12
)


ax.set_xlabel("Agronomic factor", fontsize=12, fontweight="bold")


ax.axvline(
    0.5,
    color="0.75",
    linewidth=0.8,
    zorder=1
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

plt.subplots_adjust(
    left=0.25,
    right=0.97,
    top=0.91,
    bottom=0.09
)

plt.savefig(
    FIG_DIR / "Fig1_Agronomic_Effects_Bubble_Plot.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    FIG_DIR / "Fig1_Agronomic_Effects_Bubble_Plot.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Figure 1 saved:")
print(FIG_DIR / "Fig1_Agronomic_Effects_Bubble_Plot.png")
print(FIG_DIR / "Fig1_Agronomic_Effects_Bubble_Plot.pdf")