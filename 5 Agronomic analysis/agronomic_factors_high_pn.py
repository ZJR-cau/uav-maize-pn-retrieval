"""Study agronomic drivers of high net photosynthetic rate."""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"G:\PhyMaize")
LABEL_FILE = ROOT / r"annotations\lables.csv"
META_FILE = ROOT / r"meta data\samples_description.xlsx"

# ================================================================
# User-defined parameters
# ================================================================

HIGH_PN_QUANTILE = 0.75
MIN_N = 20

QUANTILE_LABEL = f"Q{int(HIGH_PN_QUANTILE * 100)}"

OUT_DIR = ROOT / r"agronomic_analysis_results\high_pn_factor_comparison" / QUANTILE_LABEL
OUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "Sample_ID"
PN_COL = "Pn"

STAGES = ["V6", "V9", "V12"]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 15
plt.rcParams["axes.unicode_minus"] = False


# ================================================================
# Functions
# ================================================================

def norm_id(x):
    """Perform norm id."""
    x = str(x).strip()
    if x.startswith("PM_"):
        return x
    try:
        return f"PM_{int(float(x)):04d}"
    except:
        return x


def numeric_extract(s):
    """Perform numeric extract."""
    return pd.to_numeric(
        s.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce"
    )


def clean_axis(ax):
    """Perform clean axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, colors="black")
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)


# ================================================================
# Check parameters
# ================================================================

if not 0 < HIGH_PN_QUANTILE < 1:
    raise ValueError("HIGH_PN_QUANTILE must be between 0 and 1.")

print("\n================ CONFIGURATION ================\n")
print(f"High-Pn quantile : {HIGH_PN_QUANTILE:.2f} ({QUANTILE_LABEL})")
print(f"Minimum group n  : {MIN_N}")
print(f"Stages           : {STAGES}")
print(f"Output directory : {OUT_DIR}")


# ================================================================
# Load data
# ================================================================

print("\n================ LOAD DATA ================\n")

labels = pd.read_csv(LABEL_FILE)
meta = pd.read_excel(META_FILE)

labels[ID_COL] = labels[ID_COL].map(norm_id)
meta[ID_COL] = meta[ID_COL].map(norm_id)

if PN_COL not in labels.columns:
    raise ValueError(f"Missing label column: {PN_COL}")

required_meta = [
    ID_COL,
    "Year",
    "Site",
    "Stage",
    "Variety",
    "Plant_Density",
    "Nitrogen"
]

missing = [c for c in required_meta if c not in meta.columns]

if missing:
    raise ValueError(f"Missing metadata columns: {missing}")

data = labels[[ID_COL, PN_COL]].merge(
    meta[required_meta],
    on=ID_COL,
    how="inner"
)

data[PN_COL] = pd.to_numeric(data[PN_COL], errors="coerce")
data["N"] = numeric_extract(data["Nitrogen"])
data["Density"] = numeric_extract(data["Plant_Density"])
data["Variety"] = data["Variety"].astype(str).str.strip()
data["Stage"] = data["Stage"].astype(str).str.strip()

data["Environment"] = (
    data["Year"].astype("Int64").astype(str) + "_"
    + data["Site"].astype(str).str.strip() + "_"
    + data["Stage"]
)

data = data.dropna(
    subset=[PN_COL, "Environment"]
).copy()

data = data[data["Stage"].isin(STAGES)].copy()

print(f"Matched samples: {len(data)}")
print(f"Environments: {data['Environment'].nunique()}")
print(f"Stages: {data['Stage'].unique().tolist()}")


# ================================================================
# Define High-Pn
# Environment-specific dynamic quantile
# ================================================================

print("\n================ DEFINE HIGH-Pn ================\n")

quantile_col = f"Environment_Pn_Q{int(HIGH_PN_QUANTILE * 100)}"

data[quantile_col] = data.groupby(
    "Environment"
)[PN_COL].transform(
    "quantile",
    HIGH_PN_QUANTILE
)

data["High_Pn"] = (
    data[PN_COL] >= data[quantile_col]
).astype(int)

print(
    f"High-Pn threshold: "
    f"{HIGH_PN_QUANTILE:.2f} quantile "
    f"({QUANTILE_LABEL}) within each Year × Site × Stage environment"
)

print(f"High-Pn samples: {data['High_Pn'].sum()}")
print(f"High-Pn proportion: {data['High_Pn'].mean() * 100:.2f}%")


# ================================================================
# Environment summary
# ================================================================

environment_summary = (
    data.groupby("Environment")
    .agg(
        Sample_N=(ID_COL, "size"),
        Pn_Threshold=(PN_COL, lambda x: x.quantile(HIGH_PN_QUANTILE)),
        HighPn_N=("High_Pn", "sum"),
        HighPn_Rate=("High_Pn", "mean")
    )
    .reset_index()
)

environment_summary["HighPn_Rate"] *= 100

environment_summary.to_excel(
    OUT_DIR / f"environment_high_pn_definition_{QUANTILE_LABEL}.xlsx",
    index=False
)


# ================================================================
# Factor summaries within each growth stage
# ================================================================

factor_info = {
    "N": "Nitrogen",
    "Density": "Plant density",
    "Variety": "Variety"
}

summaries = {}

for stage in STAGES:

    stage_data = data[data["Stage"] == stage].copy()

    print(f"\n================ {stage} ================\n")

    for factor, label in factor_info.items():

        d = stage_data.dropna(subset=[factor]).copy()

        summary = (
            d.groupby(factor)
            .agg(
                Sample_N=(ID_COL, "size"),
                HighPn_N=("High_Pn", "sum"),
                HighPn_Rate=("High_Pn", "mean")
            )
            .reset_index()
        )

        before_n = len(summary)

        summary = summary[
            summary["Sample_N"] >= MIN_N
        ].copy()

        removed = before_n - len(summary)

        summary["HighPn_Rate"] *= 100

        summaries[(stage, factor)] = summary

        summary.to_excel(
            OUT_DIR / f"high_pn_by_{stage}_{factor}_{QUANTILE_LABEL}.xlsx",
            index=False
        )

        print(f"{stage} - {factor}")
        print(f"Removed levels with n < {MIN_N}: {removed}")
        print(summary.to_string(index=False))
        print()


# ================================================================
# Combined summary table
# ================================================================

combined_summary = []

for (stage, factor), summary in summaries.items():

    temp = summary.copy()
    temp.insert(0, "Stage", stage)
    temp.insert(1, "Factor", factor)

    combined_summary.append(temp)

combined_summary = pd.concat(
    combined_summary,
    ignore_index=True
)

combined_summary.to_excel(
    OUT_DIR / f"high_pn_by_stage_and_factor_{QUANTILE_LABEL}.xlsx",
    index=False
)


# ================================================================
# Figure
# ================================================================

print("\n================ PLOT FIGURE ================\n")

colors = {
    "N": "#8E8BFE",
    "Density": "#FEA3A2",
    "Variety": "#86D5B8"
}

factor_titles = {
    "N": "Nitrogen",
    "Density": "Plant density",
    "Variety": "Variety"
}

fig, axes = plt.subplots(
    3,
    3,
    figsize=(12, 10),
    sharey=True
)

for row, stage in enumerate(STAGES):

    for col, factor in enumerate(factor_info):

        ax = axes[row, col]
        summary = summaries[(stage, factor)].copy()

        # --------------------------------------------------------
        # Ordering
        # --------------------------------------------------------

        if factor == "N":
            summary = summary.sort_values("N")

        elif factor == "Density":
            summary = summary.sort_values("Density")

        elif factor == "Variety":
            summary[factor] = summary[factor].astype(str)

        x = np.arange(len(summary))

        # --------------------------------------------------------
        # Bars
        # --------------------------------------------------------

        ax.bar(
            x,
            summary["HighPn_Rate"],
            width=0.62,
            color=colors[factor],
            edgecolor="black",
            linewidth=0.7
        )

        # --------------------------------------------------------
        # X-axis
        # --------------------------------------------------------

        ax.set_xticks(x)

        ax.set_xticklabels(
            [str(v) for v in summary[factor]],
            fontsize=11,
            rotation=30,
            ha="right"
        )

        ax.set_xlabel(
            factor_titles[factor],
            fontsize=13,
            fontweight="bold"
        )

        # --------------------------------------------------------
        # Y-axis
        # --------------------------------------------------------

        if col == 0:
            ax.set_ylabel(
                "High-Pn samples (%)",
                fontsize=13,
                fontweight="bold"
            )
        else:
            ax.set_ylabel("")


        # --------------------------------------------------------
        # Axis
        # --------------------------------------------------------

        ax.set_ylim(0, 55)
        ax.set_yticks(np.arange(0, 56, 10))

        clean_axis(ax)


# ================================================================
# Save figure
# ================================================================

fig.suptitle(
    f"High-Pn proportion across agronomic factors within growth stages ({QUANTILE_LABEL})",
    fontsize=16,
    fontweight="bold",
    y=1.01
)

plt.tight_layout(
    pad=1.4,
    h_pad=1.8,
    w_pad=1.2
)

fig.savefig(
    OUT_DIR / f"HighPn_Proportion_By_Stage_And_Agronomic_Factors_{QUANTILE_LABEL}.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT_DIR / f"HighPn_Proportion_By_Stage_And_Agronomic_Factors_{QUANTILE_LABEL}.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("\n================ DONE ================\n")
print(f"Output directory: {OUT_DIR}")
print(
    f"Figure: HighPn_Proportion_By_Stage_And_Agronomic_Factors_{QUANTILE_LABEL}.png"
)
print(
    f"Figure: HighPn_Proportion_By_Stage_And_Agronomic_Factors_{QUANTILE_LABEL}.pdf"
)