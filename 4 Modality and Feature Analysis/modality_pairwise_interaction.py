"""Visualize pairwise interactions and information gains among modalities."""
# ================================================================
# novel_modality_visualization.py
# Novel Modality Contribution and Interaction Visualization
# ================================================================
import warnings
warnings.filterwarnings("ignore")
import math
from pathlib import Path
from itertools import combinations, permutations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

DATA_ROOT = Path(r"G:\PhyMaize")
INPUT_FILE = DATA_ROOT / "traditional_ml_results" / "20_features" / "feature_group_ablation_v2" / "shapley_modality_contribution" / "modality_combination_performance.xlsx"
ABLATION_FILE = DATA_ROOT / "traditional_ml_results" / "20_features" / "feature_group_ablation_v2" / "feature_group_ablation.xlsx"
OUTPUT_ROOT = DATA_ROOT / "traditional_ml_results" / "20_features" / "feature_group_ablation_v2" / "shapley_modality_contribution"
DPI = 600
RANDOM_STATE = 42
MODALITIES = ["M","T","L"]
MODALITY_NAMES = {"M":"MS","T":"TIR","L":"LiDAR"}
MODALITY_COLORS = {"MS":"#4C78A8","TIR":"#F58518","LiDAR":"#54A24B"}
GROUP_COLORS = ["#4C78A8","#F58518","#54A24B","#B279A2","#72B7B2","#E45756","#9D755D"]

plt.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial"],"axes.unicode_minus":False,"font.size":10,"axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,"xtick.major.size":4,"ytick.major.size":4,"figure.dpi":150,"savefig.dpi":DPI,"savefig.bbox":"tight","savefig.pad_inches":0.08})


def load_value_function():
    """Retrieve or load value function for subsequent analysis."""
    if ABLATION_FILE.exists():
        df = pd.read_excel(ABLATION_FILE)
    elif INPUT_FILE.exists():
        df = pd.read_excel(INPUT_FILE)
    else:
        raise FileNotFoundError(f"Input file not found:\n{ABLATION_FILE}\n{INPUT_FILE}")
    required = ["Feature_Group","Test_R2"]
    missing = [x for x in required if x not in df.columns]
    if missing: raise ValueError(f"Missing columns in input file: {missing}")
    mapping = {}
    for group in ["MS","TIR","LiDAR","MS+TIR","MS+LiDAR","TIR+LiDAR","MS+TIR+LiDAR"]:
        rows = df[df["Feature_Group"].astype(str) == group]
        if rows.empty: raise ValueError(f"Missing required feature group: {group}")
        mapping[group] = float(rows.iloc[0]["Test_R2"])
    return {frozenset():0.0,frozenset(["M"]):mapping["MS"],frozenset(["T"]):mapping["TIR"],frozenset(["L"]):mapping["LiDAR"],frozenset(["M","T"]):mapping["MS+TIR"],frozenset(["M","L"]):mapping["MS+LiDAR"],frozenset(["T","L"]):mapping["TIR+LiDAR"],frozenset(["M","T","L"]):mapping["MS+TIR+LiDAR"]}


def load_complexity_data():
    """Retrieve or load complexity data for subsequent analysis."""
    if not ABLATION_FILE.exists(): return None
    df = pd.read_excel(ABLATION_FILE)
    candidates = ["Model_Size_MB","Model Size (MB)","Model_Size","Model size (MB)","Model_Size_MB"]
    column = next((x for x in candidates if x in df.columns),None)
    if column is None: return None
    mapping = {}
    for group in ["MS","TIR","LiDAR","MS+TIR","MS+LiDAR","TIR+LiDAR","MS+TIR+LiDAR"]:
        rows = df[df["Feature_Group"].astype(str) == group]
        if not rows.empty: mapping[group] = float(rows.iloc[0][column])
    return mapping if len(mapping) == 7 else None


def shapley_values(value_function):
    """Perform shapley values."""
    n = len(MODALITIES)
    results = {}
    for modality in MODALITIES:
        others = [x for x in MODALITIES if x != modality]
        contribution = 0.0
        for r in range(n):
            for subset_tuple in combinations(others,r):
                subset = frozenset(subset_tuple)
                subset_with = subset | frozenset([modality])
                weight = math.factorial(r) * math.factorial(n-r-1) / math.factorial(n)
                contribution += weight * (value_function[subset_with] - value_function[subset])
        results[modality] = contribution
    return results


def style_axis(ax):
    """Perform style axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(axis="y",linestyle="--",linewidth=1.0,alpha=0.20)
    ax.set_axisbelow(True)


def save_figure(fig,filename):
    """Generate or save figure."""
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT_ROOT / f"{filename}.png",dpi=DPI,facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{filename}.pdf",dpi=DPI,facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{filename}.svg",dpi=DPI,facecolor="white")
    plt.close(fig)


def pairwise_information_gain(value_function):
    """Perform pairwise information gain."""
    pairs = list(combinations(MODALITIES,2))
    records = []
    for a,b in pairs:
        combined = value_function[frozenset([a,b])]
        best_single = max(value_function[frozenset([a])],value_function[frozenset([b])])
        gain = combined - best_single
        redundancy = value_function[frozenset([a])] + value_function[frozenset([b])] - combined
        records.append({"Pair":f"{MODALITY_NAMES[a]} + {MODALITY_NAMES[b]}","Information_Gain":gain,"Redundancy_Index":redundancy,"Combined_R2":combined})
    return pd.DataFrame(records)


def plot_modality_interaction_network(value_function,shapley):
    """Generate or save modality interaction network."""
    fig,ax = plt.subplots(figsize=(8.0,7.0))
    positions = {"MS":(-1.25,0.0),"TIR":(1.25,0.0),"LiDAR":(0.0,1.65)}
    for a,b in combinations(MODALITIES,2):
        na,nb = MODALITY_NAMES[a],MODALITY_NAMES[b]
        delta_ab = value_function[frozenset([a,b])] - value_function[frozenset([a])]
        delta_ba = value_function[frozenset([a,b])] - value_function[frozenset([b])]
        edge_value = (delta_ab + delta_ba) / 2
        x1,y1 = positions[na]; x2,y2 = positions[nb]
        width = 1.2 + 8.0 * max(abs(edge_value),0.005)
        color = "#C44E52" if edge_value >= 0 else "#4C78A8"
        ax.plot([x1,x2],[y1,y2],color=color,linewidth=width,alpha=0.60,zorder=1)
        mx,my=(x1+x2)/2,(y1+y2)/2
        ax.text(mx,my+0.12,f"ΔR² = {edge_value:+.3f}",ha="center",va="center",fontsize=10,fontweight="bold",bbox={"boxstyle":"round,pad=0.22","fc":"white","ec":"none","alpha":0.88},zorder=4)
    max_phi = max(shapley.values())
    for modality in MODALITIES:
        name = MODALITY_NAMES[modality]
        x,y = positions[name]
        radius = 0.24 + 0.34 * shapley[modality] / max_phi
        circle = Circle((x,y),radius=radius,facecolor=MODALITY_COLORS[name],edgecolor="white",linewidth=3,alpha=0.94,zorder=5)
        ax.add_patch(circle)
        ax.text(x,y+0.02,name,ha="center",va="center",fontsize=13,fontweight="bold",color="white",zorder=6)
        ax.text(x,y-radius-0.17,f"Shapley = {shapley[modality]:.3f}",ha="center",va="top",fontsize=10,fontweight="bold",zorder=6)
    full_r2 = value_function[frozenset(MODALITIES)]
    ax.text(0,-1.10,f"All-source R² = {full_r2:.3f}",ha="center",fontsize=15,fontweight="bold")
    ax.text(0,-1.42,"Node size represents Shapley contribution; edge width represents pairwise complementarity",ha="center",fontsize=9.5)
    ax.set_xlim(-2.15,2.15); ax.set_ylim(-1.70,2.20); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Modality Interaction Network",fontsize=15,fontweight="bold",pad=15)
    fig.tight_layout()
    save_figure(fig,"Fig_Modality_Interaction_Network")


def plot_information_gain_matrix(value_function):
    """Generate or save information gain matrix."""
    modalities = ["MS","TIR","LiDAR"]
    code = {"MS":"M","TIR":"T","LiDAR":"L"}
    matrix = np.zeros((3,3))
    for i,a in enumerate(modalities):
        for j,b in enumerate(modalities):
            if i != j:
                combined = value_function[frozenset([code[a],code[b]])]
                matrix[i,j] = combined - max(value_function[frozenset([code[a]])],value_function[frozenset([code[b]])])
    vmax = max(abs(matrix.min()),abs(matrix.max()),0.001)
    cmap = LinearSegmentedColormap.from_list("information_gain",["#D9EAF7","#FFFFFF","#F4B6B2"],N=256)
    norm = TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    fig,ax = plt.subplots(figsize=(6.7,5.9))
    image = ax.imshow(matrix,cmap=cmap,norm=norm,aspect="equal")
    ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(3)); ax.set_xticklabels(modalities); ax.set_yticklabels(modalities)
    ax.set_xlabel("Compared modality"); ax.set_ylabel("Modality pair")
    for i in range(3):
        for j in range(3):
            text = "—" if i == j else f"{matrix[i,j]:+.3f}"
            ax.text(j,i,text,ha="center",va="center",fontsize=14,fontweight="bold",color="#222222")
    cbar = fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04); cbar.set_label("Pairwise information gain ΔR²")
    ax.set_title("Pairwise Information Gain",fontsize=15,fontweight="bold",pad=12)
    fig.tight_layout()
    save_figure(fig,"Fig_Pairwise_Information_Gain")


def plot_shapley_complementarity(value_function,shapley):
    """Generate or save shapley complementarity."""
    pair_values = {}
    for a,b in combinations(MODALITIES,2):
        combined = value_function[frozenset([a,b])]
        gain = combined - max(value_function[frozenset([a])],value_function[frozenset([b])])
        pair_values[a] = pair_values.get(a,[])+[gain]
        pair_values[b] = pair_values.get(b,[])+[gain]
    complementarity = {m:np.mean(pair_values[m]) for m in MODALITIES}
    fig,ax = plt.subplots(figsize=(7.2,5.8))
    for m in MODALITIES:
        name = MODALITY_NAMES[m]
        ax.scatter(complementarity[m],shapley[m],s=450,alpha=0.88,color=MODALITY_COLORS[name],edgecolor="white",linewidth=2.0,zorder=3)
        ax.annotate(name,(complementarity[m],shapley[m]),xytext=(9,8),textcoords="offset points",fontsize=12,fontweight="bold")
    x_values=np.array(list(complementarity.values())); y_values=np.array(list(shapley.values()))
    ax.axvline(np.mean(x_values),linestyle="--",linewidth=1.0,alpha=0.55,color="#777777")
    ax.axhline(np.mean(y_values),linestyle="--",linewidth=1.0,alpha=0.55,color="#777777")
    ax.set_xlabel("Mean pairwise information gain ΔR²"); ax.set_ylabel("Shapley contribution to R²")
    ax.set_title("Shapley Contribution vs. Information Complementarity",fontsize=14,fontweight="bold")
    ax.text(0.02,0.97,"Higher contribution",transform=ax.transAxes,ha="left",va="top",fontsize=9.5)
    ax.text(0.98,0.03,"Higher complementarity",transform=ax.transAxes,ha="right",va="bottom",fontsize=9.5)
    style_axis(ax); fig.tight_layout()
    save_figure(fig,"Fig_Shapley_vs_Complementarity")


def plot_performance_complexity(value_function,complexity):
    """Generate or save performance complexity."""
    if complexity is None:
        print("\n[Skip] Performance–Complexity plot: Model size column not found in feature_group_ablation.xlsx.")
        return
    names = ["MS","TIR","LiDAR","MS + TIR","MS + LiDAR","TIR + LiDAR","MS + TIR + LiDAR"]
    codes = [frozenset(["M"]),frozenset(["T"]),frozenset(["L"]),frozenset(["M","T"]),frozenset(["M","L"]),frozenset(["T","L"]),frozenset(["M","T","L"])]
    groups = ["MS","TIR","LiDAR","MS+TIR","MS+LiDAR","TIR+LiDAR","MS+TIR+LiDAR"]
    x = np.array([complexity[g] for g in groups]); y = np.array([value_function[c] for c in codes])
    fig,ax = plt.subplots(figsize=(8.0,5.8))
    colors = [GROUP_COLORS[i] for i in range(7)]
    for i,(xi,yi,name) in enumerate(zip(x,y,names)):
        ax.scatter(xi,yi,s=180 if i<3 else 230,color=colors[i],edgecolor="white",linewidth=1.8,zorder=3)
        ax.annotate(name,(xi,yi),xytext=(7,7),textcoords="offset points",fontsize=9.5,fontweight="bold")
    ax.set_xlabel("Model size (MB)"); ax.set_ylabel("Test R²"); ax.set_title("Performance–Complexity Trade-off",fontsize=15,fontweight="bold",pad=12)
    style_axis(ax); fig.tight_layout()
    save_figure(fig,"Fig_Performance_Complexity_Tradeoff")


def plot_modality_addition_trajectories(value_function):
    """Generate or save modality addition trajectories."""
    all_paths = list(permutations(MODALITIES))
    fig,ax = plt.subplots(figsize=(8.5,5.8))
    x = np.arange(4)
    for path_index,path in enumerate(all_paths):
        subsets = [frozenset()]
        current = frozenset()
        for modality in path:
            current = current | frozenset([modality])
            subsets.append(current)
        y = [value_function[s] for s in subsets]
        path_name = " → ".join(MODALITY_NAMES[m] for m in path)
        ax.plot(x,y,marker="o",linewidth=1.5,alpha=0.35,color="#777777")
        ax.text(3.04,y[-1],path_name,fontsize=8.5,va="center",alpha=0.80)
    mean_y = np.mean([[value_function[frozenset()]]+[value_function[frozenset([m])] for m in p[:1]]+[value_function[frozenset(p[:2])],value_function[frozenset(p)]] for p in all_paths],axis=0)
    ax.plot(x,mean_y,marker="o",linewidth=3.0,color="#222222",label="Mean across six addition orders",zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(["None","1 modality","2 modalities","3 modalities"])
    ax.set_ylabel("Test R²"); ax.set_xlabel("Progressive modality addition")
    ax.set_title("Modality Addition Trajectories",fontsize=15,fontweight="bold",pad=12)
    ax.legend(frameon=False,loc="upper left")
    style_axis(ax); fig.tight_layout()
    save_figure(fig,"Fig_Modality_Addition_Trajectories")


def save_derived_tables(value_function,shapley):
    """Generate or save derived tables."""
    pair_df = pairwise_information_gain(value_function)
    pair_df.to_excel(OUTPUT_ROOT / "pairwise_information_gain.xlsx",index=False)
    shapley_df = pd.DataFrame({"Modality":[MODALITY_NAMES[m] for m in MODALITIES],"Shapley_Contribution_R2":[shapley[m] for m in MODALITIES],"Mean_Pairwise_Information_Gain_R2":[pair_df.iloc[i]["Information_Gain"] if i < 3 else np.nan for i in range(3)]})
    shapley_df.to_excel(OUTPUT_ROOT / "shapley_complementarity_summary.xlsx",index=False)


def main():
    """Run the main analysis workflow."""
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    print("="*70)
    print("Novel Modality Contribution Visualization")
    print("="*70)
    value_function = load_value_function()
    shapley = shapley_values(value_function)
    complexity = load_complexity_data()
    save_derived_tables(value_function,shapley)
    plot_modality_interaction_network(value_function,shapley)
    plot_information_gain_matrix(value_function)
    plot_shapley_complementarity(value_function,shapley)
    plot_performance_complexity(value_function,complexity)
    plot_modality_addition_trajectories(value_function)
    print("\nFigures and tables saved to:")
    print(OUTPUT_ROOT)
    print("="*70)


if __name__ == "__main__":
    main()
