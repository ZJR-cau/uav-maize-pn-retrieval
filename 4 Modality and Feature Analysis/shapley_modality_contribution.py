"""Estimate modality contributions with Shapley-style attribution."""
# ================================================================
# shapley_modality_contribution.py
# Shapley-style Modality Contribution Analysis
# ================================================================
import warnings
warnings.filterwarnings("ignore")
import math
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
DATA_ROOT = Path(r"G:\PhyMaize")
INPUT_FILE = DATA_ROOT / "traditional_ml_results" / "20_features" / "feature_group_ablation_v2" / "feature_group_ablation.xlsx"
OUTPUT_ROOT = DATA_ROOT / "traditional_ml_results" / "20_features" / "feature_group_ablation_v2" / "shapley_modality_contribution"
DPI = 600
RANDOM_STATE = 42
MODALITY_COLORS = {"MS": "#4C78A8", "TIR": "#F58518", "LiDAR": "#54A24B"}
GROUP_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2", "#E45756", "#9D755D"]
plt.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial"],"axes.unicode_minus":False,"font.size":10,"axes.titlesize":12,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,"xtick.major.size":4,"ytick.major.size":4,"figure.dpi":150,"savefig.dpi":DPI,"savefig.bbox":"tight","savefig.pad_inches":0.08})
MODALITY_NAMES = {"M":"MS","T":"TIR","L":"LiDAR"}
COMBINATION_NAMES = {"M":"MS","T":"TIR","L":"LiDAR","MT":"MS + TIR","ML":"MS + LiDAR","TL":"TIR + LiDAR","MTL":"MS + TIR + LiDAR"}


def load_results():
    """Retrieve or load results for subsequent analysis."""
    if not INPUT_FILE.exists(): raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    required_columns = ["Feature_Group","Feature_Number","Test_R2","Test_RMSE","Test_rRMSE","Test_MAE"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing: raise ValueError(f"Missing columns in input file: {missing}")
    return df


def get_r2_mapping(df):
    """Retrieve or load r2 mapping for subsequent analysis."""
    mapping = {}
    required_groups = ["MS","TIR","LiDAR","MS+TIR","MS+LiDAR","TIR+LiDAR","MS+TIR+LiDAR"]
    for group in required_groups:
        rows = df[df["Feature_Group"].astype(str) == group]
        if rows.empty: raise ValueError(f"Missing required feature group: {group}")
        mapping[group] = float(rows.iloc[0]["Test_R2"])
    return {frozenset():0.0,frozenset(["M"]):mapping["MS"],frozenset(["T"]):mapping["TIR"],frozenset(["L"]):mapping["LiDAR"],frozenset(["M","T"]):mapping["MS+TIR"],frozenset(["M","L"]):mapping["MS+LiDAR"],frozenset(["T","L"]):mapping["TIR+LiDAR"],frozenset(["M","T","L"]):mapping["MS+TIR+LiDAR"]}


def shapley_values(value_function):
    """Perform shapley values."""
    modalities = ["M","T","L"]
    n = len(modalities)
    results = {}
    for modality in modalities:
        others = [x for x in modalities if x != modality]
        contribution = 0.0
        for r in range(n):
            for subset_tuple in combinations(others,r):
                subset = frozenset(subset_tuple)
                subset_with_modality = subset | frozenset([modality])
                weight = math.factorial(r) * math.factorial(n-r-1) / math.factorial(n)
                contribution += weight * (value_function[subset_with_modality] - value_function[subset])
        results[modality] = contribution
    return results


def build_shapley_table(value_function, shapley):
    """Perform build shapley table."""
    full_value = value_function[frozenset(["M","T","L"])]
    records = [{"Modality":MODALITY_NAMES[m],"Shapley_Contribution_R2":shapley[m],"Contribution_Share_Percent":shapley[m]/full_value*100} for m in ["M","T","L"]]
    df = pd.DataFrame(records)
    df.loc[len(df)] = {"Modality":"Total","Shapley_Contribution_R2":df["Shapley_Contribution_R2"].sum(),"Contribution_Share_Percent":df["Contribution_Share_Percent"].sum()}
    return df


def build_marginal_contribution_table(value_function):
    """Perform build marginal contribution table."""
    records = []
    modalities = ["M","T","L"]
    for modality in modalities:
        others = [x for x in modalities if x != modality]
        for r in range(3):
            for subset_tuple in combinations(others,r):
                subset = frozenset(subset_tuple)
                subset_with = subset | frozenset([modality])
                base_name = "None" if len(subset) == 0 else " + ".join(MODALITY_NAMES[x] for x in modalities if x in subset)
                combined_name = MODALITY_NAMES[modality] if len(subset_with) == 1 else " + ".join(MODALITY_NAMES[x] for x in modalities if x in subset_with)
                records.append({"Added_Modality":MODALITY_NAMES[modality],"Base_Set":base_name,"Combined_Set":combined_name,"Base_R2":value_function[subset],"Combined_R2":value_function[subset_with],"Marginal_Delta_R2":value_function[subset_with]-value_function[subset]})
    return pd.DataFrame(records)


def build_combination_table(value_function):
    """Perform build combination table."""
    ordered_sets = [frozenset(["M"]),frozenset(["T"]),frozenset(["L"]),frozenset(["M","T"]),frozenset(["M","L"]),frozenset(["T","L"]),frozenset(["M","T","L"])]
    records = []
    for subset in ordered_sets:
        key = "".join(x for x in ["M","T","L"] if x in subset)
        records.append({"Combination_Code":key,"Combination":COMBINATION_NAMES[key],"Number_of_Modalities":len(subset),"Test_R2":value_function[subset]})
    return pd.DataFrame(records)


def save_tables(shapley_df,marginal_df,combination_df):
    """Generate or save tables."""
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    shapley_df.to_excel(OUTPUT_ROOT / "shapley_modality_contribution.xlsx",index=False)
    marginal_df.to_excel(OUTPUT_ROOT / "marginal_contribution_all_subsets.xlsx",index=False)
    combination_df.to_excel(OUTPUT_ROOT / "modality_combination_performance.xlsx",index=False)


def save_figure(fig,filename):
    """Generate or save figure."""
    fig.savefig(OUTPUT_ROOT / f"{filename}.png",dpi=DPI,facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{filename}.pdf",dpi=DPI,facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{filename}.svg",dpi=DPI,facecolor="white")
    plt.close(fig)


def style_axis(ax):
    """Perform style axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(axis="y",linestyle="--",linewidth=1.2,alpha=0.22)
    ax.set_axisbelow(True)


def plot_shapley_contribution(shapley_df):
    """Generate or save shapley contribution."""
    plot_df = shapley_df[shapley_df["Modality"] != "Total"].copy()
    x = np.arange(len(plot_df))
    values = plot_df["Shapley_Contribution_R2"].values
    colors = [MODALITY_COLORS[m] for m in plot_df["Modality"]]
    fig,ax = plt.subplots(figsize=(7.0,5.0))
    bars = ax.bar(x,values,width=0.58,color=colors,alpha=0.88,edgecolor="white",linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Modality"])
    ax.set_ylabel("Shapley contribution to R²")
    ax.set_xlabel("Remote-sensing modality")
    ax.set_ylim(0,max(values)*1.28)
    style_axis(ax)
    for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*0.025,f"{value:.3f}",ha="center",va="bottom",fontsize=15,fontweight="bold")
    full_r2 = shapley_df.loc[shapley_df["Modality"]=="Total","Shapley_Contribution_R2"].iloc[0]
    ax.text(0.98,0.96,f"All-source R² = {full_r2:.3f}",transform=ax.transAxes,ha="right",va="top",fontsize=15,fontweight="bold")
    fig.tight_layout()
    save_figure(fig,"Fig_Shapley_Modality_Contribution")


def plot_contribution_share(shapley_df):
    """Generate or save contribution share."""
    plot_df = shapley_df[shapley_df["Modality"] != "Total"].copy()
    values = plot_df["Contribution_Share_Percent"].values
    labels = plot_df["Modality"].values
    colors = [MODALITY_COLORS[x] for x in labels]
    full_r2 = shapley_df.loc[shapley_df["Modality"]=="Total","Shapley_Contribution_R2"].iloc[0]
    fig,ax = plt.subplots(figsize=(7.0,5.8))
    wedges,_ = ax.pie(values,startangle=90,counterclock=False,colors=colors,wedgeprops={"width":0.38,"edgecolor":"white","linewidth":2.0,"alpha":0.9})
    for wedge,label,value in zip(wedges,labels,values):
        angle = (wedge.theta2+wedge.theta1)/2
        x,y = np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))
        ax.annotate(f"{label}\n{value:.1f}%",xy=(x*0.82,y*0.82),xytext=(x*1.22,y*1.22),ha="center",va="center",fontsize=10,fontweight="bold",arrowprops={"arrowstyle":"-","linewidth":0.8,"color":"#555555"},bbox={"boxstyle":"round,pad=0.22","fc":"white","ec":"none","alpha":0.82})
    ax.text(0,0.07,"All-source",ha="center",va="center",fontsize=10)
    ax.text(0,-0.10,"R²",ha="center",va="center",fontsize=10)
    ax.text(0,-0.27,f"{full_r2:.3f}",ha="center",va="center",fontsize=16,fontweight="bold")
    ax.legend([Patch(facecolor=MODALITY_COLORS[x],edgecolor="none",alpha=0.9) for x in labels],labels,loc="lower center",bbox_to_anchor=(0.5,-0.08),ncol=3,frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()
    save_figure(fig,"Fig_Shapley_Contribution_Share")


def plot_combination_performance(combination_df):
    """Generate or save combination performance."""
    groups = [combination_df[combination_df["Number_of_Modalities"]==1].copy(),combination_df[combination_df["Number_of_Modalities"]==2].copy(),combination_df[combination_df["Number_of_Modalities"]==3].copy()]
    group_labels = ["Single-source","Two-source","Three-source"]
    x_positions,x_labels,values,colors = [],[],[],[]
    position = 0
    for group_index,group in enumerate(groups):
        for _,row in group.iterrows():
            x_positions.append(position); x_labels.append(row["Combination"]); values.append(row["Test_R2"])
            colors.append(MODALITY_COLORS[row["Combination"]] if row["Combination"] in MODALITY_COLORS else GROUP_COLORS[len(x_positions)-1])
            position += 1
        position += 0.6
    colors = [MODALITY_COLORS.get(c.split(" + ")[0],GROUP_COLORS[i]) if "+" not in c else GROUP_COLORS[i] for i,c in enumerate(x_labels)]
    fig,ax = plt.subplots(figsize=(9.0,5.6))
    bars = ax.bar(x_positions,values,width=0.6,color=colors,alpha=0.86,edgecolor="black",linewidth=1.2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Test R²")
    ax.set_ylim(0.40,0.64)
    style_axis(ax)
    for bar,value in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,value+0.006,f"{value:.3f}",ha="center",va="bottom",fontsize=12,fontweight="bold")
    boundaries = [len(groups[0])-0.5,len(groups[0])+len(groups[1])+0.1]
    for boundary in boundaries: ax.axvline(boundary,linestyle=":",linewidth=1.2,alpha=0.5,color="#777777")
    spans = [(0,len(groups[0])),(len(groups[0]),len(groups[0])+len(groups[1])),(len(groups[0])+len(groups[1]),len(x_positions))]
    for label,(start,end) in zip(group_labels,spans): ax.text(np.mean(x_positions[start:end]),0.625,label,ha="center",fontsize=12,fontweight="bold")
    fig.tight_layout()
    save_figure(fig,"Fig_Modality_Combination_Performance")


def plot_synergy_heatmap(value_function):
    """Generate or save synergy heatmap."""
    modalities = ["MS","TIR","LiDAR"]
    code_map = {"MS":"M","TIR":"T","LiDAR":"L"}
    matrix = np.zeros((3,3))
    for i,modality_i in enumerate(modalities):
        for j,modality_j in enumerate(modalities):
            if i != j:
                a,b = code_map[modality_i],code_map[modality_j]
                matrix[i,j] = value_function[frozenset([a,b])] - value_function[frozenset([a])]
    vmax = max(abs(matrix.min()),abs(matrix.max()),0.001)
    cmap = LinearSegmentedColormap.from_list("synergy",["#DCEAF7","#FFFFFF","#F4B6B2"],N=256)
    norm = TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    fig,ax = plt.subplots(figsize=(6.8,5.7))
    image = ax.imshow(matrix,aspect="equal",cmap=cmap,norm=norm)
    ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(3))
    ax.set_xticklabels(modalities); ax.set_yticklabels(modalities)
    ax.set_xlabel("Added modality"); ax.set_ylabel("Base modality")
    for i in range(3):
        for j in range(3):
            text = "—" if i == j else f"{matrix[i,j]:+.3f}"
            ax.text(j,i,text,ha="center",va="center",fontsize=11,fontweight="bold",color="#222222")
    cbar = fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    cbar.set_label("Incremental ΔR²")
    ax.set_title("Pairwise information complementarity")
    fig.tight_layout()
    save_figure(fig,"Fig_Modality_Pairwise_Synergy")


def plot_shapley_waterfall(shapley_df,value_function):
    """Generate or save shapley waterfall."""
    plot_df = shapley_df[shapley_df["Modality"]!="Total"].copy()
    values = plot_df["Shapley_Contribution_R2"].values
    cumulative = np.cumsum(values)
    x = np.arange(len(values))
    bottoms = np.concatenate([[0],cumulative[:-1]])
    colors = [MODALITY_COLORS[m] for m in plot_df["Modality"]]
    full_r2 = value_function[frozenset(["M","T","L"])]
    fig,ax = plt.subplots(figsize=(8.0,5.5))
    bars = ax.bar(x,values,bottom=bottoms,width=0.58,color=colors,alpha=0.88,edgecolor="white",linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(plot_df["Modality"])
    ax.set_ylabel("Cumulative contribution to R²")
    ax.set_ylim(0,full_r2*1.15)
    style_axis(ax)
    for i,(bar,value) in enumerate(zip(bars,values)): ax.text(bar.get_x()+bar.get_width()/2,bottoms[i]+value/2,f"{value:.3f}",ha="center",va="center",fontsize=10,fontweight="bold")
    ax.axhline(full_r2,linestyle="--",linewidth=1.0,alpha=0.75,color="#555555")
    ax.text(len(values)-0.05,full_r2+0.012,f"All-source R² = {full_r2:.4f}",ha="right",va="bottom",fontsize=9,fontweight="bold")
    fig.tight_layout()
    save_figure(fig,"Fig_Shapley_Cumulative_Contribution")


def print_results(shapley_df,combination_df):
    """Perform print results."""
    print("\n"+"="*70)
    print("Shapley-style Modality Contribution")
    print("="*70)
    print(shapley_df.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\n"+"="*70)
    print("Modality Combination Performance")
    print("="*70)
    print(combination_df.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    total = shapley_df[shapley_df["Modality"]=="Total"].iloc[0]
    full_r2 = combination_df.iloc[-1]["Test_R2"]
    print("\n"+"="*70)
    print("Shapley Consistency Check")
    print("="*70)
    print(f"Sum of Shapley contributions : {total['Shapley_Contribution_R2']:.9f}")
    print(f"Full-model R²                : {full_r2:.9f}")
    print(f"Contribution share           : {total['Contribution_Share_Percent']:.6f}%")


def main():
    """Run the main analysis workflow."""
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    print("="*70)
    print("Shapley-style Modality Contribution Analysis")
    print("="*70)
    df = load_results()
    value_function = get_r2_mapping(df)
    shapley = shapley_values(value_function)
    shapley_df = build_shapley_table(value_function,shapley)
    marginal_df = build_marginal_contribution_table(value_function)
    combination_df = build_combination_table(value_function)
    save_tables(shapley_df,marginal_df,combination_df)
    plot_shapley_contribution(shapley_df)
    plot_contribution_share(shapley_df)
    plot_combination_performance(combination_df)
    plot_synergy_heatmap(value_function)
    plot_shapley_waterfall(shapley_df,value_function)
    print_results(shapley_df,combination_df)
    print("\n"+"="*70)
    print("Figures and tables saved to:")
    print(OUTPUT_ROOT)
    print("="*70)
if __name__ == "__main__":
    main()