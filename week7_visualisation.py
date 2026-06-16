"""
Week 7: Analysis and Visualisation
====================================
Input (from Week 6):
  - results_summary.csv
  - results_per_fold.csv
  - fairness_by_subgroup.csv
  - features_demographic.csv
  - features_engagement_early.csv
  - features_engagement_mid.csv

Output (saved to week7_plots/):
  - fig1_f1_comparison.png         F1 bar chart (9 experiments)
  - fig2_all_metrics_heatmap.png   Accuracy/Precision/Recall/F1 heatmap
  - fig3_feature_importance.png    Random forest feature importance (mid set)
  - fig4_fairness_imd.png          F1 by IMD band
  - fig5_confusion_matrices.png    Confusion matrices (3 feature sets)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


OUT_DIR = Path("week7_plots")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# consistent colour palette
PALETTE = {
    "demographic":      "#4878CF",   # blue
    "engagement_early": "#6ACC65",   # green
    "engagement_mid":   "#D65F5F",   # red
}
MODEL_ORDER  = ["logistic_regression", "decision_tree", "random_forest"]
MODEL_LABELS = ["Logistic\nRegression", "Decision\nTree", "Random\nForest"]
FS_ORDER     = ["demographic", "engagement_early", "engagement_mid"]
FS_LABELS    = ["Demographic\n(pre-course)",
                "Engagement\n(first 4 weeks)",
                "Engagement\n(first half)"]


# load results

print("STEP 1: Loading Week 6 results")

summary  = pd.read_csv("results_summary.csv")
per_fold = pd.read_csv("results_per_fold.csv")
fairness = pd.read_csv("fairness_by_subgroup.csv")

print(f"  summary  : {summary.shape}")
print(f"  per_fold : {per_fold.shape}")
print(f"  fairness : {fairness.shape}")


# Figure 1: F1 comparison bar chart with error bars

print("\nGenerating Fig 1: F1 comparison bar chart...")

fig, ax = plt.subplots(figsize=(11, 6))

x       = np.arange(len(MODEL_ORDER))
width   = 0.25
offsets = [-width, 0, width]

for fs, label, offset in zip(FS_ORDER, FS_LABELS, offsets):
    vals = []
    errs = []
    for model in MODEL_ORDER:
        row = summary[(summary["feature_set"] == fs) &
                      (summary["model"] == model)]
        vals.append(row["f1_mean"].values[0])
        errs.append(row["f1_std"].values[0])

    bars = ax.bar(x + offset, vals, width,
                  yerr=errs, capsize=4,
                  color=PALETTE[fs], alpha=0.88,
                  label=label, error_kw={"elinewidth": 1.2})

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontsize=11)
ax.set_ylabel("F1 Score (mean +/- std, 5-fold CV)", fontsize=11)
ax.set_title("F1 Score by Model and Feature Set", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.legend(title="Feature Set", fontsize=9, title_fontsize=10)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_f1_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_f1_comparison.png")


# Figure 2: Heatmap of all four metrics

print("Generating Fig 2: All-metrics heatmap...")

metrics    = ["accuracy", "precision", "recall", "f1"]
met_labels = ["Accuracy", "Precision", "Recall", "F1"]

fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharey=True)

for ax, metric, mlabel in zip(axes, metrics, met_labels):
    pivot_data = []
    for model in MODEL_ORDER:
        row_vals = []
        for fs in FS_ORDER:
            val = summary[(summary["model"] == model) &
                          (summary["feature_set"] == fs)
                          ][f"{metric}_mean"].values[0]
            row_vals.append(val)
        pivot_data.append(row_vals)

    pivot = pd.DataFrame(pivot_data,
                         index=MODEL_LABELS,
                         columns=FS_LABELS)

    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f",
                cmap="YlGn", vmin=0.45, vmax=1.0,
                linewidths=0.5, cbar=(ax == axes[-1]),
                annot_kws={"size": 10})
    ax.set_title(mlabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    if ax != axes[0]:
        ax.set_ylabel("")

fig.suptitle("Model Performance Across All Metrics and Feature Sets",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_all_metrics_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_all_metrics_heatmap.png")


# Figure 3: Random forest feature importance on mid-engagement set

print("Generating Fig 3: Feature importance (mid engagement set)...")

mid = pd.read_csv("features_engagement_mid.csv")
feature_cols = [c for c in mid.columns
                if c not in ("student_key", "target")]
X = mid[feature_cols].values
y = mid["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

rf = RandomForestClassifier(n_estimators=100,
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(importances.index, importances.values,
        color=PALETTE["engagement_mid"], alpha=0.88)

ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11)
ax.set_title("Random Forest Feature Importance\n"
             "(Mid-Course Engagement Feature Set)",
             fontsize=13, fontweight="bold")
ax.xaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_feature_importance.png")


# Figure 4: Fairness, F1 by IMD band

print("Generating Fig 4: Fairness check by IMD band...")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(fairness["imd_band_encoded"], fairness["f1"],
        marker="o", linewidth=2, color=PALETTE["demographic"],
        markersize=7, label="F1")
ax.fill_between(fairness["imd_band_encoded"],
                fairness["f1"] - 0.02,
                fairness["f1"] + 0.02,
                alpha=0.15, color=PALETTE["demographic"])

ax.set_xlabel("IMD Band (0 = most deprived, 10 = least deprived)", fontsize=11)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Model Performance by Socioeconomic Background\n"
             "(Demographic feature set + Random Forest)",
             fontsize=13, fontweight="bold")
ax.set_xticks(fairness["imd_band_encoded"])
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

for _, row in fairness.iterrows():
    ax.annotate(f"n={int(row['n_students'])}",
                xy=(row["imd_band_encoded"], row["f1"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=7.5, color="dimgrey")

plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_fairness_imd.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_fairness_imd.png")


# Figure 5: Confusion matrices, one per feature set (random forest)

print("Generating Fig 5: Confusion matrices...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

fs_files = {
    "demographic":      "features_demographic.csv",
    "engagement_early": "features_engagement_early.csv",
    "engagement_mid":   "features_engagement_mid.csv",
}

for ax, (fs_name, fs_label) in zip(axes, zip(FS_ORDER, FS_LABELS)):
    df = pd.read_csv(fs_files[fs_name])
    feat_cols = [c for c in df.columns
                 if c not in ("student_key", "target")]
    Xf = df[feat_cols].values
    yf = df["target"].values

    Xtr, Xte, ytr, yte = train_test_split(
        Xf, yf, test_size=0.25, stratify=yf,
        random_state=RANDOM_STATE)

    rf_tmp = RandomForestClassifier(n_estimators=100,
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1)
    rf_tmp.fit(Xtr, ytr)
    yp = rf_tmp.predict(Xte)

    cm = confusion_matrix(yte, yp)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                cmap="Blues", ax=ax,
                xticklabels=["Pred Fail", "Pred Pass"],
                yticklabels=["True Fail", "True Pass"],
                vmin=0, vmax=1, cbar=False,
                annot_kws={"size": 12})

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72,
                    f"(n={cm[i,j]:,})",
                    ha="center", va="center",
                    fontsize=8, color="dimgrey")

    ax.set_title(fs_label, fontsize=10, fontweight="bold",
                 color=PALETTE[fs_name])

fig.suptitle("Confusion Matrices - Random Forest (normalised by row)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_confusion_matrices.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig5_confusion_matrices.png")


print(f"\nWeek 7 complete. All figures saved to: {OUT_DIR}/")
