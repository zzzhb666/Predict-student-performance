
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

RANDOM_STATE = 42
N_FOLDS = 5


# 1. Load the three feature sets

print("STEP 1: Loading the three feature sets from Week 5")

feature_files = {
    "demographic": "features_demographic.csv",
    "engagement":  "features_engagement.csv",
    "combined":    "features_combined.csv",
}

feature_sets = {}
for name, fname in feature_files.items():
    df = pd.read_csv(fname)
    feature_sets[name] = df
    print(f"  {name:12s}: {df.shape[0]} rows, "
          f"{df.shape[1] - 2} features")  # minus student_key + target



# 2. Define the three models


print("STEP 2: Defining models")


def make_models():
    """Fresh model instances each time, so no state leaks between runs."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }


for m in make_models():
    print(f"  - {m}")


# 3. the 9 experiments

print("STEP 3: Running 9 experiments (3 models x 3 feature sets)")

needs_scaling = {"logistic_regression"}

per_fold_rows = []     # one row per (feature_set, model, fold, metric)
summary_rows = []      # one row per (feature_set, model): mean +/- std

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                      random_state=RANDOM_STATE)

for fs_name, df in feature_sets.items():
    feature_cols = [c for c in df.columns
                    if c not in ("student_key", "target")]
    X = df[feature_cols].values
    y = df["target"].values

    for model_name in make_models():
        fold_scores = {"accuracy": [], "precision": [],
                       "recall": [], "f1": []}

        for fold_idx, (train_i, test_i) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_i], X[test_i]
            y_train, y_test = y[train_i], y[test_i]

            if model_name in needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # fresh model per fold
            model = make_models()[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_scores["accuracy"].append(
                accuracy_score(y_test, y_pred))
            fold_scores["precision"].append(
                precision_score(y_test, y_pred, zero_division=0))
            fold_scores["recall"].append(
                recall_score(y_test, y_pred, zero_division=0))
            fold_scores["f1"].append(
                f1_score(y_test, y_pred, zero_division=0))

            for metric, val in fold_scores.items():
                per_fold_rows.append({
                    "feature_set": fs_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "metric": metric,
                    "score": val[-1],
                })

        # summarise 
        row = {"feature_set": fs_name, "model": model_name}
        for metric, vals in fold_scores.items():
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_std"] = np.std(vals)
        summary_rows.append(row)

        print(f"  {fs_name:12s} | {model_name:20s} | "
              f"F1 = {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}")


# 4. Save cross-validation results

print("STEP 4: Saving cross-validation results")


summary_df = pd.DataFrame(summary_rows)
metric_order = []
for metric in ["accuracy", "precision", "recall", "f1"]:
    metric_order += [f"{metric}_mean", f"{metric}_std"]
summary_df = summary_df[["feature_set", "model"] + metric_order]
summary_df.to_csv("results_summary.csv", index=False)
print("Saved results_summary.csv")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv("results_per_fold.csv", index=False)
print("Saved results_per_fold.csv")

print("\nF1 score summary (mean):")
pivot = summary_df.pivot(index="model", columns="feature_set",
                         values="f1_mean")
print(pivot.round(4).to_string())



# 5. Fairness check: error rates across IMD subgroups

print("STEP 5: Fairness check across IMD subgroups")

combined = feature_sets["combined"].copy()
feature_cols = [c for c in combined.columns
                if c not in ("student_key", "target")]

X = combined[feature_cols].values
y = combined["target"].values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, combined.index,
    test_size=0.25, stratify=y, random_state=RANDOM_STATE)

rf = RandomForestClassifier(n_estimators=100,
                            random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Attach predictions back to the test rows so we can group by IMD band
test_df = combined.loc[idx_test].copy()
test_df["y_true"] = y_test
test_df["y_pred"] = y_pred

fairness_rows = []
for band, grp in test_df.groupby("imd_band"):
    if len(grp) == 0:
        continue
    fairness_rows.append({
        "imd_band_encoded": band,
        "n_students": len(grp),
        "accuracy": accuracy_score(grp["y_true"], grp["y_pred"]),
        "precision": precision_score(grp["y_true"], grp["y_pred"],
                                     zero_division=0),
        "recall": recall_score(grp["y_true"], grp["y_pred"],
                               zero_division=0),
        "f1": f1_score(grp["y_true"], grp["y_pred"], zero_division=0),
    })

fairness_df = pd.DataFrame(fairness_rows).sort_values("imd_band_encoded")
fairness_df.to_csv("fairness_by_subgroup.csv", index=False)
print("Saved fairness_by_subgroup.csv")
print("\nPer-subgroup performance (combined + random forest):")
print(fairness_df.round(4).to_string(index=False))

