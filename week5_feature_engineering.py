
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")


# 1. Load data（1: 加载Week 4的输出）


print("STEP 1: Loading data from Week 4")

si = pd.read_csv("student_info_with_target.csv")
print(f"Loaded {len(si)} students with target labels")

si["student_key"] = (si["id_student"].astype(str)
                     + "_" + si["code_module"]
                     + "_" + si["code_presentation"])


# 2. Build the DEMOGRAPHIC feature set
# （提取8个demographic特征：性别、年龄段、学历、贫困指数、残疾状态、地区、过往尝试次数、学分数。然后把文字类别转换成数字


print("STEP 2: Building demographic feature set")

demographic_cols = [
    "gender",            
    "age_band",          # 0-35, 35-55, 55<=
    "highest_education", # 5 levels
    "imd_band",          # socioeconomic deprivation 贫困指数 (0-10%, 10-20%, ...)
    "disability",        # Y / N 残疾？
    "region",            # geographic region
    "num_of_prev_attempts",  
    "studied_credits",   
]

demo_df = si[["student_key"] + demographic_cols].copy()

# Fill missing imd_band with a separate "Unknown" category rather than dropping rows
demo_df["imd_band"] = demo_df["imd_band"].fillna("Unknown")

categorical_cols = ["gender", "age_band", "highest_education", "imd_band",
                    "disability", "region"]
for col in categorical_cols:
    le = LabelEncoder()
    demo_df[col] = le.fit_transform(demo_df[col].astype(str))
    print(f"  Encoded {col}: {len(le.classes_)} categories")

print(f"\nDemographic feature set shape: {demo_df.shape}")
print(f"Columns: {list(demo_df.columns)}")


# 3. Build the ENGAGEMENT features from VLE click logs
# Two time windows: early (first 4 weeks, days 0-27)
#                   mid (first half of course, days 0-135)

print("STEP 3: Building engagement feature sets from VLE clicks")

svle = pd.read_csv(DATA_DIR / "studentVle.csv")
print(f"Loaded {len(svle):,} click records")

svle["student_key"] = (svle["id_student"].astype(str)
                       + "_" + svle["code_module"]
                       + "_" + svle["code_presentation"])


def build_engagement(svle_subset, suffix):
    """Build engagement features from a time-windowed slice of the VLE data."""
    eng = svle_subset.groupby("student_key").agg(
        total_clicks=("sum_click", "sum"),
        mean_clicks_per_day=("sum_click", "mean"),
        max_clicks_in_a_day=("sum_click", "max"),
        active_days=("date", "nunique"),
        first_active_day=("date", "min"),
        last_active_day=("date", "max"),
    ).reset_index()

    eng["activity_span"] = (eng["last_active_day"]
                            - eng["first_active_day"])

    eng["consistency"] = np.where(
        eng["activity_span"] > 0,
        eng["active_days"] / eng["activity_span"],
        0
    )

    eng = eng.drop(columns=["first_active_day", "last_active_day"])

    # rename columns to mark the time window (e.g. total_clicks_early)
    eng.columns = ["student_key"] + [f"{c}_{suffix}" for c in eng.columns
                                     if c != "student_key"]
    return eng


# Early engagement: first 4 weeks (days 0-27), Gray & Perkins (2019)
print("\nBuilding EARLY engagement (days 0-27)...")
svle_early = svle[svle["date"] <= 27]
engagement_early = build_engagement(svle_early, "early")
print(f"  Early engagement feature set shape: {engagement_early.shape}")

# Mid engagement: first half of the course (days 0-135)
print("Building MID engagement (days 0-135)...")
svle_mid = svle[svle["date"] <= 135]
engagement_mid = build_engagement(svle_mid, "mid")
print(f"  Mid engagement feature set shape: {engagement_mid.shape}")


# 4. Merging and aligning feature sets

print("STEP 4: Merging and aligning feature sets")

merged = demo_df.merge(engagement_early, on="student_key", how="left")
merged = merged.merge(engagement_mid, on="student_key", how="left")

# columns belonging to each engagement set
early_cols = [c for c in engagement_early.columns if c != "student_key"]
mid_cols   = [c for c in engagement_mid.columns   if c != "student_key"]

# Students with no clicks in the window -> fill with 0
for col in early_cols + mid_cols:
    merged[col] = merged[col].fillna(0)

# Attach the target
merged = merged.merge(si[["student_key", "target"]], on="student_key", how="left")
merged = merged.dropna(subset=["target"])
merged["target"] = merged["target"].astype(int)

print(f"Final merged dataset shape: {merged.shape}")
print(f"Pass rate: {(merged['target'] == 1).mean() * 100:.2f}%")


# 5. Saving three feature sets
# Group 1: demographic only           (available before course starts)
# Group 2: early engagement (4 weeks) (available 4 weeks into course)
# Group 3: mid engagement (half)      (available halfway through course)

print("STEP 5: Saving three feature sets")

features_demo = merged[["student_key"] + demographic_cols + ["target"]]
features_demo.to_csv("features_demographic.csv", index=False)
print(f"Saved features_demographic.csv ({len(demographic_cols)} features)")

features_early = merged[["student_key"] + early_cols + ["target"]]
features_early.to_csv("features_engagement_early.csv", index=False)
print(f"Saved features_engagement_early.csv ({len(early_cols)} features)")

features_mid = merged[["student_key"] + mid_cols + ["target"]]
features_mid.to_csv("features_engagement_mid.csv", index=False)
print(f"Saved features_engagement_mid.csv ({len(mid_cols)} features)")
