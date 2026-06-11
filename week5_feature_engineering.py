
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


# 3. Build the ENGAGEMENT feature from VLE click logs

print("STEP 3: Building engagement feature set from VLE clicks")

svle = pd.read_csv(DATA_DIR / "studentVle.csv")
print(f"Loaded {len(svle):,} click records")

# Create the same student_key in the VLE table for joining
svle["student_key"] = (svle["id_student"].astype(str)
                       + "_" + svle["code_module"]
                       + "_" + svle["code_presentation"])

# Aggregate per student-module: total clicks, active days, etc.
print("\nAggregating per-student engagement features...")

engagement = svle.groupby("student_key").agg(
    total_clicks=("sum_click", "sum"),
    mean_clicks_per_day=("sum_click", "mean"),
    max_clicks_in_a_day=("sum_click", "max"),
    active_days=("date", "nunique"),       # how many distinct days they clicked 在多少个日期点击过
    first_active_day=("date", "min"),
    last_active_day=("date", "max"),
).reset_index()

# how long between first and last click
engagement["activity_span"] = (engagement["last_active_day"]
                               - engagement["first_active_day"])

# Consistency measure: active_days / activity_span
# (1.0 means they were active every day; lower = more sporadic)
engagement["consistency"] = np.where(
    engagement["activity_span"] > 0,
    engagement["active_days"] / engagement["activity_span"],
    0
)

# clicks in the first 4 weeks (days 0-27) Gray & Perkins (2019)

early_clicks = svle[svle["date"] <= 27].groupby("student_key")["sum_click"].sum()
engagement = engagement.merge(
    early_clicks.rename("early_clicks_first_4_weeks"),
    on="student_key", how="left"
)
engagement["early_clicks_first_4_weeks"] = (
    engagement["early_clicks_first_4_weeks"].fillna(0)
)

print(f"\nEngagement feature set shape: {engagement.shape}")
print(f"Columns: {list(engagement.columns)}")
print(f"\nFirst 5 rows:")
print(engagement.head())



# 4.Merging and aligning feature sets.(合并对齐把demographic表和engagement表合并到一起，确保每个学生在两个表里都有记录。如果某个学生完全没有点击记录，行为特征就填0。)

print("STEP 4: Merging and aligning feature sets")

merged = demo_df.merge(engagement, on="student_key", how="left")

# Students with no VLE clicks at all , fill engagement features with 0
engagement_cols = ["total_clicks", "mean_clicks_per_day", "max_clicks_in_a_day",
                   "active_days", "activity_span", "consistency",
                   "early_clicks_first_4_weeks"]
for col in engagement_cols:
    merged[col] = merged[col].fillna(0)

# Drop helper columns not used as features
merged = merged.drop(columns=["first_active_day", "last_active_day"])

# Attach the target
merged = merged.merge(si[["student_key", "target"]], on="student_key", how="left")
merged = merged.dropna(subset=["target"])
merged["target"] = merged["target"].astype(int)

print(f"Final merged dataset shape: {merged.shape}")
print(f"Pass rate: {(merged['target'] == 1).mean() * 100:.2f}%")


# 5. Saving three feature sets(保存三个特征组。输出三个CSV文件：只有人口统计特征的、只有行为特征的、两者合并的)

print("STEP 5: Saving three feature sets")

features_demo = merged[["student_key"] + demographic_cols + ["target"]]
features_demo.to_csv("features_demographic.csv", index=False)
print(f"Saved features_demographic.csv ({len(demographic_cols)} features)")

features_eng = merged[["student_key"] + engagement_cols + ["target"]]
features_eng.to_csv("features_engagement.csv", index=False)
print(f"Saved features_engagement.csv ({len(engagement_cols)} features)")

all_feature_cols = demographic_cols + engagement_cols
features_combined = merged[["student_key"] + all_feature_cols + ["target"]]
features_combined.to_csv("features_combined.csv", index=False)
print(f"Saved features_combined.csv ({len(all_feature_cols)} features)")



