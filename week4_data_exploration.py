
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_DIR = Path("data")


print("STEP 1: Loading files")

files = {
    "courses": "courses.csv",                       
    "assessments": "assessments.csv",               # All assessments 
    "vle": "vle.csv",                               # VLE 
    "student_info": "studentInfo.csv",              # Demographic data per student
    "student_registration": "studentRegistration.csv",  # Registration/withdrawal dates
    "student_assessment": "studentAssessment.csv",  # scores
    "student_vle": "studentVle.csv",                # Daily click logs
}

data = {}
for name, filename in files.items():
    filepath = DATA_DIR / filename
    print(f"  Loading {filename}...", end=" ")
    data[name] = pd.read_csv(filepath)
    print(f"shape = {data[name].shape}")

print("\nAll files loaded successfully.\n")


# 2. Look student_info 

print("STEP 2: Look student_info")
si = data["student_info"]
print(f"\nTotal student records: {len(si)}")
print(f"Columns: {list(si.columns)}")
print(f"\nFirst 5 rows:")
print(si.head())

print(f"\nUnique values in key columns:")
for col in ["gender", "age_band", "highest_education", "imd_band",
            "disability", "final_result"]:
    if col in si.columns:
        print(f"  {col}: {si[col].unique()}")


# 3. Check missing values 缺失值

print("STEP 3: Missing values in student_info")

missing = si.isnull().sum()
missing_pct = (missing / len(si) * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
})
print(missing_summary[missing_summary["missing_count"] > 0])
print("\nThere is some missing values in OULAD.")


# 4. Look the final_result distribution 

print("STEP 4: Final result distribution")

result_counts = si["final_result"].value_counts()
print(result_counts)
print(f"\nPercentages:")
print((result_counts / len(si) * 100).round(2))


# 5. Create: pass vs fail 

# Pass = "Pass" or "Distinction"
# Fail = "Fail" or "Withdrawn"

print("STEP 5: Creating pass/fail target")

pass_outcomes = ["Pass", "Distinction"]
fail_outcomes = ["Fail", "Withdrawn"]

si["target"] = si["final_result"].apply(
    lambda x: 1 if x in pass_outcomes else (0 if x in fail_outcomes else np.nan)
)

print(f"Target distribution:")
print(si["target"].value_counts())
print(f"\nPass rate: {(si['target'] == 1).mean() * 100:.2f}%")
print(f"Fail rate: {(si['target'] == 0).mean() * 100:.2f}%")
print(f"Missing target (no result): {si['target'].isnull().sum()}")

# 6.Look at student_vle 

print("STEP 6: Look at student_vle")

svle = data["student_vle"]
print(f"Total click records: {len(svle):,}")
print(f"Columns: {list(svle.columns)}")
print(f"\nFirst 5 rows:")
print(svle.head())

clicks_per_student = svle.groupby("id_student")["sum_click"].sum()
print(f"\nClick statistics per student:")
print(f"  Mean: {clicks_per_student.mean():.1f}")
print(f"  Median: {clicks_per_student.median():.1f}")
print(f"  Max: {clicks_per_student.max():,}")
print(f"  Min: {clicks_per_student.min()}")


# 7. Save

print("STEP 7: Saving plots")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Final result 
si["final_result"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Distribution of Final Results")
axes[0].set_xlabel("Final Result")
axes[0].set_ylabel("Number of Students")
axes[0].tick_params(axis="x", rotation=45)

# Plot 2: Pass/fail by gender
gender_target = si.groupby(["gender", "target"]).size().unstack()
gender_target.plot(kind="bar", ax=axes[1], color=["salmon", "lightgreen"])
axes[1].set_title("Pass/Fail by Gender")
axes[1].set_xlabel("Gender")
axes[1].set_ylabel("Number of Students")
axes[1].legend(["Fail", "Pass"])
axes[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("exploration_plots.png", dpi=100, bbox_inches="tight")
print("Saved: exploration_plots.png")

print(" Saving processed data for Week 5")
si_clean = si.dropna(subset=["target"]).copy()
si_clean["target"] = si_clean["target"].astype(int)

si_clean.to_csv("student_info_with_target.csv", index=False)
print(f"Saved: student_info_with_target.csv ({len(si_clean)} rows)")
print(f"  - Original: {len(si)} students")
print(f"  - After removing 'no result': {len(si_clean)} students")

