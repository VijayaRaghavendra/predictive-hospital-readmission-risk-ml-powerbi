import pandas as pd
import numpy as np
import os, json
# Define the target variable globally
TARGET = "READMIT_30DAY"
print("=" * 65)
print("STEP 2: Preprocessing & Feature Engineering")
print("=" * 65)

# Setup directories
proc_dir = "data/processed"
os.makedirs(proc_dir, exist_ok=True)

print("\n[1] Loading raw data...")
# Loading with specific columns to save memory
claims = pd.read_csv("data/raw/inpatient_claims.csv")
bene = pd.read_csv("data/raw/beneficiary_summary.csv")
print(f"    Claims: {claims.shape}, Beneficiary: {bene.shape}")

print("\n[2] Preprocessing Dates & Target...")
# 1. Convert Dates (SynPUF uses DD-Mon-YYYY)
claims['ADMIT_DATE'] = pd.to_datetime(claims['CLM_ADMSN_DT'], errors='coerce')
claims['DISCHARGE_DATE'] = pd.to_datetime(claims['NCH_BENE_DSCHRG_DT'], errors='coerce')
claims = claims.dropna(subset=['ADMIT_DATE', 'DISCHARGE_DATE'])

# 2. Calculate LOS (Length of Stay)
claims['LOS_DAYS'] = (claims['DISCHARGE_DATE'] - claims['ADMIT_DATE']).dt.days.clip(lower=0)

# 3. Derive Target: 30-Day Readmission
# Sort by patient and date to find the 'next' admission
claims = claims.sort_values(['BENE_ID', 'ADMIT_DATE'])
claims['NEXT_ADMIT_DT'] = claims.groupby('BENE_ID')['ADMIT_DATE'].shift(-1)
claims['DAYS_TO_READMIT'] = (claims['NEXT_ADMIT_DT'] - claims['DISCHARGE_DATE']).dt.days
# Target = 1 if readmitted within 30 days
claims['READMIT_30DAY'] = ((claims['DAYS_TO_READMIT'] >= 0) & (claims['DAYS_TO_READMIT'] <= 30)).astype(int)

print("\n[3] Extracting Chronic Conditions from Diagnosis Codes...")
# In SynPUF, CCs are often flags, but if missing, we derive them from ICD-10 columns
CC_CODES = {
    'CHF': 'I50', 'CKD': 'N18', 'COPD': 'J44', 
    'DIABETES': ('E08', 'E09', 'E10', 'E11', 'E13'),
    'ALZHEIMERS': ('G30', 'F00'), 'STROKE': 'I6'
}
diag_cols = ['PRNCPAL_DGNS_CD'] + [f'ICD_DGNS_CD{i}' for i in range(1, 26)]

for cond, code in CC_CODES.items():
    # Check all diagnosis columns for the specific condition prefix
    claims[f'CC_{cond}'] = claims[diag_cols].apply(
        lambda x: x.str.startswith(code, na=False)
    ).any(axis=1).astype(int)

claims['CC_COUNT'] = claims[[f'CC_{c}' for c in CC_CODES]].sum(axis=1)

print("\n[4] Merging Demographic Data...")
# Dual Eligible: Check monthly status columns (if any month > 0)
dual_cols = [c for c in bene.columns if 'DUAL_STUS_CD' in c]
bene['DUAL_ELIGIBLE'] = (bene[dual_cols].notnull().any(axis=1)).astype(int)

# Map Gender and Age
bene_feat = bene[['BENE_ID', 'AGE_AT_END_REF_YR', 'SEX_IDENT_CD', 'BENE_RACE_CD', 'DUAL_ELIGIBLE']].copy()
bene_feat.columns = ['BENE_ID', 'AGE', 'GENDER', 'RACE', 'DUAL_ELIGIBLE']
bene_feat['MALE'] = (bene_feat['GENDER'] == 1).astype(int)

df = claims.merge(bene_feat, on="BENE_ID", how="left")

print("\n[5] Advanced Feature Engineering...")
# Time-based features
df["ADMIT_YEAR"]    = df["ADMIT_DATE"].dt.year
df["ADMIT_QUARTER"] = df["ADMIT_DATE"].dt.quarter
df["ADMIT_DOW"]     = df["ADMIT_DATE"].dt.dayofweek
df["MONTH_SIN"]     = np.sin(2 * np.pi * df["ADMIT_DATE"].dt.month / 12)
df["MONTH_COS"]     = np.cos(2 * np.pi * df["ADMIT_DATE"].dt.month / 12)

# Clinical Segments
df["LOS_CAT"] = pd.cut(df["LOS_DAYS"], bins=[-1, 2, 7, 999], labels=[0, 1, 2]).astype(int)
df["HIGH_RISK_DISCH"]  = (df["PTNT_DSCHRG_STUS_CD"] == 4).astype(int) # SNF Transfer
df["POST_ACUTE_DISCH"] = df["PTNT_DSCHRG_STUS_CD"].isin([2, 3]).astype(int)

# Interactions & Risk Scores
df["CHF_CKD"]        = df["CC_CHF"] * df["CC_CKD"]
df["FRAILTY_SCORE"]  = (df["AGE"] >= 80).astype(int)*2 + (df["CC_COUNT"] >= 4).astype(int)*2 + df["CC_ALZHEIMERS"]
df["SOCIAL_RISK"]    = df["DUAL_ELIGIBLE"] 
df["WEEKEND_ADMIT"]  = (df["ADMIT_DOW"] >= 5).astype(int)

# Prior Usage (Rolling 12-month count)
df = df.sort_values(['BENE_ID', 'ADMIT_DATE'])
df['PRIOR_12M_ADMITS'] = df.groupby('BENE_ID')['CLM_ID'].cumcount() # Proxy for history
df["HIGH_PRIOR_USE"]   = (df["PRIOR_12M_ADMITS"] >= 2).astype(int)

# Cost Tier (Handling Nulls: Use 0 if amount is missing, assuming no charges)
df['CLM_TOT_CHRG_AMT'] = df['CLM_TOT_CHRG_AMT'].fillna(0)
cost_q = df["CLM_TOT_CHRG_AMT"].quantile([0.33, 0.66])
df["COST_TIER"] = pd.cut(df["CLM_TOT_CHRG_AMT"], 
                         bins=[-1, cost_q[0.33], cost_q[0.66], float("inf")], 
                         labels=[0,1,2]).astype(int)

# DRG Grouping
DRG_MAP = {470:"Orthopedic", 291:"Cardiac", 292:"Cardiac", 293:"Cardiac", 871:"Infectious"}
df["DRG_GROUP"] = df["CLM_DRG_CD"].map(DRG_MAP).fillna("Other")
dummies = pd.get_dummies(df["DRG_GROUP"], prefix="DRG", drop_first=True).astype(int)
df = pd.concat([df, dummies], axis=1)

print("\n[6] Handling Nulls & Finalizing...")
# HEALTHCARE RULE: Don't use median for binary flags. 
# If a chronic condition is null, assume it's NOT present (0).
binary_cols = [c for c in df.columns if c.startswith('CC_') or c in ['DUAL_ELIGIBLE', 'MALE']]
df[binary_cols] = df[binary_cols].fillna(0)

# For continuous clinical metrics like AGE, median is acceptable but check first
df['AGE'] = df['AGE'].fillna(df['AGE'].median())

# FIXED — Keep BENE_ID and dates in the export, drop only from ML features
DROP_FROM_MODEL = ["CLM_ID", "ADMIT_DATE", "DISCHARGE_DATE", "NEXT_ADMIT_DT",
                   "DAYS_TO_READMIT", "GENDER", "RACE", "DRG_GROUP", "CLM_DRG_CD",
                   "CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT"]

# ML feature matrix (no BENE_ID, no dates — pure numbers for XGBoost)
X = df.drop(columns=[c for c in DROP_FROM_MODEL if c in df.columns] + [TARGET, "BENE_ID"])
X = X.select_dtypes(include=[np.number])
y = df[TARGET]

# Full export WITH identifiers (for Power BI — model won't use these)
id_cols_to_keep = ["BENE_ID", "ADMIT_DATE", "DISCHARGE_DATE"]
id_cols_present = [c for c in id_cols_to_keep if c in df.columns]

final_data = pd.concat([df[id_cols_present].reset_index(drop=True),
                        X.reset_index(drop=True),
                        y.reset_index(drop=True)], axis=1)

final_data.to_csv("data/processed/features.csv", index=False)

# Save feature names WITHOUT the id cols (model only uses X columns)
meta = {
    "n_samples": int(len(df)),
    "n_features": int(X.shape[1]),
    "target": TARGET,
    "positive_rate": float(y.mean()),
    "feature_names": list(X.columns)   # ← XGBoost uses only these, not BENE_ID
}

meta = {"n_samples": int(len(df)), "n_features": int(X.shape[1]), "target": TARGET, 
        "positive_rate": float(y.mean()), "feature_names": list(X.columns)}
with open(os.path.join(proc_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ STEP 2 COMPLETE")