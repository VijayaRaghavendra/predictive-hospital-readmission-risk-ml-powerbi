"""
================================================================================
  05_export_powerbi.py
  READMISSION RISK ENGINE  —  Power BI Data Export
================================================================================

WHAT THIS SCRIPT DOES:
  Takes the trained XGBoost model scores and produces THREE clean CSV files
  ready to load directly into Power BI Desktop:

  OUTPUT 1 → powerbi_patients.csv       (one row per claim/encounter)
    The main fact table. Each row = one patient encounter with risk score,
    risk tier, all clinical fields, and a generated PATIENT_ID for display.

  OUTPUT 2 → powerbi_physician_summary.csv   (one row per physician)
    Physician-level summary: how many patients, readmit rate, avg risk score.
    Doctors can use this to see which attending physicians have the highest
    readmission burden on their patient panel.

  OUTPUT 3 → powerbi_hospital_summary.csv    (one row per hospital)
    Hospital-level summary: volume, readmit rate, avg risk, critical count.
    For exec-level benchmarking across 4,902 provider organisations.

WHY NO REAL PATIENT ID EXISTS:
  This is a CMS public research dataset. Real beneficiary IDs (BENE_ID) are
  stripped in the public release for HIPAA compliance. The dataset has:
    - ORG_NPI_NUM  → hospital/facility NPI (4,902 unique hospitals)
    - AT_PHYSN_NPI → attending physician NPI (2,463 unique physicians)
  Neither is a patient identifier. We generate a sequential PATIENT_ID
  (PAT-00001, PAT-00002 …) so Power BI has a unique key per row.
  In a real hospital deployment, you would replace PATIENT_ID with the
  hospital's MRN (Medical Record Number) from the EHR system.

USAGE:
  python 05_export_powerbi.py
  Outputs saved to: outputs/
================================================================================
"""

import pandas as pd
import numpy as np
import pickle, json, os, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/processed/features.csv"
MODEL_PATH = "outputs/xgb_readmission_model.pkl"
META_PATH  = "outputs/xgb_metadata.json"
OUT_DIR    = "outputs"

TIERS = {
    "Low"      : (0.00, 0.35),
    "Moderate" : (0.35, 0.55),
    "High"     : (0.55, 0.72),
    "Critical" : (0.72, 1.01),
}

TIER_PRIORITY = {"Critical": 1, "High": 2, "Moderate": 3, "Low": 4}

RECOMMENDED_ACTIONS = {
    "Critical" : "Same-day discharge coordinator · Immediate care transition planning",
    "High"     : "48-hour post-discharge phone call · Medication reconciliation review",
    "Moderate" : "7-day follow-up appointment · Social work referral if dual-eligible",
    "Low"      : "Standard discharge instructions · Routine 30-day check-in",
}

ALERT_FLAGS = {
    "Critical" : "🚨 ALERT — Intervene before discharge",
    "High"     : "⚠️  WATCH — Schedule follow-up",
    "Moderate" : "📋 MONITOR — Standard pathway",
    "Low"      : "✅ SAFE — Routine discharge",
}

print("=" * 72)
print("  05_export_powerbi.py  —  Power BI Data Export")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD RAW DATA (before preprocessing, to keep identifier columns)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading raw data ...")
df_raw = pd.read_csv(DATA_PATH)
print(f"  Raw data: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

# Save identifier columns BEFORE preprocessing strips them
# ORG_NPI_NUM  = Hospital/Facility NPI (National Provider Identifier)
# AT_PHYSN_NPI = Attending Physician NPI
identifiers = df_raw[["ORG_NPI_NUM", "AT_PHYSN_NPI"]].copy()
identifiers["ORG_NPI_NUM"]  = identifiers["ORG_NPI_NUM"].fillna(0).astype(int).astype(str)
identifiers["AT_PHYSN_NPI"] = identifiers["AT_PHYSN_NPI"].fillna(0).astype(int).astype(str)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: EXACT SAME PREPROCESSING AS MODEL (must be identical)
# ─────────────────────────────────────────────────────────────────────────────
print("[2] Preprocessing (identical to model training) ...")

df = df_raw.copy()

zero_var = [c for c in df.columns
            if df[c].nunique(dropna=False) <= 1 or df[c].isnull().mean() == 1.0]
id_cols  = ["ORG_NPI_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI"]
drop_all = list(set(zero_var + id_cols + ["CLM_PMT_AMT"]))
drop_all = [c for c in drop_all if c in df.columns]
df.drop(columns=drop_all, inplace=True)

df["REV_CNTR_IS_450"] = (df["REV_CNTR"] == 450).astype(int)
df.drop(columns=["REV_CNTR"], inplace=True)

if df["NCH_IP_NCVRD_CHRG_AMT"].equals(df["NCH_IP_TOT_DDCTN_AMT"]):
    df.drop(columns=["NCH_IP_TOT_DDCTN_AMT"], inplace=True)

col = "REV_CNTR_DDCTBL_COINSRNC_CD"
df[col] = df.groupby("ADMIT_YEAR")[col].transform(lambda x: x.fillna(x.median()))
df[col]  = df[col].fillna(df[col].median())

df["PRIOR_12M_ADMITS"] = df["PRIOR_12M_ADMITS"].clip(
    upper=np.percentile(df["PRIOR_12M_ADMITS"], 99))
df["CLM_TOT_CHRG_AMT"] = df["CLM_TOT_CHRG_AMT"].clip(
    upper=np.percentile(df["CLM_TOT_CHRG_AMT"], 99.5))
df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"] = df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"].clip(
    upper=np.percentile(df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"], 99.5))

TARGET   = "READMIT_30DAY"
YEAR_COL = "ADMIT_YEAR"

with open(META_PATH) as f:
    meta = json.load(f)
FEAT_COLS = meta["feature_names"]

X = df[FEAT_COLS].values.astype(np.float32)
y = df[TARGET].values.astype(np.int32)

print(f"  Features prepared: {len(FEAT_COLS)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SCORE EVERY PATIENT
# ─────────────────────────────────────────────────────────────────────────────
print("[3] Scoring all patients with XGBoost model ...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

y_prob = model.predict_proba(X)[:, 1]

def assign_tier(p):
    for t, (lo, hi) in TIERS.items():
        if lo <= p < hi:
            return t
    return "Critical"

risk_tiers    = [assign_tier(p) for p in y_prob]
tier_priority = [TIER_PRIORITY[t] for t in risk_tiers]
actions       = [RECOMMENDED_ACTIONS[t] for t in risk_tiers]
alert_flags   = [ALERT_FLAGS[t] for t in risk_tiers]

print(f"  Scored: {len(y_prob):,} patients")
for t in ["Critical", "High", "Moderate", "Low"]:
    n = risk_tiers.count(t)
    print(f"    {t:<10}: {n:>6,}  ({n/len(y_prob)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: BUILD PATIENT-LEVEL EXPORT  (powerbi_patients.csv)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Building patient-level export ...")

# ── Generate PATIENT_ID ──────────────────────────────────────────────────────
# Format: PAT-YYYY-NNNNNN
# Year prefix makes it meaningful; sequential number makes it unique
# We also keep BENE_ID from the source for cross-referencing
bene_ids = [
    str(df_raw['BENE_ID'].iloc[i]) if 'BENE_ID' in df_raw.columns else "UNKNOWN"
    for i in range(len(df))
]
patient_ids = [
    f"PAT-{int(df['ADMIT_YEAR'].iloc[i])}-{i+1:06d}"
    for i in range(len(df))
]

# ── Human-readable derived columns ──────────────────────────────────────────
admit_type_label = df["CLM_IP_ADMSN_TYPE_CD"].map(
    {1: "Emergency", 2: "Urgent", 3: "Elective"}
).fillna("Other")

age_group = pd.cut(
    df["AGE"],
    bins=[0, 50, 65, 75, 85, 200],
    labels=["Under 50", "50–64", "65–74", "75–84", "85+"]
).astype(str)

gender_label = df["MALE"].map({1: "Male", 0: "Female"}).fillna("Unknown")

dual_label = df["DUAL_ELIGIBLE"].map(
    {1: "Dual (Medicare + Medicaid)", 0: "Medicare Only"}
).fillna("Unknown")

los_category = df["LOS_DAYS"].apply(lambda x:
    "Same-Day (0d)" if x == 0
    else "Short Stay (1–3d)" if x <= 3
    else "Medium Stay (4–7d)" if x <= 7
    else "Long Stay (8d+)"
)

frailty_label = df["FRAILTY_SCORE"].map(
    {0: "None (0)", 1: "Mild (1)", 2: "Moderate (2)", 3: "Severe (3)"}
).fillna("Unknown")

weekend_label = df["WEEKEND_ADMIT"].map({1: "Weekend", 0: "Weekday"})

dow_label = df["ADMIT_DOW"].map(
    {0: "Monday", 1: "Tuesday", 2: "Wednesday",
     3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
)

readmit_label = pd.Series(y).map({1: "Readmitted", 0: "No Readmission"})

risk_score_pct = (y_prob * 100).round(1)

# ── Comorbidity summary string  (useful for doctor view) ─────────────────────
def comorbidity_string(row):
    conditions = []
    if row["CC_DIABETES"]   == 1: conditions.append("Diabetes")
    if row["CC_CKD"]        == 1: conditions.append("CKD")
    if row["CC_CHF"]        == 1: conditions.append("CHF")
    if row["CC_COPD"]       == 1: conditions.append("COPD")
    if row["CC_ALZHEIMERS"] == 1: conditions.append("Alzheimer's")
    return ", ".join(conditions) if conditions else "None documented"

print("  Building comorbidity strings ...")
comorbidity_strings = df.apply(comorbidity_string, axis=1)

# ── Re-attach identifier columns (hospital NPI, physician NPI) ───────────────
# Mask the NPI as a display-friendly code for the portfolio
hospital_id  = identifiers["ORG_NPI_NUM"].apply(
    lambda x: f"HOSP-{str(x)[-5:]}" if x != "0" else "HOSP-UNKNOWN"
)
physician_id = identifiers["AT_PHYSN_NPI"].apply(
    lambda x: f"DR-{str(x)[-6:]}" if x != "0" else "DR-UNKNOWN"
)

# ── Assemble the export DataFrame ────────────────────────────────────────────
export_df = pd.DataFrame({

    # ── IDENTIFIERS ──────────────────────────────────────────────────────────
    "PATIENT_ID"         : patient_ids,          # Unique row key for Power BI
    "BENE_ID"            : bene_ids,             # Original Beneficiary ID from features.csv
    "HOSPITAL_ID"        : hospital_id,           # ORG_NPI_NUM masked (4,902 hospitals)
    "PHYSICIAN_ID"       : physician_id,          # AT_PHYSN_NPI masked (2,463 physicians)

    # ── RISK OUTPUT  (what the AI model produces) ─────────────────────────────
    "RISK_SCORE"         : y_prob.round(4),       # 0.0000 – 1.0000
    "RISK_SCORE_PCT"     : risk_score_pct,         # 0.0 – 100.0  (for gauge charts)
    "RISK_TIER"          : risk_tiers,             # Low / Moderate / High / Critical
    "TIER_PRIORITY"      : tier_priority,          # 1=Critical 2=High 3=Moderate 4=Low
    "ALERT_FLAG"         : alert_flags,            # e.g. "🚨 ALERT — Intervene before discharge"
    "RECOMMENDED_ACTION" : actions,               # Plain-English action for care team

    # ── ACTUAL OUTCOME (ground truth — for model validation in Power BI) ──────
    "READMIT_ACTUAL"     : y,                      # 1 = readmitted  0 = not
    "READMIT_LABEL"      : readmit_label.values,   # "Readmitted" / "No Readmission"

    # ── TEMPORAL ──────────────────────────────────────────────────────────────
    "ADMIT_YEAR"         : df["ADMIT_YEAR"].astype(int),
    "ADMIT_QUARTER"      : df["ADMIT_QUARTER"].astype(int),
    "ADMIT_DOW_NUM"      : df["ADMIT_DOW"].astype(int),   # 0=Mon … 6=Sun
    "ADMIT_DAY"          : dow_label.values,
    "ADMIT_TIMING"       : weekend_label.values,          # Weekday / Weekend

    # ── DEMOGRAPHICS ─────────────────────────────────────────────────────────
    "AGE"                : df["AGE"].astype(int),
    "AGE_GROUP"          : age_group,
    "GENDER"             : gender_label.values,
    "DUAL_ELIGIBLE_LABEL": dual_label.values,

    # ── CLINICAL ─────────────────────────────────────────────────────────────
    "ADMIT_TYPE"         : admit_type_label.values,       # Emergency / Urgent / Elective
    "LOS_DAYS"           : df["LOS_DAYS"].round(1),
    "LOS_CATEGORY"       : los_category.values,
    "FRAILTY_SCORE"      : df["FRAILTY_SCORE"].astype(int),
    "FRAILTY_LABEL"      : frailty_label.values,
    "SOCIAL_RISK"        : df["SOCIAL_RISK"].astype(int), # 0=Low 1=High
    "PRIOR_12M_ADMITS"   : df["PRIOR_12M_ADMITS"].round(0).astype(int),
    "COMORBIDITIES"      : comorbidity_strings.values,    # e.g. "Diabetes, CKD"
    "COMORBIDITY_COUNT"  : df["CC_COUNT"].astype(int),

    # ── INDIVIDUAL COMORBIDITY FLAGS (1/0) ─────────────────────────────────
    "CC_DIABETES"        : df["CC_DIABETES"].astype(int),
    "CC_CKD"             : df["CC_CKD"].astype(int),
    "CC_CHF"             : df["CC_CHF"].astype(int),
    "CC_COPD"            : df["CC_COPD"].astype(int),
    "CC_ALZHEIMERS"      : df["CC_ALZHEIMERS"].astype(int),

    # ── FINANCIAL ────────────────────────────────────────────────────────────
    "TOTAL_CHARGE_USD"   : df["CLM_TOT_CHRG_AMT"].round(2),
    "COINSURANCE_USD"    : df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"].round(2),

    # ── PROVIDER ─────────────────────────────────────────────────────────────
    "STATE_CODE"         : df["PRVDR_STATE_CD"].astype(int),
})

path1 = os.path.join(OUT_DIR, "powerbi_patients.csv")
export_df.to_csv(path1, index=False)
print(f"  ✓ Saved: {path1}  ({len(export_df):,} rows × {len(export_df.columns)} columns)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: PHYSICIAN SUMMARY TABLE  (powerbi_physician_summary.csv)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Building physician summary table ...")

export_df["_PHYSICIAN_ID_RAW"] = identifiers["AT_PHYSN_NPI"].values

phys_summary = export_df.groupby("PHYSICIAN_ID").agg(
    Total_Patients      = ("PATIENT_ID",    "count"),
    Readmissions        = ("READMIT_ACTUAL","sum"),
    Readmit_Rate_Pct    = ("READMIT_ACTUAL","mean"),
    Avg_Risk_Score      = ("RISK_SCORE",    "mean"),
    Critical_Patients   = ("TIER_PRIORITY", lambda x: (x == 1).sum()),
    High_Patients       = ("TIER_PRIORITY", lambda x: (x == 2).sum()),
    Avg_LOS_Days        = ("LOS_DAYS",      "mean"),
    Avg_Charge_USD      = ("TOTAL_CHARGE_USD","mean"),
    Emergency_Patients  = ("ADMIT_TYPE",    lambda x: (x == "Emergency").sum()),
    States_Covered      = ("STATE_CODE",    "nunique"),
).reset_index()

phys_summary["Readmit_Rate_Pct"] = (phys_summary["Readmit_Rate_Pct"] * 100).round(1)
phys_summary["Avg_Risk_Score"]   = phys_summary["Avg_Risk_Score"].round(4)
phys_summary["Avg_LOS_Days"]     = phys_summary["Avg_LOS_Days"].round(1)
phys_summary["Avg_Charge_USD"]   = phys_summary["Avg_Charge_USD"].round(0).astype(int)
phys_summary["Critical_Pct"]     = (
    phys_summary["Critical_Patients"] / phys_summary["Total_Patients"] * 100
).round(1)

# Risk flag for physician — is their panel above average?
avg_rate = export_df["READMIT_ACTUAL"].mean() * 100
phys_summary["Panel_Risk_Flag"] = phys_summary["Readmit_Rate_Pct"].apply(
    lambda r: "Above Average" if r > avg_rate else "Below Average"
)

phys_summary = phys_summary.sort_values("Critical_Patients", ascending=False)

path2 = os.path.join(OUT_DIR, "powerbi_physician_summary.csv")
phys_summary.to_csv(path2, index=False)
print(f"  ✓ Saved: {path2}  ({len(phys_summary):,} physicians)")
print(f"\n  Top 5 physicians by critical patient volume:")
cols_show = ["PHYSICIAN_ID","Total_Patients","Readmit_Rate_Pct","Critical_Patients","Critical_Pct","Avg_Risk_Score"]
print(phys_summary[cols_show].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: HOSPITAL SUMMARY TABLE  (powerbi_hospital_summary.csv)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Building hospital summary table ...")

hosp_summary = export_df.groupby("HOSPITAL_ID").agg(
    Total_Patients      = ("PATIENT_ID",     "count"),
    Readmissions        = ("READMIT_ACTUAL", "sum"),
    Readmit_Rate_Pct    = ("READMIT_ACTUAL", "mean"),
    Avg_Risk_Score      = ("RISK_SCORE",     "mean"),
    Critical_Patients   = ("TIER_PRIORITY",  lambda x: (x == 1).sum()),
    High_Patients       = ("TIER_PRIORITY",  lambda x: (x == 2).sum()),
    Avg_LOS_Days        = ("LOS_DAYS",       "mean"),
    Avg_Charge_USD      = ("TOTAL_CHARGE_USD","mean"),
    Unique_Physicians   = ("PHYSICIAN_ID",   "nunique"),
    State_Code          = ("STATE_CODE",     "first"),
    Emergency_Admits    = ("ADMIT_TYPE",     lambda x: (x == "Emergency").sum()),
).reset_index()

hosp_summary["Readmit_Rate_Pct"] = (hosp_summary["Readmit_Rate_Pct"] * 100).round(1)
hosp_summary["Avg_Risk_Score"]   = hosp_summary["Avg_Risk_Score"].round(4)
hosp_summary["Avg_LOS_Days"]     = hosp_summary["Avg_LOS_Days"].round(1)
hosp_summary["Avg_Charge_USD"]   = hosp_summary["Avg_Charge_USD"].round(0).astype(int)
hosp_summary["Critical_Pct"]     = (
    hosp_summary["Critical_Patients"] / hosp_summary["Total_Patients"] * 100
).round(1)
hosp_summary["Est_Preventable_Savings"] = (
    hosp_summary["Critical_Patients"] * 0.5 * 15000
).astype(int)
hosp_summary["Hospital_Volume_Tier"] = pd.cut(
    hosp_summary["Total_Patients"],
    bins=[0, 5, 20, 100, 99999],
    labels=["Small (<5)", "Medium (5-20)", "Large (20-100)", "Major (100+)"]
)

hosp_summary = hosp_summary.sort_values("Total_Patients", ascending=False)

path3 = os.path.join(OUT_DIR, "powerbi_hospital_summary.csv")
hosp_summary.to_csv(path3, index=False)
print(f"  ✓ Saved: {path3}  ({len(hosp_summary):,} hospitals)")
print(f"\n  Top 5 hospitals by patient volume:")
cols_show2 = ["HOSPITAL_ID","Total_Patients","Readmit_Rate_Pct","Critical_Patients","Avg_Charge_USD","Est_Preventable_Savings"]
print(hosp_summary[cols_show2].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: PRINT POWER BI SETUP GUIDE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  EXPORT COMPLETE")
print("=" * 72)
print(f"""
  THREE FILES FOR POWER BI:
  ─────────────────────────────────────────────────────────────────────
  1. powerbi_patients.csv          ({len(export_df):,} rows · {len(export_df.columns)} columns)
     → Main fact table. One row per patient encounter.
     → Load as your PRIMARY table in Power BI.
     → Key column: PATIENT_ID (unique key)
     → AI output: RISK_SCORE, RISK_TIER, ALERT_FLAG, RECOMMENDED_ACTION

  2. powerbi_physician_summary.csv  ({len(phys_summary):,} physicians)
     → Dimension table. One row per attending physician.
     → Relationship: PHYSICIAN_ID links to patients table
     → Use for: "Which doctors have the highest-risk patient panels?"

  3. powerbi_hospital_summary.csv   ({len(hosp_summary):,} hospitals)
     → Dimension table. One row per hospital/facility.
     → Relationship: HOSPITAL_ID links to patients table
     → Use for: "Which facilities have highest readmission burden?"

  POWER BI: HOW TO LOAD THESE FILES
  ─────────────────────────────────────────────────────────────────────
  Home → Get Data → Text/CSV → select each file
  Then: Model view → drag PHYSICIAN_ID to link patients ↔ physician table
                   → drag HOSPITAL_ID  to link patients ↔ hospital table

  POWER BI: KEY DAX MEASURES TO CREATE
  ─────────────────────────────────────────────────────────────────────
  Readmit Rate %  = DIVIDE(SUM([READMIT_ACTUAL]), COUNTROWS(patients))
  Avg Risk Score  = AVERAGE([RISK_SCORE])
  Critical Count  = CALCULATE(COUNTROWS(patients), [RISK_TIER]="Critical")
  Est Savings     = [Critical Count] * 0.5 * 15000

  COLUMN DESCRIPTIONS (for non-technical managers)
  ─────────────────────────────────────────────────────────────────────
  PATIENT_ID        Unique ID per encounter (PAT-YEAR-NUMBER)
  HOSPITAL_ID       Hospital provider code (4,902 unique facilities)
  PHYSICIAN_ID      Attending physician code (2,463 unique doctors)
  RISK_SCORE        AI probability 0.00–1.00 (higher = more likely to be readmitted)
  RISK_SCORE_PCT    Same as above scaled to 0–100% (easier for gauges)
  RISK_TIER         Low / Moderate / High / Critical
  TIER_PRIORITY     1=Critical (most urgent) to 4=Low (least urgent)
  ALERT_FLAG        Plain-English alert message for clinical staff
  RECOMMENDED_ACTION What care team should do before patient leaves hospital
  READMIT_ACTUAL    Did the patient actually return within 30 days? (1=Yes 0=No)
  COMORBIDITIES     Comma-separated list of active conditions (e.g. "Diabetes, CKD")
  LOS_CATEGORY      Same-Day / Short Stay / Medium Stay / Long Stay
  DUAL_ELIGIBLE     Medicare Only vs Dual (Medicare + Medicaid)

  NOTE ON PATIENT_ID:
  ─────────────────────────────────────────────────────────────────────
  This is a CMS research dataset — real beneficiary IDs are removed
  for HIPAA compliance. PATIENT_ID is a generated sequential key
  (PAT-YEAR-ROWNUM) so Power BI has a unique identifier per row.

  In a real hospital deployment, replace PATIENT_ID with the hospital's
  MRN (Medical Record Number) from the EHR system (Epic, Cerner, etc.)
  and PHYSICIAN_ID / HOSPITAL_ID with actual staff/facility IDs.
""")
