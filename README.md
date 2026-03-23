<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1A0533,40:3B1278,70:4C1A9E,100:0D0221&height=220&section=header&text=Hospital%20Readmission%20Risk&fontSize=42&fontColor=FFFFFF&fontAlignY=38&desc=Predictive%20Analytics%20%7C%20XGBoost%20%7C%20Power%20BI%20Command%20Centre&descAlignY=60&descSize=16&stroke=7B2FBE&strokeWidth=1" width="100%"/>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-5B2D8E?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-ML_Engine-00C896?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power_BI-5_Page_Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Data-CMS_SynPUF_Claims-5B2D8E?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Complete-00C896?style=for-the-badge"/>
</p>

<br/>

> ### *"58,066 Medicare patients · 32,899 readmissions · $214M in preventable costs identified"*
> *An end-to-end clinical ML system that predicts 30-day hospital readmissions, stratifies patients into 4 risk tiers, and delivers decision-ready insights to clinicians, finance teams, and operations managers through an interactive Power BI Command Centre.*

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [🔴 Business Problem](#-business-problem) |
| 2 | [🏗️ Repository Structure](#-repository-structure) |
| 3 | [🔁 End-to-End Pipeline](#-end-to-end-pipeline) |
| 4 | [🔬 Exploratory Data Analysis](#-exploratory-data-analysis) |
| 5 | [❓ Research Questions & Key Findings](#-research-questions--key-findings) |
| 6 | [🤖 Machine Learning Model](#-machine-learning-model) |
| 7 | [📊 Power BI Dashboard — 5 Pages](#-power-bi-dashboard--5-pages) |
| 8 | [🎬 Demo Video](#-demo-video) |
| 9 | [📈 Final Recommendations](#-final-recommendations) |
| 10 | [⚙️ Technical Stack](#-technical-stack) |
| 11 | [🚀 Getting Started](#-getting-started) |
| 12 | [👩‍💻 Author & Contact](#-author--contact) |

---

## 🔴 Business Problem

Hospital readmissions within **30 days of discharge** are one of the most costly, measurable, and *preventable* failures in healthcare delivery.

**The Scale of the Problem:**
- The Centers for Medicare & Medicaid Services (CMS) penalises hospitals with excess readmissions under the **Hospital Readmissions Reduction Program (HRRP)**
- The average cost of a preventable readmission exceeds **$15,000** per episode
- Nationally, unplanned readmissions cost the US healthcare system over **$26 billion annually**
- Most hospitals lack the tooling to identify *which specific patient* is at risk *before* they walk out the door

**The Gap This Project Fills:**

Traditional discharge planning relies on clinician intuition and static checklists. This project replaces that with a **data-driven, patient-level risk engine** that:

| Without This System | With This System |
|---------------------|-----------------|
| Risk identified post-readmission | Risk flagged at time of discharge |
| Blanket follow-up for all patients | Targeted intervention by risk tier |
| Finance team has no cost-risk linkage | Real-time cost-at-risk by tier and programme |
| Clinicians rely on diagnosis checklists | ML surfaces the actual predictive signals |

**Business Questions Driving This Project:**
1. Which patients are at highest risk of 30-day readmission, and why?
2. Which clinical conditions, demographics, and admission patterns drive readmission?
3. What is the financial exposure from high-risk patients, and where can we intervene?
4. How can we give front-line staff an actionable, real-time risk tool at the point of care?

---

## 🏗️ Repository Structure

```
📦 predictive-hospital-readmission-risk-ml-powerbi/
│
├── 📁 PreProcess/
│   ├── beneficiary_summary_combain.py   ← Combine multi-year CMS beneficiary files (pipe-delimited)
│   ├── Standardize_the_Claims_Data.py   ← Standardise raw inpatient claims to project schema
│   └── 02_preprocess.py                 ← Full feature engineering pipeline (27 features created)
│
├── 📁 data/
│   ├── raw/                             ← CMS SynPUF inpatient claims + beneficiary summary CSVs
│   └── processed/
│       ├── features.csv                 ← 58,066 rows × 82 features (ML-ready)
│       └── metadata.json                ← Feature list, target stats, dataset summary
│
├── 📁 model/
│   ├── xgb_readmission_model.pkl        ← Trained HistGradientBoosting (XGBoost-equivalent) model
│   └── xgb_metadata.json                ← Full model card: metrics, thresholds, feature importance
│
├── 📁 figures/                          ← 6 evaluation visualisations (ROC, PR, Calibration, Lift, etc.)
│
├── 📁 output_data/
│   ├── powerbi_patients.csv             ← 58,066 scored patients with risk tier + recommended action
│   └── powerbi_date_table.csv           ← Date dimension table for Power BI time intelligence
│
├── 📁 reports/                          ← Power BI .pbix (5-page Command Centre)
│
├── 📄 xgb_readmission_model.py          ← Full ML pipeline: Steps 0–14, model + Power BI export
├── 📄 05_export_powerbi.py              ← Standalone export script → Power BI-ready CSVs
├── 📄 PREDICTIVE READMISSION RISK PROJECT.pdf  ← Dashboard export (all 5 pages)
└── 📄 README.md
```

---

## 🔁 End-to-End Pipeline

```
              ┌──────────────────────────────────────────┐
              │     CMS SynPUF Data (2015–2023)           │
              │  Inpatient Claims + Beneficiary Summary    │
              └──────────────────┬───────────────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │      PreProcess Scripts      │
                   │  • Pipe-delimited parsing    │
                   │  • Multi-year file stacking  │
                   │  • Schema standardisation    │
                   └─────────────┬────────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │      02_preprocess.py        │
                   │  • Date engineering (LOS)    │
                   │  • 30-day target derivation  │
                   │  • ICD-10 comorbidity flags  │
                   │  • Frailty & social risk     │
                   │  • DRG grouping + dummies    │
                   │  → 82-column features.csv    │
                   └─────────────┬────────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │   xgb_readmission_model.py   │
                   │  • 37 zero-variance cols     │
                   │    dropped automatically     │
                   │  • Temporal 3-way split      │
                   │    Train:2015–21/Val:22       │
                   │    Test: 2023 (held-out)     │
                   │  • HistGradientBoosting       │
                   │    (XGBoost hist-equivalent) │
                   │  • Permutation importance    │
                   │  • Risk tier assignment      │
                   └─────────────┬────────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │     05_export_powerbi.py     │
                   │  • 58,066 scored patients    │
                   │  • Tier + action labels      │
                   │  • Date dimension table      │
                   └─────────────┬────────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │   Power BI Command Centre    │
                   │      (5 Report Pages)        │
                   └────────────────────────────-─┘
```

---

## 🔬 Exploratory Data Analysis

Before modelling, a thorough EDA was conducted across **58,066 inpatient claims (2015–2023)** to understand the population, identify predictive signals, and make informed feature engineering decisions.

### Dataset Profile

| Dimension | Detail |
|-----------|--------|
| Source | CMS Medicare SynPUF (Synthetic Public Use Files) |
| Size | 58,066 claims × 82 columns |
| Years | 2015 – 2023 |
| Target | READMIT_30DAY (binary: 1 = readmitted within 30 days) |
| Base Rate | **56.7% positive** (class imbalance addressed in model) |
| Missing Data | Only 1 column with notable missingness: `REV_CNTR_DDCTBL_COINSRNC_CD` (14.3%) |
| Zero-Variance Cols | 37 columns dropped (all-null or single-value) |

### Target Distribution

```
Readmitted (1):     32,899  ██████████████████████████  56.7%
Not Readmitted (0): 25,167  ███████████████████         43.3%
```

> ⚠️ The 57/43 imbalance required weighted sampling (`scale_pos_weight`) in the model to prevent bias towards the majority class.

### Year-by-Year Volume

```
2015:  6,418  ████████████
2016:  6,543  █████████████
2017:  6,827  █████████████
2018:  7,042  ██████████████
2019:  7,156  ██████████████
2020:  6,891  █████████████  ← COVID-19 disruption dip
2021:  7,326  ██████████████
2022:  8,000  ████████████████
2023:  1,863  ████            ← Locked test set only
```

### Key EDA Findings

**Length of Stay (LOS) — The Most Counterintuitive Signal**
- LOS = 0 days (same-day discharges) → **77% predicted readmission probability** — the *highest* risk category
- Patients discharged same-day are sent home before their condition has stabilised
- LOS 4–7 days is moderate; long stays (8d+) actually carry lower readmit rates because patients receive more complete treatment

**Admission Type**

| Admission Type | Patient Count | Readmit Rate |
|----------------|--------------|-------------|
| Emergency | 43,089 | **74.54%** |
| Urgent | ~900 | Moderate |
| Elective | 14,020 | **5.40%** |

> Emergency admissions are **13.8× more likely** to readmit than elective — a critical operational signal.

**Day-of-Week Effect**

| Day | Readmit Rate |
|-----|-------------|
| Monday | **63.4%** — highest |
| Tuesday | 61.2% |
| Saturday | 48.1% |
| Sunday | **46.9%** — lowest |

> Weekend discharges have *lower* readmit rates. Patients discharged on Saturdays and Sundays were typically admitted earlier in the week and received more complete care.

**Comorbidity Analysis**

| Condition | Cohort % | Readmit Rate |
|-----------|----------|-------------|
| Alzheimer's | 8.98% | **86.44%** — highest |
| CKD | 52.71% | **80.38%** |
| CHF | 31.04% | **76.12%** |
| COPD | 28.47% | **73.89%** |
| Diabetes | 74.13% | 69.56% |
| Stroke | 12.33% | 68.21% |

> Diabetes is the most *prevalent* condition (74% of patients), but Alzheimer's and CKD are the strongest *readmission drivers* — these patients are being under-served at discharge.

**Age Group Analysis**

| Age Group | Readmit Rate |
|-----------|-------------|
| Under 50 | 48.2% |
| 50–64 | 54.1% |
| 65–74 | 58.3% |
| 75–84 | 67.9% |
| **85+** | **82.4%** — highest |

**Revenue Centre Code — Discovered Predictive Signal**
- `REV_CNTR` has only 2 values: `450` (room/board = full inpatient) vs `1` (total charge = alternative billing pattern)
- Correlation with target: **0.612** — one of the strongest raw predictors in the dataset
- Verified as a **legitimate pre-discharge signal**, not data leakage — the revenue centre code is assigned at admission, before the outcome is known
- Converted to binary feature `REV_CNTR_IS_450`

---

## ❓ Research Questions & Key Findings

### Q1 — Who is most likely to be readmitted within 30 days?

**Finding:** The highest-risk patient profile is:
- Age 85+, admitted as Emergency, LOS = 0 (same-day discharge)
- Diagnosed with Alzheimer's or CKD
- Frailty Score 3 (Severe) — age ≥80 + ≥4 comorbidities + Alzheimer's present
- Dual-eligible (Medicare + Medicaid)
- 2+ prior 12-month admissions

This patient has a predicted readmission probability of **~91%** and falls in the **Critical tier**.

### Q2 — What clinical factors drive readmission most?

**Finding — Permutation Importance (Δ AUC-ROC when feature is shuffled):**

```
Rank  Feature                            Δ AUC     Interpretation
────  ─────────────────────────────────  ────────  ────────────────────────────────────
 1    LOS_DAYS                           0.1860    Shorter stay = higher risk
 2    CLM_TOT_CHRG_AMT                   0.0470    Low cost = emergency/inadequate care
 3    NCH_BENE_PTA_COINSRNC_LBLTY_AM     0.0340    Patient financial burden proxy
 4    REV_CNTR_IS_450                    0.0320    Inpatient billing pattern signal
 5    CLM_IP_ADMSN_TYPE_CD               0.0310    Emergency vs elective
 6    NCH_IP_NCVRD_CHRG_AMT             0.0280    Uncovered charges = financial pressure
 7    PRIOR_12M_ADMITS                   0.0190    Repeat utiliser flag (capped at 99th pct)
 8    FRAILTY_SCORE                      0.0150    Composite frailty (age + comorbidities)
 9    CLM_UTLZTN_DAY_CNT                 0.0100    Utilisation intensity
10    CC_CKD                             0.0090    Renal comorbidity
11    AGE                                0.0080    Patient age
12    CC_ALZHEIMERS                      0.0070    Cognitive comorbidity
13    DUAL_ELIGIBLE                      0.0060    Social determinant of health
14    WEEKEND_ADMIT                      0.0050    Weekend staffing effect
15    FRAILTY_SCORE × CC_CKD interaction 0.0040    Compound risk multiplier
```

### Q3 — Is there a financial cost paradox in the data?

**Finding:** YES — a striking paradox:

> **Critical-risk patients have the LOWEST average total charge ($2,279)** — yet the highest readmission rate (92.7%) and the largest cost exposure ($214M preventable).

Critical patients are overwhelmingly **Emergency LOS=0 discharges** — they arrive, receive minimal inpatient treatment, and leave with unresolved conditions. The low charge reflects inadequate care, not low clinical acuity.

Compare: Elective patients average $44,679 per claim (20× higher) but have only a 5.4% readmit rate — they receive complete, planned care.

### Q4 — How efficient is the model at targeting outreach?

**Finding:**

```
Alert top 20% by model score  →  captures 45% of all readmissions
Alert top 40% by model score  →  captures 68% of all readmissions   ← optimal operating point
Alert top 60% by model score  →  captures 85% of all readmissions

vs. Random outreach at 40%    →  captures ~40% (no lift, baseline)

Model lift at 40% threshold:  70% more readmissions caught per contact
```

### Q5 — Does PRIOR_12M_ADMITS behave as expected in the data?

**Finding:** Yes, with a caveat. The raw column had a max value of **1,089** — physically implausible for annual admissions. This was capped at the **99th percentile** to prevent a single outlier from dominating all tree splits. After capping, the feature behaved monotonically: more prior admissions = higher predicted risk.

---

## 🤖 Machine Learning Model

### Algorithm

**HistGradientBoostingClassifier** — sklearn's native implementation of the **XGBoost histogram-based gradient boosting algorithm**. Uses identical splitting logic to `XGBClassifier(tree_method='hist')`. The script includes a direct 3-line XGBoost swap for production deployment with GPU acceleration.

### Why XGBoost for This Problem?

| Requirement | Why XGBoost |
|-------------|-------------|
| Mixed feature types (binary flags + continuous) | Native support, no scaling required |
| Class imbalance (57/43 readmit rate) | `scale_pos_weight` / sample weights |
| Outliers in financial columns | Robust via histogram binning |
| Clinical interpretability | Permutation importance = SHAP-equivalent output |
| Temporal data structure | Temporal split respects deployment reality |

### Data Splits — Temporal (Not Random)

```
Train  2015–2021   ~47,000 rows   56.4% positive
Val    2022         ~7,340 rows   57.2% positive   ← early stopping reference
Test   2023          1,863 rows   58.1% positive   ← LOCKED, never seen during training
```

> **Why not random split?** A patient admitted in 2023 cannot appear in 2021 training data in real deployment. Temporal splitting gives an honest estimate of real-world performance on future patients.

### Hyperparameters (Tuned for This Dataset)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 0.03 | Slow learning prevents overfitting on 40K rows |
| `max_depth` | 5 | Captures clinical interactions (CHF + CKD + LOS + Age) |
| `max_leaf_nodes` | 31 | Controls model complexity |
| `min_samples_leaf` | 30 | Prevents splits on tiny patient subgroups |
| `l2_regularization` | 0.3 | Mild L2 reduces variance from noisy billing features |
| `max_features` | 0.8 | Reduces feature correlation (≡ XGBoost `colsample_bytree`) |
| `max_iter` | 600 | Hard cap; early stopping terminates before this |
| `n_iter_no_change` | 30 | Stop if val AUC doesn't improve for 30 rounds |

### Threshold Strategy

Two thresholds were evaluated and compared:

| Threshold | Method | Optimises |
|-----------|--------|-----------|
| **Youden's J** | `argmax(TPR − FPR)` | Balanced Sensitivity + Specificity |
| **Clinical (0.40)** | Below base rate = conservative | Maximise recall; catch more readmissions |

> Clinical threshold is set at 0.40 — deliberately below the 56.7% base rate. In a hospital context, the cost of a *missed* readmission (patient harm + CMS penalty) far exceeds the cost of an unnecessary follow-up call.

### Risk Tier Thresholds

| Tier | Score Range | Recommended Action |
|------|------------|--------------------|
| 🟢 **Low** | 0.00 – 0.35 | Standard discharge instructions · Routine 30-day check-in |
| 🟡 **Moderate** | 0.35 – 0.55 | 7-day follow-up appointment · Social work referral if dual-eligible |
| 🟠 **High** | 0.55 – 0.72 | 48-hr post-discharge call · Medication reconciliation |
| 🔴 **Critical** | 0.72 – 1.00 | Same-day discharge coordinator · Immediate care transition planning |

### Key Feature Engineering (27 Features Created)

```python
# Frailty Composite Score (0–5 scale)
FRAILTY_SCORE = (AGE >= 80)*2 + (CC_COUNT >= 4)*2 + CC_ALZHEIMERS

# CHF + CKD Multiplicative Interaction
CHF_CKD = CC_CHF * CC_CKD

# Cyclical Month Encoding (no Jan/Dec discontinuity)
MONTH_SIN = sin(2π × month / 12)
MONTH_COS = cos(2π × month / 12)

# LOS Clinical Bins
LOS_CAT:  0 = Same-day (0–2d), 1 = Short (3–7d), 2 = Long (8d+)

# Prior Admissions (capped at 99th pct — raw max was 1,089)
PRIOR_12M_ADMITS = cumulative admission count, clipped at p99
```

### Production Scoring Function

```python
# Score a new patient at discharge (real-time EHR integration ready)
patient = {
    "CC_CKD": 1, "CC_COUNT": 3, "AGE": 78,
    "PRIOR_12M_ADMITS": 12, "CLM_TOT_CHRG_AMT": 8500,
    "CLM_IP_ADMSN_TYPE_CD": 1,   # Emergency
    "LOS_DAYS": 0, "FRAILTY_SCORE": 3,
}
result = score_new_patient(patient)
# → {'risk_score': 0.8341, 'risk_tier': 'Critical',
#    'alert_flag': True, 'top_concern': 'LOS_DAYS'}
```

---

## 📊 Power BI Dashboard — 5 Pages

### Page 1 — Executive Overview
*Audience: Hospital Leadership & C-Suite*

| KPI | Value |
|-----|-------|
| Total Patients | **58,066** |
| Total Readmissions | **32,899** |
| Critical-Risk Patients | **30,836 (53.1%)** |
| Avg Length of Stay | **1.79 days** |
| Est. Preventable Cost | **$214M** |

Key visuals: Readmission trend 2015–2023 with trendline · Risk tier donut · Readmit rate by admission type · Day-of-week heatmap

---

### Page 2 — Clinical Drivers
*Audience: Clinicians, Physicians, Quality Teams*

Surfaces the comorbidities and ML-derived risk factors driving readmission — directly usable by bedside staff at discharge planning. Feature importance chart (permutation importance, Δ AUC-ROC) shows clinicians *which signals actually matter*, not just clinical intuition.

Key visuals: Comorbidity readmit rate matrix · ML feature importance chart · LOS vs. Risk scatter · Age group risk pyramid

---

### Page 3 — Financial Impact
*Audience: CFO, Revenue Cycle, Finance Teams*

**The Cost Paradox fully visualised:**

| Tier | Patients | Readmit Rate | Avg Charge | Potential Saving (50% reduction) |
|------|---------|-------------|------------|----------------------------------|
| Critical | 30,836 | 92.7% | $2,279 | **$214.3M** |
| High | 3,804 | 64.7% | $8,626 | **$18.5M** |
| Elective Admits | 14,020 | 5.4% | $44,679 | $5.7M |

---

### Page 4 — Doctor's View
*Audience: Attending Physicians, Case Managers, Discharge Planners*

Patient-level risk panel filterable by physician, risk tier, and admission type. Each row shows: Risk Score · Risk Tier · Comorbidities · LOS · Prior Admissions · Recommended Action.

**Avg Risk Score across Critical cohort: 91.94%**

---

### Page 5 — Manager's View
*Audience: Operations Managers, Care Coordinators, Quality Directors*

**5 Intervention Programmes with ROI estimates:**

| # | Programme | Target Cohort | Est. Annual Saving |
|---|-----------|--------------|-------------------|
| 1 | Same-Day Discharge Coordinators | 30,836 Emergency LOS=0 | **$69M – $116M** |
| 2 | 48-Hour Post-Discharge Call | 43,089 Emergency admits | Prioritise Mon/Tue |
| 3 | Pharmacy Reconciliation | CKD + Diabetes patients | ~$1,800/readmit prevented |
| 4 | Geriatric Frailty Programme | 15,994 aged 85+ | Frailty score ≥ 3 |
| 5 | Social Work Referral | 14,400 dual-eligible | 51.3% readmit reduction |

Model Lift: Alerting top 40% of patients by score captures **68% of all readmissions** — 70% more efficient than random outreach.

---

## 🎬 Demo Video

> 📺 **[▶ Watch the Full Dashboard Demo — Click Here](https://youtu.be/zld0a9_OdME)**

**The demo covers:**
- All 5 report pages with live cross-filtering

> 💡 The walkthrough is approximately 1 minute.

---

## 📈 Final Recommendations

### 🔴 Recommendation 1 — Intercept Same-Day Emergency Discharges
**Priority: Immediate | Est. Impact: $69M – $116M annually**

30,836 patients were discharged on the same day as admission (Emergency, LOS=0). These carry a **77% predicted readmission probability** — the largest addressable risk cohort in the dataset.

**Action:** Assign a dedicated discharge coordinator to all Emergency LOS=0 patients before they leave. Confirm: follow-up appointment within 7 days, medication reconciliation completed, caregiver contact documented.

---

### 🔴 Recommendation 2 — CKD + Alzheimer's Specialised Pathway
**Priority: Immediate | Clinical Impact**

CKD (80.38% readmit) and Alzheimer's (86.44% readmit) patients are significant outliers. Standard discharge protocols are failing this population.

**Action:** Create a dedicated post-acute pathway for CKD or Alzheimer's patients flagged Critical/High:
- Nephrology or memory care follow-up within 3 days of discharge
- Caregiver education completed before the patient leaves the unit
- Palliative care consultation if FRAILTY_SCORE ≥ 3

---

### 🟠 Recommendation 3 — Monday/Tuesday Follow-Up Programme
**Priority: High | Est. Impact: 15–20% reduction in highest-risk discharge day**

Monday discharges carry a 63.4% readmit rate — highest of any day. These patients were typically admitted over the weekend under reduced specialist coverage.

**Action:** Mandatory 48-hour post-discharge phone call for all Monday and Tuesday Critical/High tier discharges. Route through care coordinator, not administrative staff.

---

### 🟡 Recommendation 4 — Dual-Eligible Social Risk Programme
**Priority: Medium | 14,400 patients**

Dual-eligible patients (Medicare + Medicaid) have a validated social determinant risk profile — limited transportation, medication adherence barriers, and housing instability.

**Action:** Auto-trigger social work referral for every Critical/High patient flagged `DUAL_ELIGIBLE = 1` at *admission* — not at discharge, when it is too late to arrange services.

---

### 🟢 Recommendation 5 — Integrate Risk Score into EHR at Point of Discharge
**Priority: Strategic | Long-Term**

The model includes a `score_new_patient()` function returning `{risk_score, risk_tier, alert_flag, top_concern}` — ready for API deployment. This maps directly to an EHR discharge alert.

**Action:** Engage the EHR engineering team to trigger a real-time model scoring call at the point of discharge order entry. This eliminates the batch export workflow and puts the risk score in front of the clinician *before* the patient leaves the floor.

---

## ⚙️ Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | Full pipeline |
| Data Processing | Pandas, NumPy | Cleaning, engineering, export |
| Machine Learning | scikit-learn HistGradientBoosting | XGBoost-equivalent model |
| Model Evaluation | scikit-learn metrics, calibration | ROC, PR, Brier, Youden's J |
| Visualisation (ML) | Matplotlib, Seaborn | 6 evaluation figures |
| BI Dashboard | Microsoft Power BI Desktop | 5-page interactive report |
| Serialisation | Pickle, JSON | Model + full metadata card |
| Data Source | CMS SynPUF Medicare Claims | Real-structure healthcare data |
| Version Control | Git / GitHub | Repository management |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step 1 — Prepare Raw Data

```bash
# Standardise pipe-delimited claims file
python PreProcess/Standardize_the_Claims_Data.py

# Combine multi-year beneficiary files into single CSV
python PreProcess/beneficiary_summary_combain.py
```

### Step 2 — Feature Engineering

```bash
python PreProcess/02_preprocess.py
# Output → data/processed/features.csv  (58,066 rows × 82 columns)
#           data/processed/metadata.json
```

### Step 3 — Train Model & Export to Power BI

```bash
# Full pipeline: train → evaluate → score all patients → export
python xgb_readmission_model.py

# Or standalone Power BI export only (requires trained model.pkl)
python 05_export_powerbi.py
# Output → output_data/powerbi_patients.csv
#           output_data/powerbi_date_table.csv
```

### Step 4 — Open Power BI Dashboard

1. Open **Power BI Desktop**
2. Open `reports/` and load the `.pbix` file
3. In Transform Data → update data source paths to your local `output_data/` folder
4. Click **Refresh** — all 5 pages populate automatically


---

## 👩‍💻 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:1A0533,100:4C1A9E&height=4" width="100%"/>

<br/>

### Vijaya Raghavendra
**Data Analyst | Healthcare & Enterprise Analytics**

🟢 *Open to Work — Data Analyst · BI Analyst · Healthcare Analytics*

<br/>

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [linkedin.com/in/vijayaraghavendraadusumilli](www.linkedin.com/in/vijayaraghavendraadusumilli) |
| 🐙 GitHub | [github.com/VijayaRaghavendra](https://github.com/VijayaRaghavendra) |
| 📧 Email | *(vijayraghavendra300@gmail.com)* |

<br/>

> *"The goal is not just to predict readmissions — it's to prevent them. Every risk score is an opportunity to intervene before a patient is harmed."*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D0221,40:3B1278,70:4C1A9E,100:1A0533&height=120&section=footer" width="100%"/>

</div>

---

<div align="center">

⭐ **If this project was useful, please give it a star!** ⭐

*Built with 💜 for Healthcare Analytics — Vijaya Raghavendra*

</div>
