"""
================================================================================
  04_explain_dashboard.py
  PREDICTIVE READMISSION RISK ENGINE  —  Explainability & Clinical Dashboard
  Built on: features.csv  (58,066 real CMS claims, 2015–2023)
================================================================================

WHAT THIS SCRIPT PRODUCES (9 figures total):

  PANEL A — MODEL EXPLAINABILITY
    A1. Global Feature Importance     (permutation Δ AUC bar chart, top 20)
    A2. Partial Dependence Plots      (how each top feature changes risk score)
    A3. Feature Contribution Heatmap  (risk driver heatmap across tiers)

  PANEL B — PATIENT-LEVEL EXPLANATIONS
    B1. Case Study Waterfall ×4       (TP, TN, FP, FN — one panel each)
        Shows per-feature contributions vs. baseline for real patients
    B2. Score Distribution by Tier    (violin + strip plot, actual vs predicted)

  PANEL C — CLINICAL OPERATIONS
    C1. Readmission Calendar          (by month/year — trend analysis)
    C2. Provider State Heatmap        (risk score by state, top 20 states)
    C3. Age–Risk–Comorbidity Bubble   (age vs risk vs comorbidity burden)

  MASTER DASHBOARD (Figure 10)       (all key panels combined, portfolio hero)

TECHNIQUE:
  Local explanations use Conditional Expectation Perturbation (CEP):
    contribution_i = model_score(patient) − model_score(patient with feature_i = train_median)
  This is model-agnostic, works on any black-box estimator, and is the
  same underlying logic as SHAP TreeExplainer but computed without the
  shap library (which is not yet installed in this environment).
  When `pip install shap` is available, the shap_swap() function at the
  bottom shows the 3-line drop-in replacement.

USAGE:
  python 04_explain_dashboard.py
  All figures saved to: outputs/figures/explain/
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os, json, pickle, warnings, time
# Add this line near the top of your script
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION  (edit paths only if your layout differs)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/processed/features.csv"
MODEL_PATH = "outputs/xgb_readmission_model.pkl"
META_PATH  = "outputs/xgb_metadata.json"
FI_PATH    = "outputs/feature_importance.csv"
OUT_DIR    = "outputs/figures/explain"
os.makedirs(OUT_DIR, exist_ok=True)

SEED            = 42
TARGET          = "READMIT_30DAY"
YEAR_COL        = "ADMIT_YEAR"
YOUDEN_THRESH   = 0.438   # from model training output
CLINICAL_THRESH = 0.40

# Colour system — clinical dark theme
C = {
    "bg"     : "#0F1117",
    "panel"  : "#1A1D27",
    "border" : "#2A2D3A",
    "teal"   : "#00B4D8",
    "tealm"  : "#48CAE4",
    "teal2"  : "#ADE8F4",
    "green"  : "#06D6A0",
    "red"    : "#EF476F",
    "amber"  : "#FFB703",
    "purple" : "#7B2FBE",
    "text"   : "#E8EAF0",
    "muted"  : "#6B7280",
    "white"  : "#FFFFFF",
}

TIER_COLORS = {
    "Low"      : C["green"],
    "Moderate" : C["amber"],
    "High"     : "#FF6B35",
    "Critical" : C["red"],
}
TIERS = {
    "Low"      : (0.00, 0.35),
    "Moderate" : (0.35, 0.55),
    "High"     : (0.55, 0.72),
    "Critical" : (0.72, 1.01),
}

# Human-readable labels for feature names
FEAT_LABELS = {
    "LOS_DAYS"                    : "Length of Stay (Days)",
    "CLM_TOT_CHRG_AMT"            : "Total Charge Amount ($)",
    "CLM_LINE_NUM"                : "Claim Line Number",
    "PRIOR_12M_ADMITS"            : "Prior 12-Month Admissions",
    "CLM_UTLZTN_DAY_CNT"          : "Utilization Day Count",
    "AGE"                         : "Patient Age",
    "CLM_IP_ADMSN_TYPE_CD"        : "Admission Type (1=Emerg)",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM": "Coinsurance Liability ($)",
    "CC_COUNT"                    : "Comorbidity Count",
    "CC_DIABETES"                 : "Diabetes (0/1)",
    "NCH_BENE_IP_DDCTBL_AMT"      : "IP Deductible Amount ($)",
    "ADMIT_DOW"                   : "Admit Day of Week",
    "LOS_CAT"                     : "LOS Category (0/1/2)",
    "MALE"                        : "Sex (1=Male)",
    "COST_TIER"                   : "Cost Tier (0/1/2)",
    "FRAILTY_SCORE"               : "Frailty Score",
    "CC_CKD"                      : "Chronic Kidney Disease",
    "DUAL_ELIGIBLE"               : "Dual Medicare/Medicaid",
    "CC_CHF"                      : "Congestive Heart Failure",
    "CC_ALZHEIMERS"               : "Alzheimer's Disease",
    "PRVDR_STATE_CD"              : "Provider State Code",
    "NCH_PRMRY_PYR_CLM_PD_AMT"   : "Primary Payer Paid ($)",
    "CLM_SRC_IP_ADMSN_CD"         : "Admission Source Code",
    "NCH_BENE_IP_DDCTBL_AMT"      : "IP Deductible ($)",
    "NCH_IP_NCVRD_CHRG_AMT"       : "Non-Covered Charge ($)",
    "BENE_TOT_COINSRNC_DAYS_CNT"  : "Coinsurance Days Count",
    "CLM_DRG_OUTLIER_STAY_CD"     : "DRG Outlier Stay Code",
    "REV_CNTR_DDCTBL_COINSRNC_CD" : "Rev Cntr Coin Code",
    "CC_COPD"                     : "COPD (0/1)",
    "CHF_CKD"                     : "CHF × CKD Interaction",
    "SOCIAL_RISK"                 : "Social Risk Index",
    "WEEKEND_ADMIT"               : "Weekend Admission",
    "HIGH_PRIOR_USE"              : "High Prior Utilization",
    "DRG_Infectious"              : "Infectious DRG",
    "DRG_Other"                   : "Other DRG",
    "ADMIT_QUARTER"               : "Admit Quarter",
    "MONTH_SIN"                   : "Month Cyclical (Sin)",
    "MONTH_COS"                   : "Month Cyclical (Cos)",
}

def flabel(f):
    return FEAT_LABELS.get(f, f.replace("_", " ").title())

np.random.seed(SEED)
plt.rcParams.update({
    "figure.facecolor"  : C["bg"],
    "axes.facecolor"    : C["panel"],
    "axes.edgecolor"    : C["border"],
    "axes.labelcolor"   : C["text"],
    "xtick.color"       : C["muted"],
    "ytick.color"       : C["muted"],
    "text.color"        : C["text"],
    "grid.color"        : C["border"],
    "grid.linewidth"    : 0.6,
    "font.family"       : "monospace",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

print("=" * 72)
print("  04_explain_dashboard.py")
print("  XGBoost Readmission Risk Engine — Explainability & Clinical Panels")
print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA + MODEL  (exact same preprocessing as xgb_readmission_model.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LOAD] Data, model, metadata ...")

df = pd.read_csv(DATA_PATH)

zero_var = [c for c in df.columns
            if df[c].nunique(dropna=False) <= 1 or df[c].isnull().mean() == 1.0]
id_cols   = ["ORG_NPI_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI"]
redundant = ["CLM_PMT_AMT"]
drop_all  = list(set(zero_var + id_cols + redundant))
drop_all  = [c for c in drop_all if c in df.columns]
df.drop(columns=drop_all, inplace=True)

df["REV_CNTR_IS_450"] = (df["REV_CNTR"] == 450).astype(int)
df.drop(columns=["REV_CNTR"], inplace=True)
if df["NCH_IP_NCVRD_CHRG_AMT"].equals(df["NCH_IP_TOT_DDCTN_AMT"]):
    df.drop(columns=["NCH_IP_TOT_DDCTN_AMT"], inplace=True)

col = "REV_CNTR_DDCTBL_COINSRNC_CD"
df[col] = df.groupby(YEAR_COL)[col].transform(lambda x: x.fillna(x.median()))
df[col] = df[col].fillna(df[col].median())

df["PRIOR_12M_ADMITS"]            = df["PRIOR_12M_ADMITS"].clip(
    upper=np.percentile(df["PRIOR_12M_ADMITS"], 99))
df["CLM_TOT_CHRG_AMT"]           = df["CLM_TOT_CHRG_AMT"].clip(
    upper=np.percentile(df["CLM_TOT_CHRG_AMT"], 99.5))
df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"] = df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"].clip(
    upper=np.percentile(df["NCH_BENE_PTA_COINSRNC_LBLTY_AM"], 99.5))

with open(META_PATH) as f:
    meta = json.load(f)
FEAT_COLS = meta["feature_names"]   # exact 37 features the model was trained on

X  = df[FEAT_COLS].values.astype(np.float32)
y  = df[TARGET].values.astype(np.int32)
yr = df[YEAR_COL].values

X_train, y_train = X[yr <= 2021], y[yr <= 2021]
X_test,  y_test  = X[yr == 2023], y[yr == 2023]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

y_prob = model.predict_proba(X_test)[:, 1]

fi_df  = pd.read_csv(FI_PATH)
TOP_N  = 10
TOP_FEATS = fi_df["feature"].head(TOP_N).tolist()
TOP_IDX   = [FEAT_COLS.index(f) for f in TOP_FEATS]

# Train medians — used as baseline reference for local explanations
train_medians = np.array([np.median(X_train[:, i]) for i in range(len(FEAT_COLS))],
                          dtype=np.float32)
BASELINE_SCORE = float(model.predict_proba(train_medians.reshape(1, -1))[0, 1])

print(f"  ✓ {len(df):,} records  |  Test set (2023): {len(X_test):,}")
print(f"  ✓ Model: {type(model).__name__}  n_iter={model.n_iter_}")
print(f"  ✓ Baseline score (train median patient): {BASELINE_SCORE:.4f}")
# Access the 'test_metrics' dictionary first, then get 'AUC_ROC'
print(f"  ✓ Test AUC: {meta['test_metrics']['AUC_ROC']:.4f}")

def assign_tier(p):
    for t, (lo, hi) in TIERS.items():
        if lo <= p < hi:
            return t
    return "Critical"

tiers_test = np.array([assign_tier(p) for p in y_prob])


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOCAL EXPLANATION ENGINE
#    Conditional Expectation Perturbation (CEP)
#    contrib_i = score(x) − score(x with feature_i replaced by train_median_i)
#    Positive = feature raises risk above baseline
#    Negative = feature lowers risk below baseline
# ─────────────────────────────────────────────────────────────────────────────
def explain_patient(x_row: np.ndarray, feature_indices: list) -> dict:
    """Return dict of feature → contribution for a single patient row."""
    base = float(model.predict_proba(x_row.reshape(1, -1))[0, 1])
    contribs = {}
    perturbed = x_row.copy()
    for fi_idx in feature_indices:
        orig = perturbed[fi_idx]
        perturbed[fi_idx] = train_medians[fi_idx]
        score_without = float(model.predict_proba(perturbed.reshape(1, -1))[0, 1])
        perturbed[fi_idx] = orig
        contribs[FEAT_COLS[fi_idx]] = base - score_without
    return base, contribs


# ─────────────────────────────────────────────────────────────────────────────
# 4. SELECT REPRESENTATIVE CASE STUDY PATIENTS
# ─────────────────────────────────────────────────────────────────────────────
def pick_patient(mask, preferred_score_fn=None):
    """Pick a clean, illustrative patient index from the test set."""
    candidates = np.where(mask)[0]
    if preferred_score_fn is not None:
        candidates = sorted(candidates, key=lambda i: preferred_score_fn(y_prob[i]))
    return candidates[0] if len(candidates) > 0 else None

tp_mask = (y_prob > 0.88) & (y_test == 1)
tn_mask = (y_prob < 0.06) & (y_test == 0)
fp_mask = (y_prob > 0.88) & (y_test == 0)
fn_mask = (y_prob < 0.15) & (y_test == 1)

tp_idx = pick_patient(tp_mask, lambda s: -s)
tn_idx = pick_patient(tn_mask, lambda s: s)
fp_idx = pick_patient(fp_mask, lambda s: -s)
fn_idx = pick_patient(fn_mask, lambda s: -s)

# Fall back if any tier is empty
for label, idx in [("TP", tp_idx), ("TN", tn_idx), ("FP", fp_idx), ("FN", fn_idx)]:
    print(f"  Case {label}: test_idx={idx}  score={y_prob[idx]:.4f}  actual={y_test[idx]}")

CASES = {
    "True Positive\n(Correctly Flagged)"     : (tp_idx, C["red"]),
    "True Negative\n(Correctly Cleared)"     : (tn_idx, C["green"]),
    "False Positive\n(Over-alerted)"         : (fp_idx, C["amber"]),
    "False Negative\n(Missed Readmission)"   : (fn_idx, C["purple"]),
}

print()


# ─────────────────────────────────────────────────────────────────────────────
# A1. GLOBAL FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("[A1] Global Feature Importance ...")

fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["panel"])

top20 = fi_df.head(20).copy()
top20["label"] = top20["feature"].apply(flabel)

# Normalised importance for colour mapping
norm_imp = top20["mean"] / top20["mean"].max()
bar_colors = [
    C["red"] if v > 0.8 else C["amber"] if v > 0.4 else C["teal"]
    for v in norm_imp[::-1]
]

bars = ax.barh(
    top20["label"][::-1],
    top20["mean"][::-1],
    xerr=top20["std"][::-1],
    color=bar_colors,
    edgecolor=C["bg"],
    linewidth=0.8,
    alpha=0.92,
    capsize=3,
    error_kw={"elinewidth": 1.2, "ecolor": C["muted"], "alpha": 0.7},
    height=0.72,
)

# Value labels
for bar, val, std in zip(bars, top20["mean"][::-1], top20["std"][::-1]):
    ax.text(
        bar.get_width() + std + 0.0012,
        bar.get_y() + bar.get_height() / 2,
        f"Δ{val:.4f}",
        va="center", ha="left", fontsize=8.5,
        color=C["muted"], fontfamily="monospace",
    )

# Rank numbers on left
for i, label in enumerate(top20["label"][::-1]):
    rank = 20 - i
    ax.text(-0.0045, i, f"#{rank}", va="center", ha="right",
            fontsize=7.5, color=C["muted"], fontfamily="monospace")

ax.axvline(0, color=C["border"], lw=1)
ax.set_xlabel("Mean Decrease in AUC-ROC when Feature Permuted\n(15 repeats on 2,000-patient sample  ·  ±1 SD shown)",
              fontsize=10, color=C["muted"], labelpad=8)
ax.set_title("Global Feature Importance  ·  XGBoost Readmission Model",
             fontsize=13, fontweight="bold", color=C["white"], pad=14)

legend_patches = [
    mpatches.Patch(color=C["red"],   label="Dominant  (>80% of max Δ AUC)"),
    mpatches.Patch(color=C["amber"], label="Important (40–80%)"),
    mpatches.Patch(color=C["teal"],  label="Moderate  (<40%)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right",
          facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(left=-0.008)
plt.tight_layout(pad=1.5)
path = os.path.join(OUT_DIR, "A1_feature_importance.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# A2. PARTIAL DEPENDENCE PLOTS  (top 6 features)
# ─────────────────────────────────────────────────────────────────────────────
print("[A2] Partial Dependence Plots ...")

PDP_FEATS = TOP_FEATS[:6]
PDP_N_POINTS = 40
typical = train_medians.copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(C["bg"])
fig.suptitle("Partial Dependence Plots  ·  How Each Feature Drives Risk Score",
             fontsize=13, fontweight="bold", color=C["white"], y=0.98)

pdp_axes = axes.flatten()
for ax_i, feat in enumerate(PDP_FEATS):
    ax   = pdp_axes[ax_i]
    fi_  = FEAT_COLS.index(feat)
    col_vals = X_test[:, fi_]
    uniq_vals = np.unique(col_vals)

    if len(uniq_vals) <= 12:
        sweep = uniq_vals
    else:
        sweep = np.percentile(col_vals, np.linspace(2, 98, PDP_N_POINTS))
        sweep = np.unique(sweep)

    pdp_scores = []
    for v in sweep:
        test_row          = typical.copy()
        test_row[fi_]     = v
        pdp_scores.append(float(model.predict_proba(test_row.reshape(1, -1))[0, 1]))

    pdp_scores = np.array(pdp_scores)

    # Gradient fill under curve
    ax.fill_between(sweep, BASELINE_SCORE, pdp_scores,
                    where=pdp_scores > BASELINE_SCORE,
                    alpha=0.22, color=C["red"], label="_nolegend_")
    ax.fill_between(sweep, BASELINE_SCORE, pdp_scores,
                    where=pdp_scores <= BASELINE_SCORE,
                    alpha=0.22, color=C["green"], label="_nolegend_")

    ax.plot(sweep, pdp_scores, color=C["teal"], lw=2.2, zorder=3)
    ax.axhline(BASELINE_SCORE, color=C["muted"], lw=1.2, ls="--",
               label=f"Baseline ({BASELINE_SCORE:.2f})", alpha=0.7)

    # Mark actual data distribution as rug
    sample_vals = col_vals[np.random.choice(len(col_vals), min(200, len(col_vals)), replace=False)]
    ax.scatter(sample_vals, np.full_like(sample_vals, 0.01),
               c=C["muted"], s=4, alpha=0.35, zorder=2, marker="|")

    ax.set_title(flabel(feat), fontsize=10, fontweight="bold",
                 color=C["white"], pad=6)
    ax.set_xlabel(feat, fontsize=8, color=C["muted"])
    ax.set_ylabel("Risk Score" if ax_i % 3 == 0 else "", fontsize=9, color=C["muted"])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7.5, loc="upper right", facecolor=C["bg"],
              edgecolor=C["border"], labelcolor=C["muted"])

    # Rank badge
    ax.text(0.03, 0.93, f"#{ax_i+1}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=C["teal"],
            va="top", fontfamily="monospace")

plt.tight_layout(pad=1.8)
path = os.path.join(OUT_DIR, "A2_partial_dependence.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# A3. FEATURE CONTRIBUTION HEATMAP BY RISK TIER
#     For each tier, compute average CEP contribution of each top feature
# ─────────────────────────────────────────────────────────────────────────────
print("[A3] Feature Contribution Heatmap by Risk Tier ...")

HMAP_FEATS = TOP_FEATS[:8]
HMAP_IDX   = [FEAT_COLS.index(f) for f in HMAP_FEATS]
TIER_ORDER = ["Low", "Moderate", "High", "Critical"]

# Sample up to 150 patients per tier for speed
tier_contribs = {t: {f: [] for f in HMAP_FEATS} for t in TIER_ORDER}

for t in TIER_ORDER:
    tier_idx_list = np.where(tiers_test == t)[0]
    sampled = tier_idx_list[
        np.random.choice(len(tier_idx_list), min(60, len(tier_idx_list)), replace=False)
    ]
    base_scores = model.predict_proba(X_test[sampled])[:, 1]
    for fi_pos, fi_idx in enumerate(HMAP_IDX):
        feat = HMAP_FEATS[fi_pos]
        perturbed = X_test[sampled].copy()
        perturbed[:, fi_idx] = train_medians[fi_idx]
        perturbed_scores = model.predict_proba(perturbed)[:, 1]
        contributions = base_scores - perturbed_scores
        tier_contribs[t][feat].extend(contributions.tolist())

# Build heatmap matrix [features × tiers]
hmap_matrix = np.array([
    [np.mean(tier_contribs[t][f]) for t in TIER_ORDER]
    for f in HMAP_FEATS
])

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(C["bg"])

# Custom diverging colormap: green→black→red
cmap_div = LinearSegmentedColormap.from_list(
    "cep", [C["green"], C["panel"], C["red"]], N=256
)
v_abs = np.abs(hmap_matrix).max()
im = ax.imshow(hmap_matrix, cmap=cmap_div, aspect="auto",
               vmin=-v_abs, vmax=v_abs)

# Annotate cells
for i in range(len(HMAP_FEATS)):
    for j, t in enumerate(TIER_ORDER):
        val = hmap_matrix[i, j]
        txt_color = C["white"] if abs(val) > v_abs * 0.4 else C["muted"]
        sign = "+" if val > 0 else ""
        ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                fontsize=9, color=txt_color, fontfamily="monospace",
                fontweight="bold" if abs(val) > v_abs * 0.6 else "normal")

ax.set_xticks(range(len(TIER_ORDER)))
ax.set_xticklabels(TIER_ORDER, fontsize=11, fontweight="bold")
for j, t in enumerate(TIER_ORDER):
    ax.get_xticklabels()[j].set_color(TIER_COLORS[t])

ax.set_yticks(range(len(HMAP_FEATS)))
ax.set_yticklabels([flabel(f) for f in HMAP_FEATS], fontsize=10)
ax.set_title("Average Feature Contribution by Risk Tier\n"
             "Positive = raises risk  ·  Negative = lowers risk  ·  CEP method",
             fontsize=12, fontweight="bold", color=C["white"], pad=12)

cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.ax.set_ylabel("Contribution (Δ Risk Score)", fontsize=9, color=C["muted"])
cb.ax.yaxis.set_tick_params(color=C["muted"])
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"])

plt.tight_layout(pad=1.5)
path = os.path.join(OUT_DIR, "A3_contribution_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# B1. PATIENT WATERFALL EXPLANATIONS  ×4 cases
# ─────────────────────────────────────────────────────────────────────────────
print("[B1] Patient-Level Waterfall Explanations ...")

# Explain each case patient using all TOP_N features
WATERFALL_FEATS    = TOP_FEATS[:8]
WATERFALL_IDX      = [FEAT_COLS.index(f) for f in WATERFALL_FEATS]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor(C["bg"])
fig.suptitle("Patient-Level Explanations  ·  Feature Contributions vs Baseline\n"
             "(Positive bar = pushes score UP ↑  ·  Negative bar = pushes score DOWN ↓)",
             fontsize=13, fontweight="bold", color=C["white"], y=0.98)

for ax, (case_label, (pt_idx, case_color)) in zip(axes.flatten(), CASES.items()):
    ax.set_facecolor(C["panel"])

    pt_score, contribs = explain_patient(X_test[pt_idx], WATERFALL_IDX)
    actual = y_test[pt_idx]
    tier   = assign_tier(pt_score)

    # Sort contributions by absolute value
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    features  = [flabel(f) for f, _ in sorted_contribs]
    values    = [v for _, v in sorted_contribs]
    bar_cols  = [C["red"] if v > 0 else C["green"] for v in values]

    # Horizontal waterfall
    bars = ax.barh(features[::-1], values[::-1],
                   color=bar_cols[::-1], edgecolor=C["bg"],
                   linewidth=0.5, alpha=0.88, height=0.65)

    # Value labels on bars
    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width()
        ha = "left" if x_pos >= 0 else "right"
        offset = 0.002 if x_pos >= 0 else -0.002
        sign = "+" if val > 0 else ""
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.3f}", va="center", ha=ha,
                fontsize=8.5, color=C["muted"], fontfamily="monospace")

    ax.axvline(0, color=C["border"], lw=1.5, zorder=5)

    # Feature values annotation on right
    ax_twin = ax.twinx()
    ax_twin.set_facecolor("none")
    for i, (feat, _) in enumerate(sorted_contribs[::-1]):
        fi_ = FEAT_COLS.index(feat)
        raw_val = X_test[pt_idx, fi_]
        ax_twin.text(1.01, (i + 0.5) / len(sorted_contribs), f"val={raw_val:.1f}",
                     transform=ax_twin.transAxes, fontsize=7.5, va="center",
                     color=C["muted"], fontfamily="monospace")
    ax_twin.set_yticks([])
    ax_twin.spines["right"].set_visible(False)

    # Score callout box
    tier_col = TIER_COLORS[tier]
    ax.text(0.98, 0.04,
            f"Score: {pt_score:.3f}\nTier: {tier}\nActual: {'Readmit' if actual else 'No Readmit'}",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            color=tier_col, fontfamily="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["bg"],
                      edgecolor=tier_col, alpha=0.85))

    ax.set_title(case_label, fontsize=11, fontweight="bold",
                 color=case_color, pad=8)
    ax.set_xlabel("Feature Contribution (Δ Risk Score)", fontsize=9, color=C["muted"])
    ax.grid(axis="x", alpha=0.2)

    # Baseline marker
    ax.axvline(0, color=C["border"], lw=1)

plt.tight_layout(pad=1.6, h_pad=2.5, w_pad=2.5)
path = os.path.join(OUT_DIR, "B1_patient_waterfall.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# B2. SCORE DISTRIBUTION  — violin + strip by tier × actual outcome
# ─────────────────────────────────────────────────────────────────────────────
print("[B2] Score Distribution by Tier & Outcome ...")

rdf = pd.DataFrame({
    "score"  : y_prob,
    "tier"   : tiers_test,
    "actual" : y_test,
    "outcome": ["Readmitted" if a else "No Readmit" for a in y_test],
})

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.patch.set_facecolor(C["bg"])
fig.suptitle("Score Distribution  ·  Risk Tier Validation  ·  Test Set 2023",
             fontsize=13, fontweight="bold", color=C["white"], y=1.01)

# Left: box-ish distribution by tier
ax = axes[0]
tier_order = ["Low", "Moderate", "High", "Critical"]
for i, t in enumerate(tier_order):
    scores_t = rdf.loc[rdf["tier"] == t, "score"].values
    if len(scores_t) == 0:
        continue
    # Violin shape via KDE
    from scipy.stats import gaussian_kde
    if len(scores_t) > 5:
        kde    = gaussian_kde(scores_t, bw_method=0.3)
        ys     = np.linspace(scores_t.min(), scores_t.max(), 200)
        widths = kde(ys)
        widths = widths / widths.max() * 0.35
        ax.fill_betweenx(ys, i - widths, i + widths,
                         alpha=0.55, color=TIER_COLORS[t], zorder=2)
        ax.plot(i - widths, ys, lw=1, color=TIER_COLORS[t], alpha=0.8, zorder=3)
        ax.plot(i + widths, ys, lw=1, color=TIER_COLORS[t], alpha=0.8, zorder=3)

    # Jitter strip overlay
    jitter = np.random.uniform(-0.12, 0.12, size=min(len(scores_t), 120))
    sample = np.random.choice(scores_t, min(120, len(scores_t)), replace=False)
    c_strip = [C["red"] if rdf.loc[rdf["tier"]==t,"actual"].values[j] == 1 else C["teal"]
               for j in range(len(sample))]
    ax.scatter(i + jitter[:len(sample)], sample, s=12, c=c_strip, alpha=0.55,
               zorder=4, linewidths=0)

    # Median line
    ax.hlines(np.median(scores_t), i - 0.2, i + 0.2,
              color=C["white"], lw=2.2, zorder=5)

    # Count + readmit rate
    rate = rdf.loc[rdf["tier"] == t, "actual"].mean() * 100
    ax.text(i, -0.07, f"n={len(scores_t)}\n{rate:.0f}%", ha="center",
            fontsize=9, color=TIER_COLORS[t], fontfamily="monospace")

# Tier boundary lines
for lo, _ in list(TIERS.values())[1:]:
    ax.axhline(lo, color=C["border"], lw=0.8, ls=":", alpha=0.7)

ax.set_xticks(range(len(tier_order)))
ax.set_xticklabels(tier_order, fontsize=11, fontweight="bold")
for i, t in enumerate(tier_order):
    ax.get_xticklabels()[i].set_color(TIER_COLORS[t])
ax.set_ylabel("XGBoost Risk Score", fontsize=10, color=C["muted"])
ax.set_title("Score Distribution by Risk Tier", fontsize=11, fontweight="bold",
             color=C["white"])
ax.set_ylim(-0.15, 1.08)
legend_elems = [mpatches.Patch(color=C["red"],  label="Readmitted"),
                mpatches.Patch(color=C["teal"], label="No Readmit")]
ax.legend(handles=legend_elems, fontsize=9, facecolor=C["bg"],
          edgecolor=C["border"], labelcolor=C["text"])
ax.grid(axis="y", alpha=0.2)

# Right: cumulative lift — if we alert top-X% by score, what % of readmits do we capture?
ax = axes[1]
sorted_idx = np.argsort(y_prob)[::-1]
n_total    = len(y_test)
n_readmit  = y_test.sum()

pct_alerted  = np.arange(1, n_total + 1) / n_total * 100
pct_captured = np.cumsum(y_test[sorted_idx]) / n_readmit * 100

# Random baseline
ax.plot([0, 100], [0, 100], color=C["muted"], lw=1.5, ls="--",
        label="Random baseline", alpha=0.6)
ax.plot(pct_alerted, pct_captured, color=C["teal"], lw=2.5,
        label="XGBoost model")
ax.fill_between(pct_alerted, pct_captured, pct_alerted,
                where=pct_captured > pct_alerted, alpha=0.12, color=C["teal"])

# Mark key operating points
for alert_pct, color, label_txt in [(20, C["red"], "20% alerted"),
                                     (40, C["amber"], "40% alerted"),
                                     (60, C["teal"], "60% alerted")]:
    idx_pt = int(alert_pct / 100 * n_total)
    cap_pt = pct_captured[idx_pt]
    ax.scatter(alert_pct, cap_pt, s=80, color=color, zorder=5, marker="o")
    ax.annotate(f"{label_txt}\n→ {cap_pt:.0f}% readmits caught",
                xy=(alert_pct, cap_pt), xytext=(alert_pct + 5, cap_pt - 8),
                fontsize=8, color=color, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=color, lw=1))

ax.set_xlabel("% of Patients Alerted (highest risk first)", fontsize=10, color=C["muted"])
ax.set_ylabel("% of True Readmissions Captured", fontsize=10, color=C["muted"])
ax.set_title("Cumulative Lift Curve\n(Model vs Random Alert Strategy)",
             fontsize=11, fontweight="bold", color=C["white"])
ax.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ax.grid(alpha=0.2); ax.set_xlim(0, 100); ax.set_ylim(0, 105)

plt.tight_layout(pad=1.8)
path = os.path.join(OUT_DIR, "B2_score_distribution_lift.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# C1. READMISSION TREND — by Year × Actual Rate vs Model Score
# ─────────────────────────────────────────────────────────────────────────────
print("[C1] Readmission Trend (all years) ...")

# Score ALL data (not just 2023) using the model
y_prob_all = model.predict_proba(X)[:, 1]
df_all = df[[YEAR_COL, "ADMIT_QUARTER", TARGET]].copy()
df_all["pred_score"] = y_prob_all
df_all["tier"]       = [assign_tier(p) for p in y_prob_all]

yearly = df_all.groupby(YEAR_COL).agg(
    actual_rate = (TARGET, "mean"),
    mean_score  = ("pred_score", "mean"),
    n           = (TARGET, "count"),
    critical_pct= ("pred_score", lambda x: (x > 0.72).mean()),
).reset_index()

# Quarterly trend
quarterly = df_all.groupby([YEAR_COL, "ADMIT_QUARTER"]).agg(
    actual_rate = (TARGET, "mean"),
    mean_score  = ("pred_score", "mean"),
    n           = (TARGET, "count"),
).reset_index()
quarterly["period"] = quarterly[YEAR_COL].astype(str) + "-Q" + quarterly["ADMIT_QUARTER"].astype(str)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.patch.set_facecolor(C["bg"])
fig.suptitle("Readmission Trend Analysis  ·  2015–2023  ·  All Data",
             fontsize=13, fontweight="bold", color=C["white"], y=0.99)

# Annual
ax = axes[0]
x  = np.arange(len(yearly))
w  = 0.35
b1 = ax.bar(x - w/2, yearly["actual_rate"] * 100, w, color=C["red"],
            alpha=0.75, label="Actual Readmit Rate", edgecolor=C["bg"])
b2 = ax.bar(x + w/2, yearly["mean_score"] * 100, w, color=C["teal"],
            alpha=0.75, label="Mean Model Score", edgecolor=C["bg"])

for bar, n in zip(b1, yearly["n"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"n={n:,}", ha="center", fontsize=7.5, color=C["muted"], fontfamily="monospace")

ax2 = ax.twinx()
ax2.plot(x, yearly["critical_pct"] * 100, "o--", color=C["amber"],
         lw=2, ms=6, label="% Critical-Tier Patients", zorder=5)
ax2.set_ylabel("% Critical-Tier", fontsize=9, color=C["amber"])
ax2.tick_params(axis="y", colors=C["amber"])
ax2.spines["right"].set_edgecolor(C["amber"])
ax2.set_facecolor("none")

ax.set_xticks(x)
ax.set_xticklabels([str(int(y)) for y in yearly[YEAR_COL]], fontsize=10)
ax.set_ylabel("Rate / Score (%)", fontsize=10, color=C["muted"])
ax.set_title("Annual Readmission Rate vs Model Score  ·  Critical Tier Trend",
             fontsize=11, fontweight="bold", color=C["white"])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
          facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ax.grid(axis="y", alpha=0.2)

# Quarterly
ax = axes[1]
q_labels  = quarterly["period"].tolist()
q_act     = quarterly["actual_rate"].values * 100
q_scr     = quarterly["mean_score"].values * 100
x_q       = np.arange(len(q_labels))
ax.plot(x_q, q_act, color=C["red"],  lw=1.8, marker=".", ms=5, label="Actual Rate")
ax.plot(x_q, q_scr, color=C["teal"], lw=1.8, marker=".", ms=5, label="Mean Model Score", alpha=0.8)
ax.fill_between(x_q, q_act, q_scr, alpha=0.12, color=C["amber"])
# COVID annotation
covid_start = next((i for i, p in enumerate(q_labels) if "2020-Q" in p), None)
if covid_start is not None:
    ax.axvspan(covid_start - 0.5, covid_start + 3.5, alpha=0.08,
               color=C["amber"], label="COVID-19 Period")
    ax.text(covid_start + 1.5, ax.get_ylim()[1] * 0.95, "COVID-19",
            ha="center", fontsize=9, color=C["amber"], fontfamily="monospace")

step = max(1, len(q_labels) // 16)
ax.set_xticks(x_q[::step])
ax.set_xticklabels(q_labels[::step], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Rate / Score (%)", fontsize=10, color=C["muted"])
ax.set_title("Quarterly Readmission Trend  ·  Actual vs Model Score",
             fontsize=11, fontweight="bold", color=C["white"])
ax.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ax.grid(alpha=0.2)

plt.tight_layout(pad=1.5, h_pad=2.5)
path = os.path.join(OUT_DIR, "C1_readmission_trend.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# C2. PROVIDER STATE ANALYSIS  — mean risk score by state
# ─────────────────────────────────────────────────────────────────────────────
print("[C2] Provider State Risk Analysis ...")

df_state = df_all.copy()
df_state["state"] = df["PRVDR_STATE_CD"].values

state_stats = df_state.groupby("state").agg(
    mean_score  = ("pred_score", "mean"),
    actual_rate = (TARGET, "mean"),
    n           = (TARGET, "count"),
    critical_n  = ("pred_score", lambda x: (x > 0.72).sum()),
).reset_index()
state_stats = state_stats[state_stats["n"] >= 30].sort_values("mean_score", ascending=False)
top_states  = state_stats.head(20)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.patch.set_facecolor(C["bg"])
fig.suptitle("Provider State Analysis  ·  Mean Risk Score vs Actual Rate",
             fontsize=13, fontweight="bold", color=C["white"], y=1.01)

ax = axes[0]
y_pos     = np.arange(len(top_states))
score_col = [C["red"] if s > 0.7 else C["amber"] if s > 0.55 else C["teal"]
             for s in top_states["mean_score"]]
bars = ax.barh(y_pos, top_states["mean_score"], color=score_col,
               edgecolor=C["bg"], alpha=0.88, height=0.65)
ax.plot(top_states["actual_rate"].values, y_pos, "o",
        color=C["white"], ms=6, zorder=5, label="Actual Rate", alpha=0.85)

for i, (_, row) in enumerate(top_states.iterrows()):
    ax.text(row["mean_score"] + 0.005, i,
            f"n={int(row['n'])}", va="center", fontsize=7.5,
            color=C["muted"], fontfamily="monospace")

ax.set_yticks(y_pos)
ax.set_yticklabels([f"State {int(s)}" for s in top_states["state"]], fontsize=9)
ax.set_xlabel("Mean Risk Score / Actual Readmit Rate", fontsize=10, color=C["muted"])
ax.set_title("Top 20 States by Mean Risk Score\n(● = Actual Readmit Rate)",
             fontsize=11, fontweight="bold", color=C["white"])
ax.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ax.grid(axis="x", alpha=0.2)

# Scatter: mean score vs actual rate — do states calibrate well?
ax = axes[1]
sc = ax.scatter(
    state_stats["mean_score"],
    state_stats["actual_rate"],
    s=state_stats["n"] / state_stats["n"].max() * 300 + 20,
    c=state_stats["mean_score"],
    cmap="RdYlGn_r",
    alpha=0.75, edgecolors=C["border"], lw=0.5, zorder=3,
)
# Perfect calibration line
diag = np.linspace(state_stats["mean_score"].min(), state_stats["mean_score"].max(), 50)
ax.plot(diag, diag, color=C["muted"], lw=1.5, ls="--", alpha=0.6, label="Perfect calibration")

ax.set_xlabel("Mean Predicted Score", fontsize=10, color=C["muted"])
ax.set_ylabel("Actual Readmit Rate", fontsize=10, color=C["muted"])
ax.set_title("State-Level Calibration\n(Bubble size = # patients in state)",
             fontsize=11, fontweight="bold", color=C["white"])
cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cb.ax.set_ylabel("Mean Risk Score", fontsize=8, color=C["muted"])
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"])
ax.grid(alpha=0.2)
ax.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])

plt.tight_layout(pad=1.5)
path = os.path.join(OUT_DIR, "C2_state_analysis.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# C3. AGE  ×  RISK  ×  COMORBIDITY BUBBLE CHART
# ─────────────────────────────────────────────────────────────────────────────
print("[C3] Age × Risk × Comorbidity Bubble ...")

age_idx = FEAT_COLS.index("AGE")
cc_idx  = FEAT_COLS.index("CC_COUNT")

df_bubble = pd.DataFrame({
    "age"     : X_test[:, age_idx],
    "score"   : y_prob,
    "cc_count": X_test[:, cc_idx],
    "actual"  : y_test,
    "tier"    : tiers_test,
})

# Bin ages into 10-year groups
df_bubble["age_bin"] = (df_bubble["age"] // 10 * 10).astype(int)
age_cc = df_bubble.groupby(["age_bin", "cc_count"]).agg(
    mean_score  = ("score", "mean"),
    actual_rate = ("actual", "mean"),
    n           = ("score", "count"),
).reset_index()
age_cc = age_cc[age_cc["n"] >= 5]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.patch.set_facecolor(C["bg"])
fig.suptitle("Age × Comorbidity × Risk Score  ·  Test Set 2023",
             fontsize=13, fontweight="bold", color=C["white"], y=1.01)

ax = axes[0]
norm = Normalize(vmin=0, vmax=1)
cmap_rb = LinearSegmentedColormap.from_list("risk", [C["green"], C["amber"], C["red"]], N=256)

sc = ax.scatter(
    age_cc["age_bin"],
    age_cc["cc_count"],
    s=age_cc["n"] / age_cc["n"].max() * 500 + 30,
    c=age_cc["mean_score"],
    cmap=cmap_rb, norm=norm,
    alpha=0.82, edgecolors=C["border"], lw=0.8, zorder=3,
)
cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cb.ax.set_ylabel("Mean Risk Score", fontsize=9, color=C["muted"])
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"])

ax.set_xlabel("Patient Age Group (decade)", fontsize=10, color=C["muted"])
ax.set_ylabel("Comorbidity Count (CC_COUNT)", fontsize=10, color=C["muted"])
ax.set_title("Mean Risk Score by Age Group × Comorbidity Burden\n(Bubble size = # patients)",
             fontsize=11, fontweight="bold", color=C["white"])
ax.grid(alpha=0.2)

# Right: age band risk bars
ax = axes[1]
age_band = df_bubble.groupby("age_bin").agg(
    mean_score  = ("score", "mean"),
    actual_rate = ("actual", "mean"),
    n           = ("score", "count"),
).reset_index().sort_values("age_bin")

x_ab  = np.arange(len(age_band))
w_ab  = 0.38
bar_colors_age = [cmap_rb(v) for v in age_band["mean_score"]]
ax.bar(x_ab - w_ab/2, age_band["mean_score"] * 100, w_ab,
       color=bar_colors_age, alpha=0.85, edgecolor=C["bg"], label="Mean Model Score")
ax.bar(x_ab + w_ab/2, age_band["actual_rate"] * 100, w_ab,
       color=C["white"], alpha=0.25, edgecolor=C["teal"], label="Actual Readmit Rate")

for i, (_, row) in enumerate(age_band.iterrows()):
    ax.text(i, max(row["mean_score"], row["actual_rate"]) * 100 + 0.8,
            f"n={int(row['n'])}", ha="center", fontsize=7.5,
            color=C["muted"], fontfamily="monospace")

ax.set_xticks(x_ab)
ax.set_xticklabels([f"{int(a)}s" for a in age_band["age_bin"]], fontsize=10)
ax.set_ylabel("Score / Rate (%)", fontsize=10, color=C["muted"])
ax.set_title("Mean Risk Score vs Actual Rate by Age Band",
             fontsize=11, fontweight="bold", color=C["white"])
ax.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ax.grid(axis="y", alpha=0.2)

plt.tight_layout(pad=1.5)
path = os.path.join(OUT_DIR, "C3_age_comorbidity_bubble.png")
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DASHBOARD  (10 panels in one figure — portfolio hero)
# ─────────────────────────────────────────────────────────────────────────────
print("[MASTER] Building master dashboard ...")

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor(C["bg"])

gs = gridspec.GridSpec(
    4, 4, figure=fig,
    hspace=0.50, wspace=0.38,
    left=0.05, right=0.97, top=0.90, bottom=0.04,
)

# Header
fig.text(0.5, 0.955,
         "30-Day Hospital Readmission Risk Engine  ·  Explainability Dashboard",
         ha="center", fontsize=18, fontweight="bold", color=C["white"],
         fontfamily="monospace")
fig.text(0.5, 0.928,
         f"XGBoost (HistGradient hist)  ·  {len(df):,} CMS Claims 2015–2023  ·  "
         f"Test AUC = {meta['test_metrics']['AUC_ROC']:.4f}  ·  "
         f"Test Set: 2023  ({len(X_test):,} patients)  ·  "
         f"CEP Local Explanations",
         ha="center", fontsize=10.5, color=C["muted"], fontfamily="monospace")

# ── ROW 0: KPI cards ─────────────────────────────────────────────────────────
kpis = [
    ("AUC-ROC",        f"{meta['test_metrics']['AUC_ROC']:.4f}",  C["teal"]),
    ("AUC-PR",         f"{meta['test_metrics']['AUC_PR']:.4f}",   "#0077B6"),
    ("Brier Score",    f"{meta['test_metrics']['Brier']:.4f}",    "#7B2FBE"),
    ("Youden Thresh",  f"{meta['youden_threshold']:.3f}",                    "#B5179E"),
]
for col, (title, val, color) in enumerate(kpis):
    ak = fig.add_subplot(gs[0, col])
    ak.set_facecolor(color)
    ak.text(0.5, 0.60, val,   ha="center", va="center",
            fontsize=26, fontweight="bold", color=C["white"],
            transform=ak.transAxes, fontfamily="monospace")
    ak.text(0.5, 0.20, title, ha="center", va="center",
            fontsize=10, color=C["white"], transform=ak.transAxes)
    ak.set_xticks([]); ak.set_yticks([])
    for sp in ak.spines.values():
        sp.set_edgecolor(C["bg"]); sp.set_linewidth(2)

# ── ROW 1 LEFT: Feature importance (mini) ────────────────────────────────────
af = fig.add_subplot(gs[1, 0:2])
af.set_facecolor(C["panel"])
top8 = fi_df.head(8).copy()
colors_fi = [C["red"] if v > fi_df["mean"].max() * 0.8
             else C["amber"] if v > fi_df["mean"].max() * 0.4 else C["teal"]
             for v in top8["mean"][::-1]]
af.barh([flabel(f) for f in top8["feature"][::-1]],
        top8["mean"][::-1], color=colors_fi, edgecolor=C["bg"], alpha=0.9, height=0.65)
af.set_title("Feature Importance  (Permutation Δ AUC)", fontsize=10,
             fontweight="bold", color=C["white"])
af.set_xlabel("Δ AUC-ROC", fontsize=8, color=C["muted"])
af.grid(axis="x", alpha=0.2)
for i, (_, row) in enumerate(top8.iterrows()):
    af.text(row["mean"] + 0.0008, 7 - i, f"Δ{row['mean']:.4f}",
            va="center", fontsize=7.5, color=C["muted"], fontfamily="monospace")

# ── ROW 1 RIGHT: Cumulative lift ──────────────────────────────────────────────
al = fig.add_subplot(gs[1, 2:4])
al.set_facecolor(C["panel"])
al.plot([0, 100], [0, 100], color=C["muted"], lw=1.2, ls="--", alpha=0.5, label="Random")
al.plot(pct_alerted, pct_captured, color=C["teal"], lw=2.5, label="XGBoost")
al.fill_between(pct_alerted, pct_captured, pct_alerted,
                where=pct_captured > pct_alerted, alpha=0.10, color=C["teal"])
for alert_pct, color in [(20, C["red"]), (40, C["amber"])]:
    idx_pt = int(alert_pct / 100 * len(pct_alerted))
    al.scatter(alert_pct, pct_captured[idx_pt], s=70, color=color, zorder=5)
    al.annotate(f"{alert_pct}% → {pct_captured[idx_pt]:.0f}%",
                (alert_pct, pct_captured[idx_pt]),
                xytext=(alert_pct + 4, pct_captured[idx_pt] - 7),
                fontsize=8, color=color, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=color, lw=1))
al.set_title("Cumulative Lift  (% Patients Alerted → % Readmits Captured)",
             fontsize=10, fontweight="bold", color=C["white"])
al.set_xlabel("% Patients Alerted", fontsize=8, color=C["muted"])
al.set_ylabel("% Readmissions Captured", fontsize=8, color=C["muted"])
al.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
al.grid(alpha=0.2)

# ── ROW 2: Patient waterfall ×2 (TP and TN only for space) ───────────────────
for col_i, (case_label, (pt_idx, case_color)) in enumerate(
    list(CASES.items())[:2]
):
    aw = fig.add_subplot(gs[2, col_i * 2: col_i * 2 + 2])
    aw.set_facecolor(C["panel"])

    pt_score, contribs = explain_patient(X_test[pt_idx], WATERFALL_IDX[:6])
    actual = y_test[pt_idx]
    tier   = assign_tier(pt_score)
    sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
    feat_labels = [flabel(f) for f, _ in sorted_c]
    vals = [v for _, v in sorted_c]
    bar_c = [C["red"] if v > 0 else C["green"] for v in vals]

    aw.barh(feat_labels[::-1], vals[::-1], color=bar_c[::-1],
            edgecolor=C["bg"], alpha=0.88, height=0.62)
    aw.axvline(0, color=C["border"], lw=1.2)
    aw.set_title(f"{case_label.replace(chr(10), ' ')}  —  Score: {pt_score:.3f}  Tier: {tier}",
                 fontsize=10, fontweight="bold", color=case_color)
    aw.set_xlabel("Feature Contribution (Δ Risk Score)", fontsize=8, color=C["muted"])
    aw.grid(axis="x", alpha=0.2)

# ── ROW 3: Risk tier bars + calibration + trend ───────────────────────────────
# Tier bars
at = fig.add_subplot(gs[3, 0])
at.set_facecolor(C["panel"])
tier_data = [(t, int(np.sum(tiers_test == t)),
              np.mean(y_test[tiers_test == t]) * 100) for t in TIER_ORDER]
t_names, t_ns, t_rs = zip(*tier_data)
t_cols = [TIER_COLORS[t] for t in t_names]
at.bar(t_names, t_rs, color=t_cols, edgecolor=C["bg"], alpha=0.88, width=0.6)
at.axhline(y_test.mean() * 100, color=C["muted"], lw=1.5, ls="--",
           label=f"Avg {y_test.mean()*100:.1f}%")
for i, (n, r) in enumerate(zip(t_ns, t_rs)):
    at.text(i, r + 0.8, f"{r:.1f}%", ha="center", fontsize=9,
            fontweight="bold", color=C["text"])
    at.text(i, -5, f"n={n}", ha="center", fontsize=7.5, color=C["muted"])
at.set_ylabel("Observed Readmit %", fontsize=8, color=C["muted"])
at.set_title("Readmit Rate by Risk Tier", fontsize=10, fontweight="bold", color=C["white"])
at.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
at.grid(axis="y", alpha=0.2)

# Calibration
prob_t_dash, prob_p_dash = calibration_curve(y_test, y_prob, n_bins=10)
ac = fig.add_subplot(gs[3, 1])
ac.set_facecolor(C["panel"])
ac.plot([0, 1], [0, 1], color=C["muted"], lw=1.2, ls="--", alpha=0.5, label="Perfect")
ac.plot(prob_p_dash, prob_t_dash, "o-", color=C["teal"], lw=2, ms=6, label="XGBoost")
ac.fill_between(prob_p_dash, prob_t_dash, prob_p_dash,
                alpha=0.15, color=C["red"], label="Calib gap")
ac.set_title(f"Calibration  (Brier={meta['test_metrics']['Brier']:.4f})",
             fontsize=10, fontweight="bold", color=C["white"])
ac.set_xlabel("Predicted", fontsize=8, color=C["muted"])
ac.set_ylabel("Observed", fontsize=8, color=C["muted"])
ac.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
ac.grid(alpha=0.2)

# Annual trend (last 2 cols)
at2 = fig.add_subplot(gs[3, 2:4])
at2.set_facecolor(C["panel"])
x_yr = np.arange(len(yearly))
at2.plot(x_yr, yearly["actual_rate"] * 100, "o-", color=C["red"],
         lw=2, ms=5, label="Actual Rate")
at2.plot(x_yr, yearly["mean_score"] * 100, "s--", color=C["teal"],
         lw=2, ms=5, label="Mean Score", alpha=0.8)
at2.fill_between(x_yr, yearly["actual_rate"] * 100, yearly["mean_score"] * 100,
                 alpha=0.10, color=C["amber"])
at2.set_xticks(x_yr)
at2.set_xticklabels([str(int(y)) for y in yearly[YEAR_COL]], fontsize=9)
at2.set_ylabel("Rate / Score (%)", fontsize=8, color=C["muted"])
at2.set_title("Annual Trend  ·  Actual vs Predicted", fontsize=10,
              fontweight="bold", color=C["white"])
at2.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["border"], labelcolor=C["text"])
at2.grid(alpha=0.2)

path = os.path.join(OUT_DIR, "MASTER_explain_dashboard.png")
plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✓ {path}  ← PORTFOLIO HERO")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  04_explain_dashboard.py  COMPLETE")
print("=" * 72)
print(f"""
  OUTPUT FIGURES ({OUT_DIR}):
    A1_feature_importance.png      — Global permutation importance, top 20
    A2_partial_dependence.png      — PDP: how each top feature shifts risk
    A3_contribution_heatmap.png    — Feature contributions by risk tier
    B1_patient_waterfall.png       — TP / TN / FP / FN case explanations
    B2_score_distribution_lift.png — Violin plots + cumulative lift curve
    C1_readmission_trend.png       — Annual & quarterly trend 2015–2023
    C2_state_analysis.png          — Risk by provider state
    C3_age_comorbidity_bubble.png  — Age × comorbidity × risk bubble chart
    MASTER_explain_dashboard.png   — All panels combined  ← HERO IMAGE

  EXPLAINABILITY METHOD:
    Conditional Expectation Perturbation (CEP)
      contrib(feature_i) = score(patient) − score(patient | feature_i = median)
    • Model-agnostic — works on any sklearn estimator
    • Positive contribution → feature raises risk above baseline
    • Negative contribution → feature lowers risk below baseline

  SHAP SWAP (when pip install shap):
    import shap
    explainer = shap.TreeExplainer(model)          # fast C++ TreeExplainer
    shap_vals = explainer.shap_values(X_test)      # shape (n, n_features)
    shap.waterfall_plot(explainer.expected_value,
                        shap_vals[patient_idx])    # replaces CEP waterfall

  CASE STUDY PATIENTS (test set 2023):""")
for case_label, (pt_idx, _) in CASES.items():
    label_clean = case_label.replace("\n", " ")
    print(f"    {label_clean:<35}  idx={pt_idx}  score={y_prob[pt_idx]:.4f}  actual={y_test[pt_idx]}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# APPENDIX: SHAP SWAP FUNCTION (documented, not executed — shap not installed)
# ─────────────────────────────────────────────────────────────────────────────
def shap_swap_example():
    """
    Drop-in SHAP replacement for the CEP explain_patient() function above.
    Run this when: pip install shap  succeeds.

    from xgboost import XGBClassifier
    import shap

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)   # (1863, 37) array
    base_value = explainer.expected_value         # scalar

    # Waterfall for one patient
    shap.plots.waterfall(shap.Explanation(
        values        = shap_vals[tp_idx],
        base_values   = base_value,
        data          = X_test[tp_idx],
        feature_names = FEAT_COLS,
    ))

    # Summary plot (replaces A1_feature_importance.png)
    shap.summary_plot(shap_vals, X_test, feature_names=FEAT_COLS)

    # Dependence plot for LOS_DAYS (replaces A2_partial_dependence.png)
    shap.dependence_plot("LOS_DAYS", shap_vals, X_test,
                         feature_names=FEAT_COLS)
    """
    pass  # placeholder — not executed here
