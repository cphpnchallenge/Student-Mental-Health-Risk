#!/usr/bin/env python3
"""
Recreate the 8 core EDA figures directly from ssaqs_analysis_table.csv --
no dependency on eda_results.json, eda_within_subject_correlations.csv, or
eda_between_subject_correlations.csv (those are run_eda.py's cached outputs).
Every statistic (ICC, within-person correlations + FDR, between-person
correlations + FDR, weekly/weekday aggregates, the three predictive models)
is computed inline from the raw table, using the same methods as run_eda.py.

Saves each figure as its own PNG under analysis/figures/eda_from_table/,
rather than the base64 figs.json the HTML report uses.
"""
import math
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statlib as S

warnings.filterwarnings("ignore")

OUT = "/Users/ruizhang/work/student-mental-health/clean_data"
FIG = "/Users/ruizhang/work/student-mental-health/figures"
os.makedirs(FIG, exist_ok=True)

INK = "#7c8798"; BLUE = "#3b82f6"; AMBER = "#f59e0b"
TEAL = "#14b8a6"; ROSE = "#f43f5e"; GRID = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.titlecolor": INK, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

df = pd.read_csv(f"{OUT}/ssaqs_analysis_table.csv", parse_dates=["t_local"])
print(f"loaded {len(df)} rows, {df.subject.nunique()} participants "
      f"directly from ssaqs_analysis_table.csv")


def save(name, fig):
    path = f"{FIG}/{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, alpha=.25, lw=.7)
    ax.set_axisbelow(True)


# =====================================================================
# A. ICC -- one-way, share of variance that is between-person
# =====================================================================
def icc1(frame, col):
    g = frame.groupby("subject")[col]
    k = g.size()
    gm = frame[col].mean()
    msb = (k * (g.mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((frame[col] - frame.subject.map(g.mean())) ** 2).sum() / (len(frame) - len(k))
    k0 = k.mean()
    return max(0.0, (msb - msw) / (msb + (k0 - 1) * msw))


ICC = {c: round(icc1(df, c), 3) for c in ["stress", "anxiety"]}
print("ICC:", ICC)

# =====================================================================
# B. within-person correlations (Fisher-z pooled across participants) + FDR
# =====================================================================
FEATURES = {
    "steps_sum_1h": "Steps, last 1 h",
    "steps_sum_4h": "Steps, last 4 h",
    "steps_sum_24h": "Steps, last 24 h",
    "act_frac_sed_1h": "% sedentary min, last 1 h",
    "act_frac_sed_4h": "% sedentary min, last 4 h",
    "act_frac_sed_24h": "% sedentary min, last 24 h",
    "act_mvpa_min_1h": "Mod+vig active min, last 1 h",
    "act_mvpa_min_4h": "Mod+vig active min, last 4 h",
    "act_mvpa_min_24h": "Mod+vig active min, last 24 h",
    "act_frac_very_24h": "% very-active min, last 24 h",
    "sleep_score": "Sleep score (prior night)",
    "deep_sleep_min": "Deep sleep min (prior night)",
    "hours_since_sleep_onset": "Hours since sleep onset",
    "hrv_rmssd_night": "HRV RMSSD (prior night)",
    "hrv_rmssd_night_sd": "HRV RMSSD variability (night)",
    "hrv_lfhf_night": "HRV LF/HF ratio (night)",
    "hrv_hf_night": "HRV HF power (night)",
    "spo2_night_mean": "SpO₂ mean (prior night)",
    "spo2_night_min": "SpO₂ min (prior night)",
    "spo2_night_sd": "SpO₂ variability (night)",
    "spo2_night_pct_lt90": "% night with SpO₂ < 90",
    "dev_stress_same_day": "Device daily stress score (same day)",
    "dev_stress_prev_day": "Device daily stress score (prev day)",
    "wear_frac_24h": "Watch wear fraction, last 24 h",
    "local_hour": "Local time of day",
    "week_in_study": "Week of semester",
    "resp_latency_s": "Time taken to answer (s)",
    "resp_delay_s": "Delay before answering (s)",
}
MIN_OBS, MIN_SUBJ = 20, 10

within = []
for feat, label in FEATURES.items():
    if feat not in df.columns:
        continue
    for lab in ["stress", "anxiety"]:
        rs, ns = [], []
        for s, g in df.groupby("subject"):
            rr, _, nn = S.spearman(g[feat], g[lab])
            if nn >= MIN_OBS and np.isfinite(rr):
                rs.append(rr); ns.append(nn)
        if len(rs) < MIN_SUBJ:
            continue
        z = S.fisher_z(rs)
        t, p, k = S.ttest_1samp(z)
        zm = float(np.mean(z)); zse = float(np.std(z, ddof=1) / math.sqrt(k))
        within.append(dict(feature=feat, label_name=label, target=lab,
                           mean_r=float(np.tanh(zm)),
                           ci_lo=float(np.tanh(zm - 1.96 * zse)),
                           ci_hi=float(np.tanh(zm + 1.96 * zse)),
                           p=p, n_subjects=k))
W = pd.DataFrame(within)
for lab in ["stress", "anxiety"]:
    m = W.target == lab
    W.loc[m, "q"] = S.bh_fdr(W.loc[m, "p"].values)

# =====================================================================
# C. between-person correlations (participant means) + FDR
# =====================================================================
agg_feats = [f for f in FEATURES if f in df.columns and f not in
             ("local_hour", "week_in_study")]
sm = df.groupby("subject").agg(
    stress=("stress", "mean"), anxiety=("anxiety", "mean"),
    **{f: (f, "mean") for f in agg_feats}
).reset_index()

between = []
for feat, label in FEATURES.items():
    if feat not in sm.columns:
        continue
    for lab in ["stress", "anxiety"]:
        rr, pp, nn = S.spearman(sm[feat], sm[lab])
        if np.isfinite(rr):
            between.append(dict(feature=feat, label_name=label, target=lab,
                                r=rr, p=pp, n=nn))
B = pd.DataFrame(between)
for lab in ["stress", "anxiety"]:
    m = B.target == lab
    B.loc[m, "q"] = S.bh_fdr(B.loc[m, "p"].values)

# =====================================================================
# D. weekly / weekday aggregates, person-centered
# =====================================================================
dfc = df.copy()
for lab in ["stress", "anxiety"]:
    dfc[lab + "_c"] = dfc[lab] - dfc.groupby("subject")[lab].transform("mean")

by_week = dfc[dfc.week_in_study.between(0, 21)].groupby("week_in_study").agg(
    n=("stress", "size"), stress=("stress", "mean"), anxiety=("anxiety", "mean")
).reset_index()

by_weekday = dfc.groupby("weekday").agg(
    n=("stress", "size"), stress_c=("stress_c", "mean"),
    anxiety_c=("anxiety_c", "mean")
).reset_index()

# =====================================================================
# E. predictive models: person only / + time context / + sensors,
#    random 5-fold (40 seeds) and leave-one-participant-out
# =====================================================================
MODEL_FEATS = ["steps_sum_24h", "steps_sum_4h", "act_frac_sed_24h", "act_mvpa_min_24h",
               "sleep_score", "deep_sleep_min", "hrv_rmssd_night", "hrv_lfhf_night",
               "spo2_night_mean", "spo2_night_sd", "spo2_night_pct_lt90",
               "wear_frac_24h"]

md = df.dropna(subset=MODEL_FEATS + ["stress", "anxiety"]).copy()
dummies = pd.get_dummies(md.subject.astype(str), prefix="s", drop_first=True).astype(float)
md["hour_sin"] = np.sin(2 * np.pi * md.local_hour / 24)
md["hour_cos"] = np.cos(2 * np.pi * md.local_hour / 24)
time_cols = ["hour_sin", "hour_cos", "is_weekend", "is_class_day", "week_in_study"]
blocks = {
    "person only": dummies.values,
    "+ time context": np.column_stack([dummies.values, md[time_cols].values]),
    "+ sensors": np.column_stack([dummies.values, md[time_cols].values,
                                  md[MODEL_FEATS].values]),
}


def cv_r2(X, y, folds):
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if te.sum() == 0 or tr.sum() < 50:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        b = S.ridge_fit((X[tr] - mu) / sd, y[tr], lam=10.0)
        pred[te] = S.ridge_pred(b, (X[te] - mu) / sd)
    m = np.isfinite(pred)
    ss_res = ((y[m] - pred[m]) ** 2).sum()
    ss_tot = ((y[m] - y[m].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


N_SEEDS = 40
model_res = []
for lab in ["stress", "anxiety"]:
    y = md[lab].values.astype(float)
    for name, X in blocks.items():
        vals = []
        for seed in range(N_SEEDS):
            folds = np.random.default_rng(seed).integers(0, 5, size=len(y))
            vals.append(cv_r2(X, y, folds))
        model_res.append(dict(target=lab, model=name,
                              r2=float(np.mean(vals)), r2_sd=float(np.std(vals, ddof=1))))
MD = pd.DataFrame(model_res)

lopo_folds = pd.factorize(md.subject)[0]
X_nop = {
    "time context only": md[time_cols].values,
    "+ sensors": np.column_stack([md[time_cols].values, md[MODEL_FEATS].values]),
}
lopo = []
for lab in ["stress", "anxiety"]:
    y = md[lab].values.astype(float)
    for name, X in X_nop.items():
        lopo.append(dict(target=lab, model=name, r2=float(cv_r2(X, y, lopo_folds))))
LO = pd.DataFrame(lopo)

print(f"model_n={len(md)}, subjects={md.subject.nunique()}")

# =====================================================================
# FIGURES -- identical layouts to make_figs.py, all inputs computed above
# =====================================================================

# 1 --------------------------------------------------------- label distributions
fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
for ax, lab, col in zip(axes, ["stress", "anxiety"], [BLUE, AMBER]):
    ax.hist(df[lab], bins=np.arange(0, 102, 4), color=col, alpha=.85, edgecolor="none")
    ax.axvline(df[lab].mean(), color=ROSE, lw=1.4, ls="--")
    ax.set_title(f"Self-reported {lab}   (mean {df[lab].mean():.1f})", fontsize=10)
    ax.set_xlabel("0–100 slider"); ax.set_ylabel("responses")
    grid(ax)
save("labels", fig)

# 2 ---------------------------------------------- person differences (caterpillar)
g = df.groupby("subject")["stress"]
order = g.mean().sort_values().index
fig, ax = plt.subplots(figsize=(8.6, 2.9))
x = np.arange(len(order))
lo = df.groupby("subject")["stress"].quantile(.25).loc[order].values
hi = df.groupby("subject")["stress"].quantile(.75).loc[order].values
mu = g.mean().loc[order].values
ax.vlines(x, lo, hi, color=BLUE, alpha=.45, lw=5)
ax.plot(x, mu, "o", color=BLUE, ms=4.5, label="person mean")
mua = df.groupby("subject")["anxiety"].mean().loc[order].values
ax.plot(x, mua, "^", color=AMBER, ms=4.5, label="anxiety mean")
ax.axhline(df.stress.mean(), color=ROSE, lw=1.2, ls="--", label="cohort mean stress")
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7)
ax.set_xlabel("participant (sorted by mean stress)"); ax.set_ylabel("0–100")
ax.set_title(f"Person differences dominate: mean stress ranges {mu.min():.0f}→"
             f"{mu.max():.0f} (ICC = {ICC['stress']})", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
grid(ax)
save("people", fig)

# 3 ------------------------------------------------ within-person forest plot
keep = ["wear_frac_24h", "hours_since_sleep_onset", "week_in_study", "hrv_lfhf_night",
        "act_frac_sed_4h", "act_frac_sed_24h", "steps_sum_24h", "steps_sum_4h",
        "act_mvpa_min_24h", "sleep_score", "deep_sleep_min", "hrv_rmssd_night",
        "spo2_night_mean", "spo2_night_pct_lt90", "dev_stress_same_day",
        "dev_stress_prev_day"]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
rows = []
for f in keep:
    a = W[(W.target == "stress") & (W.feature == f)]
    b = W[(W.target == "anxiety") & (W.feature == f)]
    if len(a) and len(b):
        rows.append((a.iloc[0], b.iloc[0]))
rows = sorted(rows, key=lambda t: abs(t[0].mean_r))
y = np.arange(len(rows))
for i, (a, b) in enumerate(rows):
    ax.plot([a.ci_lo, a.ci_hi], [i + .16] * 2, color=BLUE, lw=2, alpha=.65)
    ax.plot(a.mean_r, i + .16, "o", color=BLUE, ms=5.5,
            markeredgecolor="none" if a.q < .05 else BLUE,
            markerfacecolor=BLUE if a.q < .05 else "none")
    ax.plot([b.ci_lo, b.ci_hi], [i - .16] * 2, color=AMBER, lw=2, alpha=.65)
    ax.plot(b.mean_r, i - .16, "^", color=AMBER, ms=5.5,
            markerfacecolor=AMBER if b.q < .05 else "none")
ax.axvline(0, color=INK, lw=.9)
for xv in (-.3, -.2, -.1, .1, .2, .3):
    ax.axvline(xv, color=GRID, lw=.6, alpha=.22)
ax.set_yticks(y); ax.set_yticklabels([r[0].label_name for r in rows], fontsize=8.5)
ax.set_xlabel("within-person Spearman ρ  (mean across participants, 95% CI)")
ax.set_xlim(-.35, .35)
ax.set_title("No physiological feature tracks day-to-day stress or anxiety\n"
             "filled marker = survives FDR q<0.05  •  ● stress  ▲ anxiety",
             fontsize=10)
save("forest", fig)

# 4 ----------------------------------------------------- between-person scatters
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
pairs = [("dev_stress_same_day", "stress", "Device daily stress score", BLUE),
         ("act_frac_sed_24h", "anxiety", "% sedentary minutes (24 h)", AMBER),
         ("hrv_rmssd_night_sd", "stress", "Night HRV RMSSD variability", TEAL)]
for ax, (f, lab, xl, col) in zip(axes, pairs):
    d = sm[[f, lab]].dropna()
    ax.scatter(d[f], d[lab], s=26, color=col, alpha=.8, edgecolor="none")
    if len(d) > 3:
        k, c = np.polyfit(d[f], d[lab], 1)
        xs = np.linspace(d[f].min(), d[f].max(), 20)
        ax.plot(xs, k * xs + c, color=ROSE, lw=1.3, ls="--")
    rr = B[(B.feature == f) & (B.target == lab)]
    tag = f"ρ = {rr.iloc[0].r:+.2f}, p = {rr.iloc[0].p:.3f}" if len(rr) else ""
    ax.set_title(tag, fontsize=9)
    ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(f"mean {lab}", fontsize=9)
    grid(ax, "both")
fig.suptitle("Between-person: participants who differ on sensors also differ on "
             "mean self-report (n = 32, none survive FDR)", fontsize=10, y=1.06)
save("between", fig)

# 5 ------------------------------------------------------------- semester arc
wk = by_week[by_week.n >= 30]
fig, ax = plt.subplots(figsize=(8.6, 2.9))
ax.plot(wk.week_in_study, wk.stress, "-o", color=BLUE, ms=4, lw=1.6, label="stress")
ax.plot(wk.week_in_study, wk.anxiety, "-^", color=AMBER, ms=4, lw=1.6, label="anxiety")
ax.set_xlabel("week of semester"); ax.set_ylabel("mean self-report (0–100)")
ax.set_title("Semester arc: cohort mean by week (n≥30)", fontsize=10)
ax.legend(frameon=False, fontsize=9)
grid(ax)
save("weeks", fig)

# 6 -------------------------------------------------------------- weekday effect
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
wd = by_weekday.set_index("weekday").reindex(range(7)).reset_index()
fig, ax = plt.subplots(figsize=(8.6, 2.6))
xw = np.arange(7); w = .38
ax.bar(xw - w / 2, wd.stress_c, w, color=BLUE, label="stress", alpha=.9)
ax.bar(xw + w / 2, wd.anxiety_c, w, color=AMBER, label="anxiety", alpha=.9)
ax.axhline(0, color=INK, lw=.9)
ax.set_xticks(xw); ax.set_xticklabels(names)
ax.set_ylabel("deviation from\nperson's own mean")
ax.set_title("Deviation from each participant's own mean, by weekday", fontsize=10)
ax.legend(frameon=False, fontsize=9)
grid(ax)
save("weekday", fig)

# 7 ----------------------------------------------------------------- models
fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.9),
                         gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
mods = ["person only", "+ time context", "+ sensors"]
xm = np.arange(3); w = .38
for i, (lab, col) in enumerate([("stress", BLUE), ("anxiety", AMBER)]):
    sub = [MD[(MD.target == lab) & (MD.model == m)].iloc[0] for m in mods]
    v = [s.r2 for s in sub]; e = [s.r2_sd for s in sub]
    ax.bar(xm + (i - .5) * w, v, w, color=col, label=lab, alpha=.9,
           yerr=e, capsize=3, error_kw=dict(ecolor=INK, lw=.9))
    for xi, vi, ei in zip(xm + (i - .5) * w, v, e):
        ax.text(xi, vi + ei + .008, f"{vi:.3f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(xm); ax.set_xticklabels(mods, fontsize=9)
ax.set_ylabel("out-of-sample R²"); ax.set_ylim(0, .40)
ax.set_title("Same students in training: person identity carries it all",
             fontsize=9.5)
ax.legend(frameon=False, fontsize=9, loc="upper left")
grid(ax)

ax = axes[1]
mods2 = ["time context only", "+ sensors"]
xl2 = np.arange(2)
for i, (lab, col) in enumerate([("stress", BLUE), ("anxiety", AMBER)]):
    v = [LO[(LO.target == lab) & (LO.model == m)].r2.iloc[0] for m in mods2]
    ax.bar(xl2 + (i - .5) * w, v, w, color=col, alpha=.9)
    for xi, vi in zip(xl2 + (i - .5) * w, v):
        ax.text(xi, vi - .012, f"{vi:+.3f}", ha="center", va="top",
                fontsize=8, color=INK)
ax.axhline(0, color=ROSE, lw=1.1)
ax.set_xticks(xl2); ax.set_xticklabels(["time\ncontext", "+ sensors"], fontsize=9)
ax.set_ylim(-.16, .06); ax.set_ylabel("out-of-sample R²")
ax.set_title("Unseen student: worse than\npredicting the cohort mean", fontsize=9.5)
grid(ax)
save("models", fig)

# 8 ------------------------------------------------------------- data coverage
cov = (100 * df.notna().mean())
groups = {
    "Labels + context": ["stress", "anxiety", "local_hour", "weekday", "is_class_day"],
    "Activity, 24 h": ["steps_sum_24h", "act_frac_sed_24h", "act_mvpa_min_24h"],
    "Activity, 4 h": ["steps_sum_4h", "act_frac_sed_4h"],
    "Activity, 1 h": ["steps_sum_1h", "act_frac_sed_1h"],
    "Prior-night sleep": ["sleep_score", "deep_sleep_min"],
    "Prior-night HRV": ["hrv_rmssd_night", "hrv_lfhf_night"],
    "Prior-night SpO₂": ["spo2_night_mean", "spo2_night_min"],
    "Device daily stress": ["dev_stress_same_day"],
    "HRV within 4 h": ["hrv_rmssd_mean_4h"],
    "SpO₂ within 4 h": ["spo2_mean_4h"],
}
vals = [np.mean([cov[c] for c in cs if c in cov]) for cs in groups.values()]
fig, ax = plt.subplots(figsize=(8.6, 3.2))
yy = np.arange(len(groups))[::-1]
cols = [TEAL if v >= 90 else (AMBER if v >= 50 else ROSE) for v in vals]
ax.barh(yy, vals, color=cols, alpha=.9)
for y_, v in zip(yy, vals):
    ax.text(v + 1.5, y_, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(list(groups), fontsize=9)
ax.set_xlim(0, 112); ax.set_xlabel("% of surveys with the feature available")
ax.set_title("Feature availability across the joined table", fontsize=10)
grid(ax, "x")
save("coverage", fig)

print(f"\n8 figures written to {FIG}")
