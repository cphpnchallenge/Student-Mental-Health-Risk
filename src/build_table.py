#!/usr/bin/env python3
"""
SSAQS: build a cleaned, joined analysis table.

One row per self-reported stress/anxiety survey, with sensor features
aggregated over 1h / 4h / 24h lookback windows ending at the moment the
survey was sent, plus prior-night sleep/HRV/SpO2 and device daily stress.

Excluded: subjects 3, 12, 14 (questionnaire data only, no wearable).
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = "/Users/ruizhang/work/student-mental-health/data"
OUT = "/Users/ruizhang/work/student-mental-health/clean_data"
os.makedirs(OUT, exist_ok=True)

EXCLUDE = {3, 12, 14}
WINDOWS = [1, 4, 24]          # hours
SPO2_FLOOR = 80.0             # drop implausible readings (device floors at 50)
HRV_MIN_COVERAGE = 0.80       # drop low-coverage HRV epochs
SLEEP_LOOKBACK_H = 18         # how far back to search for the prior sleep record
NIGHT_LEN_H = 10              # sleep session assumed to span <= 10h from onset

ACT_LEVELS = ["SEDENTARY", "LIGHTLY_ACTIVE", "MODERATELY_ACTIVE", "VERY_ACTIVE"]
ACT_SHORT = {"SEDENTARY": "sed", "LIGHTLY_ACTIVE": "light",
             "MODERATELY_ACTIVE": "mod", "VERY_ACTIVE": "very"}

# ---------------------------------------------------------------- metadata
uc = pd.read_csv(f"{BASE}/users-courses.csv", encoding="utf-8-sig")
cd = pd.read_csv(f"{BASE}/course-details.csv", encoding="utf-8-sig")
meta = uc.merge(cd, left_on="course", right_on="idcourse", how="left")
meta["startDate"] = pd.to_datetime(meta["startDate"], format="%m/%d/%Y")
meta["endDate"] = pd.to_datetime(meta["endDate"], format="%m/%d/%Y")
meta = meta.set_index("userid")

DAYCOLS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

subjects = sorted(int(d) for d in os.listdir(BASE)
                  if d.isdigit() and int(d) not in EXCLUDE)

quality = []
rows = []


def iso(series):
    return pd.to_datetime(series, format="ISO8601", utc=True)


def win_idx(ts, t_end, hours):
    """Index range for records in [t_end - hours, t_end)."""
    lo = t_end - np.timedelta64(hours, "h")
    return np.searchsorted(ts, lo, "left"), np.searchsorted(ts, t_end, "left")


for sid in subjects:
    d = f"{BASE}/{sid}"

    # ---- surveys (labels) -------------------------------------------------
    dq = pd.read_csv(f"{d}/daily_questions.csv")
    dq["t_utc"] = pd.to_datetime(dq["timeStampSent"], unit="s", utc=True)
    dq["t_local"] = dq["t_utc"] + pd.to_timedelta(dq["timeZoneOffset"], unit="s")
    dq = dq.sort_values("t_utc").reset_index(drop=True)
    # the naive-local sensor files (hrv, stress) need a UTC offset. Use the
    # offset the participant was actually in on that date -- 9 of 32 report
    # more than one, and one travelled from UTC-6 to UTC+2 mid-study.
    off_by_date = (dq.set_index(dq["t_local"].dt.tz_localize(None).dt.normalize())
                     ["timeZoneOffset"].groupby(level=0).last().sort_index())
    off_default = int(dq["timeZoneOffset"].mode()[0])

    def offsets_for(naive_ts):
        """Offset in seconds in force on each naive-local timestamp's date."""
        if off_by_date.nunique() == 1:
            return np.full(len(naive_ts), float(off_by_date.iloc[0]))
        idx = off_by_date.index.values
        pos = np.clip(np.searchsorted(idx, naive_ts.dt.normalize().values,
                                      "right") - 1, 0, len(idx) - 1)
        return off_by_date.values[pos].astype(float)

    # ---- continuous sensors ----------------------------------------------
    steps = pd.read_csv(f"{d}/steps.csv")
    steps["t"] = iso(steps["timestamp"])
    steps = steps.sort_values("t").reset_index(drop=True)
    st_t, st_v = steps["t"].values, steps["steps"].values.astype(float)

    act = pd.read_csv(f"{d}/activity_level.csv")
    act["t"] = iso(act["timestamp"])
    act = act.sort_values("t").reset_index(drop=True)
    ac_t = act["t"].values
    ac_code = act["level"].map({l: i for i, l in enumerate(ACT_LEVELS)}).values

    # ---- sleep-time sensors ----------------------------------------------
    ox = pd.read_csv(f"{d}/oxygen.csv")
    ox["t"] = iso(ox["timestamp"])
    n_ox_raw = len(ox)
    ox = ox[ox["value"] >= SPO2_FLOOR].sort_values("t").reset_index(drop=True)
    ox_dropped = 1 - len(ox) / max(n_ox_raw, 1)
    ox_t, ox_v = ox["t"].values, ox["value"].values.astype(float)

    hrv = pd.read_csv(f"{d}/hrv.csv")
    _hv_naive = pd.to_datetime(hrv["timestamp"], format="ISO8601")
    hrv["t"] = (_hv_naive - pd.to_timedelta(offsets_for(_hv_naive), unit="s")
                ).dt.tz_localize("UTC")
    n_hrv_raw = len(hrv)
    hrv = hrv[hrv["coverage"] >= HRV_MIN_COVERAGE].sort_values("t").reset_index(drop=True)
    hrv_dropped = 1 - len(hrv) / max(n_hrv_raw, 1)
    hv_t = hrv["t"].values
    hv_rmssd = hrv["rmssd"].values.astype(float)
    hv_lf = hrv["low_frequency"].values.astype(float)
    hv_hf = hrv["high_frequency"].values.astype(float)

    sleep = pd.read_csv(f"{d}/sleep.csv")
    sleep["t"] = iso(sleep["timestamp"])   # onset / bedtime
    sleep = sleep.sort_values("t").reset_index(drop=True)
    sl_t = sleep["t"].values

    # ---- device daily stress ---------------------------------------------
    ds = pd.read_csv(f"{d}/stress.csv")
    ds["date"] = pd.to_datetime(ds["DATE"], format="ISO8601").dt.date
    ds["updated"] = (pd.to_datetime(ds["UPDATED_AT"], format="ISO8601")
                     - pd.to_timedelta(offsets_for(
                         pd.to_datetime(ds["UPDATED_AT"], format="ISO8601")), unit="s")
                     ).dt.tz_localize("UTC")
    ds_ok = ds[~ds["CALCULATION_FAILED"].astype(bool)]
    ds_map = dict(zip(ds_ok["date"], ds_ok["STRESS_SCORE"].astype(float)))
    ds_upd = dict(zip(ds_ok["date"], ds_ok["updated"]))
    ds_dropped = 1 - len(ds_ok) / max(len(ds), 1)

    m = meta.loc[sid]
    class_days = {i for i, c in enumerate(DAYCOLS) if int(m[c]) == 1}
    cls_start = pd.to_timedelta(m["startTime"] + ":00")
    cls_end = pd.to_timedelta(m["endTime"] + ":00")

    quality.append(dict(subject=sid, university=int(m["university"]),
                        course=m["course"], semester=int(m["semester"]),
                        n_surveys=len(dq),
                        spo2_pct_dropped=round(100 * ox_dropped, 1),
                        hrv_pct_dropped=round(100 * hrv_dropped, 1),
                        devstress_pct_failed=round(100 * ds_dropped, 1),
                        n_sleep_nights=len(sleep)))

    for _, q in dq.iterrows():
        te = np.datetime64(q["t_utc"].tz_convert("UTC").tz_localize(None), "ns")
        tl = q["t_local"]
        tl_naive = tl.tz_localize(None) if tl.tzinfo else tl
        r = dict(
            subject=sid,
            university=int(m["university"]),
            course=m["course"],
            semester=int(m["semester"]),
            undergraduate=int(m["undergraduate"]),
            t_utc=q["t_utc"],
            t_local=tl_naive,
            date_local=tl_naive.date(),
            local_hour=tl_naive.hour + tl_naive.minute / 60.0,
            weekday=tl_naive.weekday(),
            is_weekend=int(tl_naive.weekday() >= 5),
            day_in_study=(tl_naive.normalize() - m["startDate"]).days,
            stress=int(q["stress"]),
            anxiety=int(q["anxiety"]),
            resp_latency_s=int(q["timeStampStop"] - q["timeStampStart"]),
            # 23 rows carry a start earlier than the send timestamp; treat
            # those as bad records rather than negative delays
            resp_delay_s=(int(q["timeStampStart"] - q["timeStampSent"])
                          if q["timeStampStart"] >= q["timeStampSent"] else np.nan),
        )
        r["week_in_study"] = r["day_in_study"] // 7

        wd = tl_naive.weekday()
        r["is_class_day"] = int(wd in class_days)
        tod = pd.to_timedelta(f"{tl_naive.hour}:{tl_naive.minute}:00")
        r["in_class_window"] = int(r["is_class_day"] and cls_start <= tod <= cls_end)
        r["hours_to_class_start"] = ((tod - cls_start).total_seconds() / 3600
                                     if r["is_class_day"] else np.nan)

        # -------- windowed features ---------------------------------------
        for W in WINDOWS:
            i0, i1 = win_idx(st_t, te, W)
            seg = st_v[i0:i1]
            steps_sum, steps_nrec = float(seg.sum()), len(seg)

            i0, i1 = win_idx(ac_t, te, W)
            seg = ac_code[i0:i1]
            n = len(seg)
            r[f"act_nmin_{W}h"] = n
            # the watch logs a steps row only when steps > 0, so absence of
            # steps rows is a true zero whenever the watch was worn (activity
            # minutes present) and missing otherwise
            r[f"steps_sum_{W}h"] = steps_sum if (steps_nrec or n) else np.nan
            r[f"steps_nrec_{W}h"] = steps_nrec
            if n:
                cnt = np.bincount(seg, minlength=4)
                for k, lv in enumerate(ACT_LEVELS):
                    r[f"act_frac_{ACT_SHORT[lv]}_{W}h"] = cnt[k] / n
                r[f"act_mvpa_min_{W}h"] = float(cnt[2] + cnt[3])
            else:
                for lv in ACT_LEVELS:
                    r[f"act_frac_{ACT_SHORT[lv]}_{W}h"] = np.nan
                r[f"act_mvpa_min_{W}h"] = np.nan

            i0, i1 = win_idx(ox_t, te, W)
            seg = ox_v[i0:i1]
            r[f"spo2_mean_{W}h"] = seg.mean() if len(seg) else np.nan
            r[f"spo2_min_{W}h"] = seg.min() if len(seg) else np.nan
            r[f"spo2_n_{W}h"] = len(seg)

            i0, i1 = win_idx(hv_t, te, W)
            if i1 > i0:
                r[f"hrv_rmssd_mean_{W}h"] = hv_rmssd[i0:i1].mean()
                lf, hf = hv_lf[i0:i1].mean(), hv_hf[i0:i1].mean()
                r[f"hrv_lfhf_{W}h"] = lf / hf if hf > 0 else np.nan
            else:
                r[f"hrv_rmssd_mean_{W}h"] = np.nan
                r[f"hrv_lfhf_{W}h"] = np.nan
            r[f"hrv_n_{W}h"] = i1 - i0

        # -------- prior-night block ---------------------------------------
        j = np.searchsorted(sl_t, te, "left") - 1
        if j >= 0 and (te - sl_t[j]) / np.timedelta64(1, "h") <= SLEEP_LOOKBACK_H:
            r["sleep_score"] = float(sleep["overall_score"].iloc[j])
            r["deep_sleep_min"] = float(sleep["deep_sleep_in_minutes"].iloc[j])
            r["hours_since_sleep_onset"] = float((te - sl_t[j]) / np.timedelta64(1, "h"))
            ns = sl_t[j]
            ne = min(ns + np.timedelta64(NIGHT_LEN_H, "h"), te)
            a, b = np.searchsorted(hv_t, ns, "left"), np.searchsorted(hv_t, ne, "left")
            if b > a:
                r["hrv_rmssd_night"] = hv_rmssd[a:b].mean()
                r["hrv_rmssd_night_sd"] = hv_rmssd[a:b].std()
                lf, hf = hv_lf[a:b].mean(), hv_hf[a:b].mean()
                r["hrv_lfhf_night"] = lf / hf if hf > 0 else np.nan
                r["hrv_hf_night"] = hf
                r["hrv_n_night"] = b - a
            a, b = np.searchsorted(ox_t, ns, "left"), np.searchsorted(ox_t, ne, "left")
            if b > a:
                seg = ox_v[a:b]
                r["spo2_night_mean"] = seg.mean()
                r["spo2_night_min"] = seg.min()
                r["spo2_night_sd"] = seg.std()
                r["spo2_night_pct_lt90"] = 100 * (seg < 90).mean()
                r["spo2_n_night"] = len(seg)

        r["worn_24h"] = int(r["act_nmin_24h"] > 0)
        r["wear_frac_24h"] = r["act_nmin_24h"] / 1440.0
        r["has_prior_night"] = int("sleep_score" in r)

        # NB: the same-day device score is a whole-day aggregate. Its
        # UPDATED_AT is often later than the survey, so it can contain
        # post-survey information -- flagged, and excluded from the models.
        r["dev_stress_same_day"] = ds_map.get(tl_naive.date(), np.nan)
        u = ds_upd.get(tl_naive.date())
        r["dev_stress_leaky"] = (np.nan if u is None
                                 else int(np.datetime64(u.tz_convert("UTC")
                                                        .tz_localize(None), "ns") > te))
        r["dev_stress_prev_day"] = ds_map.get((tl_naive - pd.Timedelta(days=1)).date(),
                                              np.nan)
        rows.append(r)

df = pd.DataFrame(rows)

lead = ["subject", "university", "course", "semester", "undergraduate",
        "t_utc", "t_local", "date_local", "local_hour", "weekday", "is_weekend",
        "day_in_study", "week_in_study", "is_class_day", "in_class_window",
        "hours_to_class_start", "stress", "anxiety",
        "resp_latency_s", "resp_delay_s"]
night = [c for c in df.columns if "night" in c or c in
         ("sleep_score", "deep_sleep_min", "hours_since_sleep_onset")]
dev = ["dev_stress_same_day", "dev_stress_prev_day", "dev_stress_leaky"]
rest = [c for c in df.columns if c not in lead + night + dev]
df = df[lead + rest + night + dev]

df.to_csv(f"{OUT}/ssaqs_analysis_table.csv", index=False)
qdf = pd.DataFrame(quality)
qdf.to_csv(f"{OUT}/ssaqs_data_quality.csv", index=False)

print(f"rows={len(df)}  subjects={df.subject.nunique()}  cols={df.shape[1]}")
print("\nfeature coverage (% non-null), incomplete only:")
cov = (100 * df.notna().mean()).round(1).sort_values()
print(cov[cov < 100].to_string())
print("\nper-subject quality:\n", qdf.to_string(index=False))
