#!/usr/bin/env python3
"""check.py - data-quality checker for a volunteer weather-station pack.

Run:   python check.py <pack_folder>
Writes flags.csv into the folder you run it from, and prints a short summary.
Pack layout: stations.csv, data/Sxx.csv, gps/Sxx.nmea (optional), maintenance_log.md (optional).
Needs: pandas, numpy.  Nothing is hard-coded to station names, dates or the data pack.
All times are UTC.  See README.md for what each check does.
"""
import sys, os, re, math, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

NUM = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "wind_dir_deg", "rain_mm", "ghi_wm2"]
FLAGS = []
SCALE_H = 8000.0          # m, rough scale height used to compare pressure between elevations
SENTINELS = {-999.0, -9999.0, 9999.0, 65535.0, 32767.0, -32768.0}
UNIT_FACTORS = {"m/s reported as km/h": 1 / 3.6, "km/h reported as m/s": 3.6,
                "mph reported as km/h": 1 / 1.609, "km/h reported as mph": 1.609,
                "knots reported as km/h": 1 / 1.852, "km/h reported as knots": 1.852}


# ---------------------------------------------------------------- helpers
def tstr(t):
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def add(station, channel, start, end, issue, action, note):
    FLAGS.append(dict(station_id=station, channel=channel, start_utc=pd.Timestamp(start), end_utc=pd.Timestamp(end),
                      issue=issue, action=action, note=note))


def warn(msg):
    print("  [check skipped] " + msg, file=sys.stderr)


def load(pack, sid):
    path = os.path.join(pack, "data", sid + ".csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    d["ts"] = pd.to_datetime(d["timestamp_utc"].str.strip(), errors="coerce")
    for c in NUM:
        d[c] = pd.to_numeric(d[c].str.strip().replace("", np.nan), errors="coerce") if c in d else np.nan
    return d.dropna(subset=["ts"]).reset_index(drop=True)


def clean(d):
    return d.drop_duplicates("ts", keep="first").set_index("ts").sort_index()


def runs(mask):
    m = np.asarray(mask, bool).astype(np.int8)
    if not m.any():
        return []
    x = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(x == 1)[0], np.where(x == -1)[0] - 1))


def others_med(df, s):
    return df.drop(columns=s).median(axis=1)


def parse_gps(path):
    out = dict(diffs=[], lat=[], lon=[], alt=[])
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            lts, sent = line.strip().split(",", 1)
            lts = pd.to_datetime(lts.strip())
            body, cs = sent.strip().lstrip("$").rsplit("*", 1)
            x = 0
            for ch in body:
                x ^= ord(ch)
            if x != int(cs.strip()[:2], 16):
                continue
            f = body.split(",")
            if f[0].endswith("RMC") and f[2] == "A" and f[3]:
                g = pd.to_datetime(f[9] + f[1][:6], format="%d%m%y%H%M%S")
                out["diffs"].append((lts, (lts - g).total_seconds()))
                out["lat"].append((int(f[3][:2]) + float(f[3][2:]) / 60) * (1 if f[4] == "N" else -1))
                out["lon"].append((int(f[5][:3]) + float(f[5][3:]) / 60) * (1 if f[6] == "E" else -1))
            elif f[0].endswith("GGA") and f[6] not in ("", "0") and f[9]:
                out["alt"].append(float(f[9]))
        except Exception:
            continue
    return out


def est_shift(ts, g, refm):
    """shift (in 10-min steps) that makes this block's sunlight match the other stations'"""
    best_k, best_c, c0 = 0, -2.0, np.nan
    base = pd.DatetimeIndex(ts)
    for k in [0] + [k for k in range(-72, 73) if k != 0]:
        r = refm.reindex(base + pd.Timedelta(minutes=10 * k)).to_numpy()
        m = ~np.isnan(r) & ~np.isnan(g)
        if m.sum() < 100 or np.std(g[m]) == 0 or np.std(r[m]) == 0:
            continue
        c = np.corrcoef(g[m], r[m])[0, 1]
        if k == 0:
            c0 = c
            if c >= 0.9:
                return 0, c, c0
        if c > best_c:
            best_k, best_c = k, c
    return best_k, best_c, c0


def cpt(s, minseg):
    s = s.dropna()
    n = len(s)
    if n < 2 * minseg + 2:
        return None
    v = s.to_numpy(float)
    cs, cs2 = np.cumsum(v), np.cumsum(v * v)
    best = None
    for i in range(minseg, n - minseg + 1):
        n1, n2 = i, n - i
        m1, m2 = cs[i - 1] / n1, (cs[-1] - cs[i - 1]) / n2
        ss = (cs2[i - 1] - n1 * m1 * m1) + ((cs2[-1] - cs2[i - 1]) - n2 * m2 * m2)
        d = abs(m2 - m1) / (math.sqrt(max(ss, 0) / (n - 2)) or 1e-9)
        if best is None or d > best[0]:
            best = (d, s.index[i], m2 - m1)
    return best


def fit_models(y, minseg=4):
    """constant vs one step vs one ramp (flat, then linear) on a daily series"""
    y = y.dropna()
    n = len(y)
    if n < 2 * minseg + 2:
        return None
    tt = (y.index - y.index[0]).days.to_numpy(float)
    v = y.to_numpy(float)
    sse0 = ((v - v.mean()) ** 2).sum()
    if sse0 <= 1e-12:
        return None
    st, rp = None, None
    for i in range(minseg, n - minseg + 1):
        m1, m2 = v[:i].mean(), v[i:].mean()
        sse = ((v[:i] - m1) ** 2).sum() + ((v[i:] - m2) ** 2).sum()
        if st is None or sse < st[0]:
            st = (sse, i, m2 - m1)
        A = np.c_[np.ones(n), np.maximum(0, tt - tt[i])]
        coef = np.linalg.lstsq(A, v, rcond=None)[0]
        sse = ((v - A @ coef) ** 2).sum()
        if rp is None or sse < rp[0]:
            rp = (sse, i, coef[1] * (tt[-1] - tt[i]))
    return dict(r2_step=1 - st[0] / sse0, step_i=st[1], step_amp=st[2],
                r2_ramp=1 - rp[0] / sse0, ramp_i=rp[1], ramp_amp=rp[2], index=y.index)


def windows(mask, min_len, max_gap):
    rr = runs(mask)
    merged = []
    for a, b in rr:
        if merged and a - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a + 1 >= min_len]


# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print("usage: python check.py <pack_folder>")
        sys.exit(1)
    pack = sys.argv[1]

    meta = {}
    mp = os.path.join(pack, "stations.csv")
    if os.path.exists(mp):
        m = pd.read_csv(mp)
        for _, r in m.iterrows():
            meta[str(r["station_id"]).strip()] = (float(r["lat"]), float(r["lon"]), float(r["elevation_m"]))
    ids = list(meta) if meta else sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pack, "data", "*.csv")))
    raw = {s: load(pack, s) for s in ids}
    raw = {s: d for s, d in raw.items() if d is not None and len(d)}
    sids = list(raw)
    if not sids:
        print("no station data found")
        pd.DataFrame(columns=["station_id", "channel", "start_utc", "end_utc", "issue", "action", "note"]).to_csv("flags.csv", index=False)
        return
    log = []
    lp = os.path.join(pack, "maintenance_log.md")
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8", errors="replace"):
            mm = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(S\w+)\s+(.*)", line.strip())
            if mm:
                log.append((pd.Timestamp(mm.group(1)), mm.group(2), mm.group(3)))
    gps = {s: parse_gps(os.path.join(pack, "gps", s + ".nmea")) for s in sids}
    elev = {s: meta[s][2] if s in meta else 0.0 for s in sids}

    allts = pd.concat([d["ts"] for d in raw.values()])
    idx = pd.date_range(allts.min().floor("10min"), allts.max().ceil("10min"), freq="10min")

    # ---- 1. LOGGER CLOCK (sunlight must line up with the other stations) ----
    G = pd.DataFrame({s: clean(d)["ghi_wm2"].reindex(idx) for s, d in raw.items()})
    D = {}
    for s in sids:
        d = raw[s].copy()
        try:
            if len(sids) >= 3:
                seg = (d["ts"].diff().dt.total_seconds() < 0).cumsum().to_numpy()
                refm = others_med(G, s)
                for sg in np.unique(seg):
                    msk = seg == sg
                    k, c, c0 = est_shift(d.loc[msk, "ts"], d.loc[msk, "ghi_wm2"].to_numpy(), refm)
                    if k != 0 and c >= 0.8 and c - (c0 if not np.isnan(c0) else -1) > 0.1:
                        t0, t1 = d.loc[msk, "ts"].min(), d.loc[msk, "ts"].max()
                        jumps = []
                        dd = gps[s]["diffs"]
                        for i in range(1, len(dd)):
                            if abs(dd[i][1] - dd[i - 1][1]) > 60:
                                jumps.append((dd[i][0], dd[i][1]))
                        near = [j for j in jumps if abs((j[0] - t0).total_seconds()) < 86400 * 1.0]
                        gtxt = (f" GPS clock difference also jumps to {near[0][1]:.0f} s at logger time {tstr(near[0][0])}."
                                if near else "")
                        add(s, "all", t0, t1, f"logger clock off by {k * 10 / 60:+.1f} h", "correct",
                            f"Sunlight matches the other stations only after adding {k * 10} min (corr {c:.2f} vs {c0:.2f} as logged); "
                            f"add {k * 10} min to these rows.{gtxt}")
                        d.loc[msk, "ts"] = d.loc[msk, "ts"] + pd.Timedelta(minutes=10 * k)
        except Exception as e:
            warn(f"clock {s}: {e}")
        D[s] = d

    # ---- 2. REPEATED TIMES AND GAPS ----
    C = {}
    for s in sids:
        d = D[s]
        try:
            dup = d["ts"].duplicated(keep="first").to_numpy()
            if dup.any():
                orig = d[~dup].set_index("ts")
                for a, b in runs(dup):
                    rows = d.iloc[a:b + 1]
                    x = orig.loc[rows["ts"], NUM].to_numpy()
                    y = rows[NUM].to_numpy()
                    same = ((x == y) | (np.isnan(x) & np.isnan(y))).all(axis=1).mean()
                    if same >= 0.99:
                        add(s, "all", rows.ts.min(), rows.ts.max(), "duplicate rows (identical copy)", "exclude",
                            f"{len(rows)} rows repeat earlier timestamps with identical values; keep the first copy so sums are not double-counted.")
                    else:
                        add(s, "all", rows.ts.min(), rows.ts.max(), "repeated timestamps, different values", "exclude",
                            f"{len(rows)} rows reuse earlier timestamps but {100 * (1 - same):.0f}% differ; cannot tell which is right.")
            C[s] = clean(d)
            t = C[s].index.to_series()
            gap = t.diff().dt.total_seconds().div(60)
            for i in np.where(gap.to_numpy() > 60)[0]:
                a, b = t.iloc[i - 1], t.iloc[i]
                add(s, "all", a + pd.Timedelta(minutes=10), b - pd.Timedelta(minutes=10),
                    f"no data for {gap.iloc[i] / 60:.1f} h", "exclude",
                    f"{int(gap.iloc[i] / 10) - 1} readings missing; daily summaries for these days are incomplete.")
        except Exception as e:
            warn(f"duplicates/gaps {s}: {e}")
            C[s] = clean(d)

    # master 10-minute frames (pressure also reduced to a common elevation for comparing stations)
    M = {ch: pd.DataFrame({s: C[s][ch].reindex(idx) for s in sids}) for ch in NUM}
    red = np.array([math.exp(elev[s] / SCALE_H) for s in sids])
    MP = M["pressure_hpa"] * red                       # reduced pressure, same shape
    can_compare = len(sids) >= 4

    def span(s):
        return C[s].index.min(), C[s].index.max()

    def mask_flags(only_new_from=0):
        for f in FLAGS[only_new_from:]:
            if f["action"] != "exclude" or f["station_id"] not in sids:
                continue
            chs = NUM if f["channel"] == "all" else [f["channel"]]
            for ch in chs:
                if ch in M:
                    M[ch].loc[f["start_utc"]:f["end_utc"], f["station_id"]] = np.nan
                    if ch == "pressure_hpa":
                        MP.loc[f["start_utc"]:f["end_utc"], f["station_id"]] = np.nan

    n_before = len(FLAGS)

    # ---- 3. IMPOSSIBLE VALUES, SENTINELS, ONE-READING SPIKES ----
    try:
        for s in sids:
            for ch in ("temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "ghi_wm2"):
                v = M[ch][s]
                bad = v.isin(SENTINELS)
                if ch == "temp_c":
                    bad |= v.isin([-40.0, 85.0])
                if ch == "wind_kmh":
                    bad |= (v < 0) | (v > 250)
                if ch == "ghi_wm2":
                    bad |= (v < -10) | (v > 1500)
                if ch == "rh_pct":
                    bad |= v < 0
                if can_compare and ch in ("temp_c", "rh_pct", "pressure_hpa"):
                    X = (MP if ch == "pressure_hpa" else M[ch])
                    X = X - X.median()
                    R = X[s] - others_med(X, s)
                    base = R.rolling(13, center=True, min_periods=5).median()
                    thr = {"temp_c": 8.0, "rh_pct": 35.0, "pressure_hpa": 5.0}[ch]
                    bad |= (R - base).abs() > thr
                bad = bad.fillna(False).to_numpy()
                for a, b in runs(bad):
                    vals = v.iloc[a:b + 1]
                    t0, t1 = v.index[a], v.index[b]
                    add(s, ch, t0, t1, "impossible or one-off spike value" if b - a < 3 else "run of impossible values",
                        "exclude", f"{b - a + 1} reading(s), value {vals.iloc[0]:g}"
                        + (" (typical sensor error code)" if vals.iloc[0] in (-40.0, 85.0) or vals.iloc[0] in SENTINELS else "")
                        + "; neighbouring readings and the other stations are normal.")
    except Exception as e:
        warn(f"spikes: {e}")

    # ---- 4. FROZEN SENSORS ----
    try:
        for s in sids:
            for ch, n_min in (("temp_c", 18), ("rh_pct", 18), ("pressure_hpa", 24), ("wind_kmh", 18), ("ghi_wm2", 18)):
                v = M[ch][s].to_numpy()
                eq = (v[1:] == v[:-1]) & ~np.isnan(v[1:])
                for a, b in runs(eq):
                    n = b - a + 2
                    val = v[a]
                    if n < n_min:
                        continue
                    if ch == "rh_pct" and 99.5 <= val <= 100:
                        continue
                    if ch in ("wind_kmh", "ghi_wm2") and val == 0:
                        continue
                    t0, t1 = M[ch].index[a], M[ch].index[b + 1]
                    add(s, ch, t0, t1, "sensor frozen at one value", "exclude",
                        f"{n} readings ({n / 6:.0f} h) all exactly {val:g} while the other stations kept changing.")
    except Exception as e:
        warn(f"frozen: {e}")

    # ---- 5. RAIN GAUGE BLOCKED / DUMPED ----
    try:
        Rn = M["rain_mm"]
        day_sum = Rn.resample("1D").sum(min_count=1)
        day_cnt = Rn.resample("1D").count()
        for s in sids:
            big = Rn[s][Rn[s] >= 3.0]
            used = set()
            if can_compare:
                oth = others_med(day_sum, s)
                blocked = ((day_sum[s] == 0) & (day_cnt[s] >= 100) & (oth >= 2.0)).to_numpy()
                for a, b in runs(blocked):
                    d0, d1 = day_sum.index[a], day_sum.index[b] + pd.Timedelta(hours=23, minutes=50)
                    hourly_oth = others_med(Rn.resample("1h").sum(min_count=1), s)
                    first_rain = hourly_oth.loc[d0:d0 + pd.Timedelta(hours=23)]
                    first_rain = first_rain[first_rain > 0]
                    start = first_rain.index[0] if len(first_rain) else d0
                    later = big[(big.index > start) & (big.index <= d1 + pd.Timedelta(days=14))]
                    end, dump = d1, 0.0
                    if len(later):
                        t_first = later.index[0]
                        run_end = t_first
                        while run_end + pd.Timedelta(minutes=10) in later.index:
                            run_end += pd.Timedelta(minutes=10)
                        end, dump = run_end, later.loc[t_first:run_end].sum()
                        used.update(later.loc[t_first:run_end].index)
                    miss = day_sum.drop(columns=s).median(axis=1).loc[d0:d1].sum()
                    add(s, "rain_mm", start, end, "rain gauge blocked, then released", "exclude",
                        f"Other stations' median rain {miss:.1f} mm in this window; this gauge recorded 0, then "
                        + (f"{dump:.1f} mm in a few readings at {tstr(end)} (stored water released). " if dump else "nothing until the end. ")
                        + "Period total is roughly right but daily values are wrong.")
            for t, val in big.items():
                if t not in used:
                    add(s, "rain_mm", t, t, "implausible rain in 10 minutes", "exclude", f"{val:.1f} mm in one 10-minute reading.")
    except Exception as e:
        warn(f"rain: {e}")

    mask_flags(n_before)
    n_before = len(FLAGS)

    # ---- 6. LONG STRETCHES FAR FROM THE OTHER STATIONS ----
    def hourly(frame):
        return frame.resample("1h").mean()

    HT, HR, HP, HW, HG = (hourly(M["temp_c"]), hourly(M["rh_pct"]), hourly(MP), hourly(M["wind_kmh"]), hourly(M["ghi_wm2"]))

    def dev_of(H, s):
        A = H - H.median()                      # each station's own typical level removed
        return A[s] - others_med(A, s)

    def adj(d):
        return d - d.groupby(d.index.hour).transform("median")

    if can_compare:
        try:
            for s in sids:
                for ch, H, thr, min_len, gap in (("rh_pct", HR, 25.0, 24, 12), ("temp_c", HT, 6.0, 12, 6), ("pressure_hpa", HP, 4.0, 12, 6)):
                    d = dev_of(H, s)
                    d = d - d.rolling(216, center=True, min_periods=72).median()     # ignore slow level changes (steps/drifts are tested later)
                    d = adj(d)
                    for a, b in windows((d.abs() > thr).to_numpy(), min_len, gap):
                        sign = np.sign(d.iloc[a:b + 1].mean())
                        k = 0
                        while a > 0 and k < 12 and sign * d.iloc[a - 1] > 0.4 * thr:
                            a -= 1; k += 1
                        k = 0
                        while b < len(d) - 1 and k < 12 and sign * d.iloc[b + 1] > 0.4 * thr:
                            b += 1; k += 1
                        add(s, ch, d.index[a], d.index[b] + pd.Timedelta(minutes=50),
                            "sensor far from the other stations for a long stretch", "exclude",
                            f"{b - a + 1} h with mean difference {d.iloc[a:b + 1].mean():+.1f} from the other stations "
                            f"(normal day-to-day scatter is far smaller)."
                            + (" Readings above 100% are physically impossible." if ch == "rh_pct" and M["rh_pct"][s].loc[d.index[a]:d.index[b]].max() > 100 else ""))
        except Exception as e:
            warn(f"long departures: {e}")
    mask_flags(n_before)
    n_before = len(FLAGS)

    # isolated readings above 100% humidity that are not part of a bigger fault
    try:
        for s in sids:
            v = M["rh_pct"][s]
            over = v[v > 100]
            if over.empty:
                continue
            grp = (over.index.to_series().diff() > pd.Timedelta(hours=6)).cumsum()
            for _, g in over.groupby(grp.to_numpy()):
                add(s, "rh_pct", g.index.min(), g.index.max(), "humidity above 100%", "correct",
                    f"{len(g)} reading(s) up to {g.max():g}; sensor overshoot at saturation, clip to 100.")
    except Exception as e:
        warn(f"rh>100: {e}")

    # ---- 7. LASTING STEPS AND DRIFTS (own minus the other stations, daily) ----
    HT, HR, HP, HW, HG = (hourly(M["temp_c"]), hourly(M["rh_pct"]), hourly(MP), hourly(M["wind_kmh"]), hourly(M["ghi_wm2"]))
    cands = []
    if can_compare:
        try:
            for s in sids:
                end_t = span(s)[1]
                sun = others_med(HG, s) > 150
                series = {}
                dT = dev_of(HT, s)
                series["temp_c day"] = (dT[sun].resample("1D").mean(), "temp_c", 1.5, "add", HT)
                series["temp_c night"] = (dT[~sun].resample("1D").mean(), "temp_c", 1.5, "add", HT)
                series["rh_pct"] = (dev_of(HR, s).resample("1D").mean(), "rh_pct", 10.0, "add", HR)
                series["pressure_hpa"] = (dev_of(HP, s).resample("1D").mean(), "pressure_hpa", 2.0, "add", HP)
                Wn = (HW / HW.mean()).resample("1D").mean()
                refw = others_med(Wn, s)
                rw = np.log((Wn[s] / refw).where((refw > 0.05) & (Wn[s] > 0.01)))
                series["wind_kmh"] = (rw, "wind_kmh", math.log(1.25), "log", HW)
                Gn = (HG / HG.mean()).resample("1D").mean()
                refg = others_med(Gn, s)
                rg = np.log((Gn[s] / refg).where((refg > 0.05) & (Gn[s] > 0.01)))
                series["ghi_wm2"] = (rg, "ghi_wm2", math.log(1.12), "log", HG)
                for name, (y, ch, thr, mode, H) in series.items():
                    fm = fit_models(y)
                    if not fm:
                        continue
                    kind = None
                    if fm["r2_ramp"] >= 0.5 and fm["r2_ramp"] - fm["r2_step"] >= 0.08 and abs(fm["ramp_amp"]) >= thr:
                        kind = "ramp"
                    elif fm["r2_step"] >= 0.5 and abs(fm["step_amp"]) >= thr:
                        kind = "step"
                    if kind:
                        i = fm["ramp_i"] if kind == "ramp" else fm["step_i"]
                        amp = fm["ramp_amp"] if kind == "ramp" else fm["step_amp"]
                        t = fm["index"][i]
                        if kind == "step":                                   # refine to the hour
                            h = (dev_of(H, s) if mode == "add" else dev_of(H, s)).loc[t - pd.Timedelta(days=2):t + pd.Timedelta(days=3)]
                            r = cpt(h, 6)
                            if r:
                                t = r[1]
                        cands.append(dict(s=s, name=name, ch=ch, mode=mode, kind=kind, t=t, amp=amp, end=end_t,
                                          r2=fm["r2_ramp"] if kind == "ramp" else fm["r2_step"]))
        except Exception as e:
            warn(f"steps/drifts: {e}")
    # a station's step moves the median that the others are compared with: keep the culprit, drop mirror images
    keep_c = []
    for c in sorted(cands, key=lambda c: -c["r2"] * (abs(c["amp"]) / (1 if c["mode"] == "add" else 0.1))):
        mirror = any(k["ch"] == c["ch"] and k["name"].split()[0] == c["name"].split()[0] and k["kind"] == "step" and c["kind"] == "step"
                     and abs((k["t"] - c["t"]).days) <= 3 and np.sign(k["amp"]) != np.sign(c["amp"])
                     and abs(c["amp"]) <= 1.3 * abs(k["amp"]) for k in keep_c)
        if not mirror:
            keep_c.append(c)
    for c in keep_c:
        s, ch, t, amp = c["s"], c["ch"], c["t"], c["amp"]
        if c["kind"] == "ramp":
            add(s, ch, t, c["end"], "growing bias" + (" in daytime only" if c["name"].endswith("day") else
                                                       (" at night only" if c["name"].endswith("night") else "")), "exclude",
                f"Difference from the other stations grows steadily from about {tstr(t)}, reaching {amp:+.1f}"
                f"{'' if c['mode'] == 'add' else ' (log-ratio)'} by the end of the data"
                + ("; night readings stay close to the others, so it follows sunlight." if c["name"].endswith("day") else "."))
        elif c["mode"] == "add":
            unit = {"temp_c": "C", "rh_pct": "%", "pressure_hpa": "hPa"}[ch]
            add(s, ch, t, c["end"], "lasting offset appears (instrument change?)", "correct",
                f"Difference from the other stations jumps {amp:+.1f} {unit} at {tstr(t)} and stays; subtract {amp:.1f} {unit} from this point on.")
        else:
            ratio = math.exp(amp)
            if ch == "wind_kmh":
                best = min(UNIT_FACTORS.items(), key=lambda kv: abs(math.log(ratio / kv[1])))
                if abs(math.log(ratio / best[1])) < 0.1:
                    add(s, ch, t, c["end"], f"wind speed unit changed ({best[0]})", "correct",
                        f"Readings relative to the other stations change by x{ratio:.2f} at {tstr(t)} with no offset; "
                        f"that matches {best[0]}, so multiply by {1 / best[1]:.3g} from this point on.")
                    continue
            add(s, ch, t, c["end"], f"lasting scale change (x{ratio:.2f})", "exclude",
                f"Readings relative to the other stations change by x{ratio:.2f} at {tstr(t)} and the factor is not a known unit change.")

    # ---- 8. GPS LOCATION VERSUS stations.csv ----
    try:
        for s in sids:
            g = gps[s]
            if s not in meta or not g["lat"]:
                continue
            la, lo, el = meta[s]
            mla, mlo = float(np.median(g["lat"])), float(np.median(g["lon"]))
            dn = (mla - la) * 111.19
            de = (mlo - lo) * 111.19 * math.cos(math.radians(mla))
            dist = math.hypot(dn, de)
            alt = float(np.median(g["alt"])) if g["alt"] else None
            a0, a1 = span(s)
            if dist > 1.0:
                add(s, "metadata", a0, a1, "station position on the install sheet disagrees with GPS", "correct",
                    f"GPS median {mla:.4f},{mlo:.4f} is {dist:.1f} km from the sheet ({dn:+.1f} km N, {de:+.1f} km E)"
                    + (f"; GPS height {alt:.0f} m vs sheet {el:.0f} m" if alt is not None else "") + "; the sheet is the likely typo.")
            elif alt is not None and abs(alt - el) > 50:
                add(s, "metadata", a0, a1, "station elevation on the install sheet disagrees with GPS", "correct",
                    f"GPS height {alt:.0f} m vs sheet {el:.0f} m.")
    except Exception as e:
        warn(f"gps: {e}")

    # ---- 9. THINGS THAT LOOK ODD BUT ARE REAL ----
    if can_compare:
        try:
            for s in sids:
                a0, a1 = span(s)
                others = [x for x in sids if x != s]
                de_ = elev[s] - float(np.median([elev[x] for x in others]))
                dT = HT[s].median() - float(np.median([HT[x].median() for x in others]))
                if abs(de_) > 300 and abs(dT) > 3 and abs(dT - (-6.5 * de_ / 1000)) < 2.5:
                    add(s, "temp_c", a0, a1, "runs much colder/warmer than the others", "keep",
                        f"Median offset {dT:+.1f} C matches the elevation difference ({de_:+.0f} m, about {-6.5 * de_ / 1000:+.1f} C); normal.")
                raw_p = M["pressure_hpa"][s].median() - float(np.median([M["pressure_hpa"][x].median() for x in others]))
                red_p = HP[s].median() - float(np.median([HP[x].median() for x in others]))
                if abs(raw_p) > 50 and abs(red_p) < 0.4 * abs(raw_p):
                    add(s, "pressure_hpa", a0, a1, "pressure far from the others", "keep",
                        f"Station pressure differs by {raw_p:+.0f} hPa but only {red_p:+.0f} hPa once reduced to a common elevation; normal.")
                dt, dr = adj(dev_of(HT, s)), adj(dev_of(HR, s))
                for a, b in windows((dt.abs() > 8).to_numpy(), 3, 1):
                    if np.sign(dt.iloc[a:b + 1].mean()) != np.sign(dr.iloc[a:b + 1].mean()) and abs(dr.iloc[a:b + 1].mean()) > 15:
                        add(s, "temp_c", dt.index[a], dt.index[b] + pd.Timedelta(minutes=50), "temperature far from the others", "keep",
                            f"Temperature {dt.iloc[a:b + 1].mean():+.1f} C and humidity {dr.iloc[a:b + 1].mean():+.0f}% move in opposite directions "
                            f"for {b - a + 1} h, as real warm-dry or cold-moist air does; sensors look healthy.")
        except Exception as e:
            warn(f"keep: {e}")

    # ---- 10. TIDY UP, ATTACH LOG NOTES, WRITE ----
    fl = []
    for f in FLAGS:
        if f["action"] != "keep":
            contained = any(g is not f and g["action"] == "exclude" and g["station_id"] == f["station_id"]
                            and (g["channel"] == f["channel"] or g["channel"] == "all")
                            and g["start_utc"] <= f["start_utc"] and g["end_utc"] >= f["end_utc"]
                            and (g["end_utc"] - g["start_utc"]) > (f["end_utc"] - f["start_utc"]) for g in FLAGS)
            if contained:
                continue
        fl.append(f)
    fl.sort(key=lambda f: (f["station_id"], f["channel"], f["issue"], f["action"], f["start_utc"]))
    merged = []
    for f in fl:
        p = merged[-1] if merged else None
        if (p and p["action"] == "exclude" and f["action"] == "exclude" and p["station_id"] == f["station_id"]
                and p["channel"] == f["channel"] and p["issue"] == f["issue"]
                and f["start_utc"] - p["end_utc"] <= pd.Timedelta(minutes=30)):
            p["end_utc"] = max(p["end_utc"], f["end_utc"])
            p["note"] = p["note"] + " (adjacent stretches merged)" if "merged" not in p["note"] else p["note"]
        else:
            merged.append(dict(f))
    fl = merged
    seen, out = set(), []
    for f in fl:
        key = (f["station_id"], f["channel"], f["start_utc"], f["end_utc"], f["action"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    for f in out:
        near = [e for e in log if e[1] == f["station_id"] and (abs((e[0] - f["start_utc"].normalize()).days) <= 2
                                                              or abs((e[0] - f["end_utc"].normalize()).days) <= 2)]
        if near:
            f["note"] += f" Log {near[0][0].strftime('%m-%d')}: \"{near[0][2][:70]}\""
        f["start_utc"], f["end_utc"] = tstr(f["start_utc"]), tstr(f["end_utc"])
    df = pd.DataFrame(out, columns=["station_id", "channel", "start_utc", "end_utc", "issue", "action", "note"])
    df = df.sort_values(["station_id", "start_utc", "channel"]).reset_index(drop=True)
    df.to_csv("flags.csv", index=False)

    print(f"{len(df)} flags written to flags.csv  ({', '.join(f'{k} {v}' for k, v in df['action'].value_counts().items())})")
    for _, r in df.iterrows():
        print(f"{r.station_id} {r.channel:12s} {r.start_utc[5:]} -> {r.end_utc[5:]}  {r.action:7s} {r.issue}")


if __name__ == "__main__":
    main()