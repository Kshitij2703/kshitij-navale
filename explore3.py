#!/usr/bin/env python3
"""explore3.py - closes the open questions from explore2 and sweeps every station.
Usage: python explore3.py <pack_folder>
Prints a compact report and saves it to explore3_output.txt (in the folder you run it from).
Needs: pandas, numpy. All times are UTC (logger clock after the S06 correction).
"""
import sys, os, re, math, glob
import numpy as np
import pandas as pd

NUM = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "wind_dir_deg", "rain_mm", "ghi_wm2"]
LOG = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def ft(t):
    return "NaT" if pd.isna(t) else pd.Timestamp(t).strftime("%m-%d %H:%M")


def load(pack, sid):
    path = os.path.join(pack, "data", sid + ".csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    d["ts"] = pd.to_datetime(d["timestamp_utc"].str.strip(), errors="coerce")
    for c in NUM:
        d[c] = pd.to_numeric(d[c].str.strip().replace("", np.nan), errors="coerce") if c in d else np.nan
    return d.reset_index(drop=True)


def clean(d):
    return d.dropna(subset=["ts"]).drop_duplicates("ts", keep="first").set_index("ts").sort_index()


def runs(mask):
    m = np.asarray(mask, bool).astype(np.int8)
    if not m.any():
        return []
    x = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(x == 1)[0], np.where(x == -1)[0] - 1))


def others_med(df, s):
    return df.drop(columns=s).median(axis=1)


def est_shift(ts, g, refm):
    best_k, best_c, c0 = 0, -2.0, np.nan
    base = pd.DatetimeIndex(ts)
    for k in range(-72, 73):
        r = refm.reindex(base + pd.Timedelta(minutes=10 * k)).to_numpy()
        m = ~np.isnan(r) & ~np.isnan(g)
        if m.sum() < 100 or np.std(g[m]) == 0 or np.std(r[m]) == 0:
            continue
        c = np.corrcoef(g[m], r[m])[0, 1]
        if k == 0:
            c0 = c
        if c > best_c:
            best_k, best_c = k, c
    return best_k, best_c, c0


def cpt(s, minseg):
    """best single step in a series: returns (strength, time of first point after step, step, mean before, mean after)"""
    s = s.dropna()
    n = len(s)
    if n < 2 * minseg + 2:
        return None
    v = s.to_numpy(float)
    cs, cs2 = np.cumsum(v), np.cumsum(v * v)
    best = None
    for i in range(minseg, n - minseg + 1):
        n1, n2 = i, n - i
        m1 = cs[i - 1] / n1
        m2 = (cs[-1] - cs[i - 1]) / n2
        ss = (cs2[i - 1] - n1 * m1 * m1) + ((cs2[-1] - cs2[i - 1]) - n2 * m2 * m2)
        sd = math.sqrt(max(ss, 0) / (n - 2)) or 1e-9
        d = abs(m2 - m1) / sd
        if best is None or d > best[0]:
            best = (d, s.index[i], m2 - m1, m1, m2)
    return best


def fit(x, y):
    m = x.notna() & y.notna()
    if m.sum() < 24:
        return (np.nan, np.nan, np.nan)
    b, a = np.polyfit(x[m], y[m], 1)
    mx = x[m].mean()
    return a, b, (y[m].mean() / mx if mx else np.nan)


def main():
    if len(sys.argv) < 2:
        print("usage: python explore3.py <pack_folder>")
        sys.exit(1)
    pack = sys.argv[1]
    mp = os.path.join(pack, "stations.csv")
    ids = ([str(x).strip() for x in pd.read_csv(mp)["station_id"]] if os.path.exists(mp)
           else sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pack, "data", "*.csv"))))
    raw = {s: load(pack, s) for s in ids}
    raw = {s: d for s, d in raw.items() if d is not None}
    sids = list(raw)
    allts = pd.concat([d["ts"] for d in raw.values()]).dropna()
    idx = pd.date_range(allts.min().floor("10min"), allts.max().ceil("10min"), freq="10min")
    yr = idx[0].year

    # ---- clock-corrected, de-duplicated series (same logic as explore2) ----
    G = pd.DataFrame({s: clean(d)["ghi_wm2"].reindex(idx) for s, d in raw.items()})
    C, joins = {}, {}
    for s in sids:
        d = raw[s].copy()
        seg = (d["ts"].diff().dt.total_seconds() < 0).cumsum().to_numpy()
        refm = others_med(G, s)
        for sg in np.unique(seg):
            m = seg == sg
            k, c, c0 = est_shift(d.loc[m, "ts"], d.loc[m, "ghi_wm2"].to_numpy(), refm)
            if k != 0 and c - (c0 if not np.isnan(c0) else -1) > 0.05:
                d.loc[m, "ts"] = d.loc[m, "ts"] + pd.Timedelta(minutes=10 * k)
                joins[s] = (d.loc[m, "ts"].min(), k)
        C[s] = clean(d)
    H = {ch: pd.DataFrame({s: C[s][ch].resample("1h").mean() for s in sids}) for ch in NUM if ch != "rain_mm"}
    Rh = pd.DataFrame({s: C[s]["rain_mm"].resample("1h").sum(min_count=1) for s in sids})

    def visit(s, ch, lo, hi, label):
        if s not in sids:
            return
        own, ref = H[ch][s], others_med(H[ch], s)
        r = cpt((own - ref).loc[lo:hi], 12)
        if r is None:
            P(f"{label}: {s} {ch}: not enough data in window")
            return
        dd, t, step, m1, m2 = r
        h = pd.Timedelta(hours=1)
        fb, fa = fit(ref.loc[:t - h], own.loc[:t - h]), fit(ref.loc[t:], own.loc[t:])
        P(f"{label}: {s} {ch} step {step:+.2f} at {ft(t)} (strength {dd:.1f}); "
          f"before: own={fb[0]:+.1f}+{fb[1]:.2f}*others, mean ratio {fb[2]:.2f} | "
          f"after: own={fa[0]:+.1f}+{fa[1]:.2f}*others, mean ratio {fa[2]:.2f}")

    # ---------- A. LOG VISITS ----------
    P("=== A. EXACT SIZE/TIME OF THE SUSPECTED CHANGES (own vs median of others, full-period fits each side) ===")
    P("(step = change in own-minus-others; mean ratio = own mean / others' mean; 1.00 = same as others)")
    visit("S02", "pressure_hpa", f"{yr}-03-19", f"{yr}-03-24", "pressure board swap")
    visit("S07", "wind_kmh", f"{yr}-03-24", f"{yr}-03-28", "new anemometer")
    visit("S07", "wind_dir_deg", f"{yr}-03-24", f"{yr}-03-28", "  (vane, raw degrees)")
    visit("S04", "temp_c", f"{yr}-03-12", f"{yr}-03-18", "repaint")
    visit("S04", "ghi_wm2", f"{yr}-03-12", f"{yr}-03-18", "repaint")
    visit("S04", "rh_pct", f"{yr}-03-12", f"{yr}-03-18", "repaint")
    if joins:
        P("S06-style clock join continuity (own minus others, 12 h before vs 12 h after the join; should not jump):")
        for s, (T, k) in joins.items():
            for ch in ("temp_c", "pressure_hpa", "rh_pct"):
                d = H[ch][s] - others_med(H[ch], s)
                b, a = d.loc[T - pd.Timedelta(hours=12): T - pd.Timedelta(hours=1)], d.loc[T: T + pd.Timedelta(hours=11)]
                P(f"   {s} {ch}: join {ft(T)}  before {b.mean():+.2f}  after {a.mean():+.2f}  (hourly sd {d.std():.2f})")

    # ---------- B. S05 HUMIDITY ----------
    P("\n=== B. S05 HUMIDITY: own/others' median %, every 3 h (day-by-day) ===")
    if "S05" in sids:
        own, ref = H["rh_pct"]["S05"], others_med(H["rh_pct"], "S05")
        for day in pd.date_range(f"{yr}-03-07", f"{yr}-03-11"):
            row = []
            for h in range(0, 24, 3):
                t = day + pd.Timedelta(hours=h)
                a, b = own.get(t, np.nan), ref.get(t, np.nan)
                row.append(f"{h:02d}h {a:.0f}/{b:.0f}" if pd.notna(a) and pd.notna(b) else f"{h:02d}h  ./.")
            P(day.strftime("%m-%d ") + "  ".join(row))
        r10 = C["S05"]["rh_pct"]
        over = r10[r10 > 100]
        perday = over.groupby(over.index.date).size()
        P("readings above 100% per day: " + (", ".join(f"{k.strftime('%m-%d')}:{v}" for k, v in perday.items()) or "none"))
        P(f"last reading above 100%: {ft(over.index.max()) if len(over) else 'none'}; latest S05 reading: {ft(r10.index.max())}")

    # ---------- C. RAIN ----------
    P("\n=== C. RAIN FUNNEL BLOCKAGE (stations with a 10-min value above 3 mm) ===")
    cumh = Rh.fillna(0).cumsum()
    any_c = False
    for s in sids:
        r10 = C[s]["rain_mm"]
        big = r10[r10 > 3]
        if big.empty:
            continue
        any_c = True
        dev = cumh[s] - others_med(cumh, s)
        first = dev[dev < -3].index.min()
        P(f"{s}: big readings: " + ", ".join(f"{ft(t)} {v:.1f}" for t, v in big.items()) + f"  (sum {big.sum():.1f} mm)")
        if pd.notna(first):
            before = dev.loc[:first]
            start = before[before >= -0.5].index.max()
            P(f"   deficit vs others first below -3 mm at {ft(first)}; last hour within 0.5 mm: {ft(start)} "
              f"(own {cumh[s].loc[start]:.1f} mm vs others {others_med(cumh, s).loc[start]:.1f} mm so far)")
            seen = Rh.drop(columns=s).median(axis=1)
            w = pd.DataFrame({"own": Rh[s], "others": seen}).loc[start:first]
            w = w[(w["others"] >= 0.2) & (w["own"].fillna(0) == 0)]
            P(f"   hours in that stretch where others recorded rain but {s} recorded none: {len(w)}; first {ft(w.index.min()) if len(w) else '-'}")
            back = dev.loc[big.index.max():]
            ok = back[back >= -3].index.min()
            P(f"   deficit back within 3 mm at {ft(ok)}; deficit just before the dump {dev.loc[:big.index.min()].iloc[-2]:.1f} mm")
        dd = pd.DataFrame({"own": Rh[s].resample("1D").sum(), "others": Rh.drop(columns=s).median(axis=1).resample("1D").sum()})
        lo, hi = big.index.min() - pd.Timedelta(days=6), big.index.min() + pd.Timedelta(days=1)
        P("   daily mm own/others: " + "  ".join(f"{t.strftime('%m-%d')} {r.own:.1f}/{r.others:.1f}" for t, r in dd.loc[lo:hi].iterrows()))
    if not any_c:
        P("none")

    # ---------- D. OUTLIERS AND FROZEN STRETCHES ----------
    P("\n=== D. ALL IMPOSSIBLE READINGS AND FROZEN STRETCHES (12+ readings = 2 h) ===")
    n = 0
    for s in sids:
        d = raw[s]
        for ch, lo, hi in (("temp_c", -30, 45), ("rh_pct", 0, 100), ("wind_kmh", 0, 150), ("ghi_wm2", 0, 1400)):
            v = d[ch].to_numpy()
            rr = runs((v < lo) | (v > hi))
            if rr:
                txt = ", ".join(f"{ft(d.ts[a])} {v[a]:g}" + (f"(x{b - a + 1})" if b > a else "") for a, b in rr[:8])
                P(f"{s} {ch} out of range [{len(rr)}]: {txt}" + (" ..." if len(rr) > 8 else ""))
                n += 1
        for ch in ("temp_c", "rh_pct", "pressure_hpa", "wind_kmh"):
            v = d[ch].to_numpy()
            eq = (v[1:] == v[:-1]) & ~np.isnan(v[1:])
            for a, b in runs(eq):
                k = b - a + 2
                if k >= 12 and not (ch == "rh_pct" and 99.5 <= v[a] <= 100):
                    P(f"{s} {ch} frozen: {ft(d.ts[a])} -> {ft(d.ts[b + 1])} ({k} readings, {k / 6:.1f} h) at {v[a]:g}")
                    n += 1
    if n == 0:
        P("none")

    # ---------- E. S04 DETAILS ----------
    P("\n=== E. S04: REPAINT (03-15) DAY/NIGHT TEST AND 04-11 ===")
    if "S04" in sids:
        d = H["temp_c"]["S04"] - others_med(H["temp_c"], "S04")
        sun = others_med(H["ghi_wm2"], "S04") > 150
        wins = {"7 days before": (f"{yr}-03-08", f"{yr}-03-14"), "7 days after": (f"{yr}-03-16", f"{yr}-03-22")}
        for nm, (a, b) in wins.items():
            x, sx = d.loc[a:b], sun.loc[a:b]
            gs = H["ghi_wm2"]["S04"].loc[a:b].resample("1D").sum() / others_med(H["ghi_wm2"], "S04").loc[a:b].resample("1D").sum()
            P(f"  {nm}: temp minus others: daytime {x[sx].mean():+.2f}, night {x[~sx].mean():+.2f}; sunlight own/others {gs.mean():.2f}")
        t = d.loc[f"{yr}-04-10 12:00":f"{yr}-04-11 21:00"]
        P(f"  04-10 12:00 -> 04-11 21:00 (usual offset {d.median():+.1f}), every 3 h: "
          + " ".join(f"{ft(i)[6:]}:{v:+.1f}" for i, v in t.iloc[::3].items()))

    # ---------- F. SWEEP ----------
    P("\n=== F. SWEEP OF EVERY STATION: biggest lasting step per channel (daily, minimum 4 days each side) ===")
    P("strength > 3.5 shown. 'x' = ratio change (wind, sunlight), others are additive; 'log' = nearest maintenance-log visit")
    ev = []
    lp = os.path.join(pack, "maintenance_log.md")
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(S\d+)\s+", line.strip())
            if m:
                ev.append((pd.Timestamp(m.group(1)), m.group(2)))
    Dm = {ch: H[ch].resample("1D").mean() for ch in H}
    # wind direction: vectors, only when wind > 2 km/h
    U, V = {}, {}
    for s in sids:
        x = C[s]
        th = np.deg2rad(x["wind_dir_deg"])
        ok = (x["wind_kmh"] > 2) & th.notna()
        U[s] = np.cos(th).where(ok).resample("1h").mean()
        V[s] = np.sin(th).where(ok).resample("1h").mean()
    U, V = pd.DataFrame(U), pd.DataFrame(V)
    rows = []
    for s in sids:
        for ch, mode in (("temp_c", "add"), ("rh_pct", "add"), ("pressure_hpa", "add"), ("wind_kmh", "log"),
                         ("ghi_wm2", "log"), ("wind_dir_deg", "dir")):
            if mode == "add":
                ser = Dm[ch][s] - others_med(Dm[ch], s)
            elif mode == "log":
                ref = others_med(Dm[ch], s)
                ser = np.log((Dm[ch][s] / ref).where((ref > 1) & (Dm[ch][s] > 0.01)))
            else:
                ru, rv = U.drop(columns=s).mean(axis=1), V.drop(columns=s).mean(axis=1)
                ok = (np.hypot(U[s], V[s]) > 0.3) & (np.hypot(ru, rv) > 0.3)
                diff = ((np.degrees(np.arctan2(V[s], U[s]) - np.arctan2(rv, ru)) + 180) % 360 - 180).where(ok)
                ser = diff.resample("1D").median()
            r = cpt(ser, 4)
            if r and r[0] > 3.5:
                step = (math.exp(r[2]) if mode == "log" else r[2])
                near = [f"log {e[0].strftime('%m-%d')}" for e in ev if e[1] == s and abs((e[0] - r[1]).days) <= 2]
                rows.append((r[0], s, ch, r[1], step, mode, near))
    rows.sort(reverse=True)
    for d_, s, ch, t, step, mode, near in rows[:25]:
        txt = f"x{step:.2f}" if mode == "log" else f"{step:+.2f}"
        P(f"{s} {ch:13s} from {t.strftime('%m-%d')} {txt:>7} (strength {d_:.1f}) {' '.join(near)}")
    quiet = [s for s in sids if not any(r[1] == s for r in rows)]
    P("no lasting step above 3.5 in any channel:", ", ".join(quiet) if quiet else "none")

    with open("explore3_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()