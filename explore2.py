#!/usr/bin/env python3
"""explore2.py - targeted checks after explore.py.
Usage: python explore2.py <pack_folder>
Prints a compact report and saves it to explore2_output.txt (in the folder you run it from).
Needs: pandas, numpy.
"""
import sys, os, re, math
import numpy as np
import pandas as pd

CH = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "ghi_wm2"]
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
        if c in d:
            d[c] = pd.to_numeric(d[c].str.strip().replace("", np.nan), errors="coerce")
        else:
            d[c] = np.nan
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


def parse_gps(path):
    lat, lon = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            _, sent = line.strip().split(",", 1)
            body, cs = sent.strip().lstrip("$").rsplit("*", 1)
            x = 0
            for ch in body:
                x ^= ord(ch)
            if x != int(cs.strip()[:2], 16):
                continue
            f = body.split(",")
            if f[0].endswith("RMC") and f[2] == "A" and f[3]:
                lat.append((int(f[3][:2]) + float(f[3][2:]) / 60) * (1 if f[4] == "N" else -1))
                lon.append((int(f[5][:3]) + float(f[5][3:]) / 60) * (1 if f[6] == "E" else -1))
        except Exception:
            continue
    return lat, lon


def main():
    if len(sys.argv) < 2:
        print("usage: python explore2.py <pack_folder>")
        sys.exit(1)
    pack = sys.argv[1]
    mp = os.path.join(pack, "stations.csv")
    meta = pd.read_csv(mp) if os.path.exists(mp) else None
    if meta is not None:
        ids = [str(x).strip() for x in meta["station_id"]]
    else:
        import glob
        ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pack, "data", "*.csv")))
    raw = {s: load(pack, s) for s in ids}
    raw = {s: d for s, d in raw.items() if d is not None}
    sids = list(raw)
    allts = pd.concat([d["ts"] for d in raw.values()]).dropna()
    idx = pd.date_range(allts.min().floor("10min"), allts.max().ceil("10min"), freq="10min")

    # ---------- 1. CLOCK / DUPLICATES / GAPS ----------
    P("=== 1. CLOCK, REPEATED TIMES, GAPS ===")
    rawc = {s: clean(d) for s, d in raw.items()}
    G = pd.DataFrame({s: rawc[s]["ghi_wm2"].reindex(idx) for s in sids})
    shifts = {}      # station -> list of (row_mask, k)
    quiet = []
    for s in sids:
        d = raw[s]
        seg = (d["ts"].diff().dt.total_seconds() < 0).cumsum().to_numpy()
        refm = others_med(G, s)
        info, applied = [], []
        for sg in np.unique(seg):
            m = seg == sg
            k, c, c0 = est_shift(d.loc[m, "ts"], d.loc[m, "ghi_wm2"].to_numpy(), refm)
            rows = np.where(m)[0]
            info.append((rows[0], rows[-1], d.loc[m, "ts"].min(), d.loc[m, "ts"].max(), k, c, c0))
            if k != 0 and c - (c0 if not np.isnan(c0) else -1) > 0.05:
                applied.append((m, k))
        shifts[s] = applied
        dup2 = d["ts"].notna() & d["ts"].duplicated(keep="first")
        gaps = d["ts"].diff().dt.total_seconds().div(60)
        gp = gaps[gaps > 60]
        if len(info) == 1 and not dup2.any() and gp.empty and info[0][4] == 0:
            quiet.append(s)
            continue
        P(f"{s}: sunlight-clock check per block (rows, logger times, best shift to match other stations):")
        for r0, r1, t0, t1, k, c, c0 in info:
            P(f"   rows {r0}-{r1} {ft(t0)}->{ft(t1)}: add {k * 10:+d} min (corr {c:.2f}; as-is {c0:.2f})")
        if dup2.any():
            orig = d[~dup2].set_index("ts")
            second = d[dup2]
            a = orig.loc[second["ts"], NUM].to_numpy()
            b = second[NUM].to_numpy()
            same = ((a == b) | (np.isnan(a) & np.isnan(b))).all(axis=1)
            P(f"   repeated times: {len(second)} rows, file rows {second.index.min()}-{second.index.max()}, "
              f"times {ft(second['ts'].min())}->{ft(second['ts'].max())}; "
              f"identical to first copy: {int(same.sum())}/{len(second)}")
            reft = others_med(pd.DataFrame({x: rawc[x]["temp_c"].reindex(idx) for x in sids}), s)
            off = (rawc[s]["temp_c"].reindex(idx) - reft).median()
            e1 = (orig.loc[second["ts"], "temp_c"].to_numpy() - reft.reindex(second["ts"]).to_numpy() - off)
            e2 = (second["temp_c"].to_numpy() - reft.reindex(second["ts"]).to_numpy() - off)
            P(f"   temp error vs other stations at those times: first copy {np.nanmean(np.abs(e1)):.2f}, "
              f"second copy {np.nanmean(np.abs(e2)):.2f} C")
        for i, v in gp.items():
            P(f"   gap {ft(d['ts'][i - 1])} -> {ft(d['ts'][i])} ({v / 60:.1f} h, {int(v / 10) - 1} rows missing)")
    P("clock matches others (0 shift), no repeats/gaps:", ",".join(quiet) if quiet else "none")

    # corrected, de-duplicated series used from here on
    C = {}
    for s in sids:
        d = raw[s].copy()
        for m, k in shifts[s]:
            d.loc[m, "ts"] = d.loc[m, "ts"] + pd.Timedelta(minutes=10 * k)
        C[s] = clean(d)
    fixed = [f"{s}({k * 10:+d}min)" for s in sids for _, k in shifts[s]]
    P("clock corrected for later sections:", ", ".join(fixed) if fixed else "none")

    # ---------- 2. IMPOSSIBLE VALUES AND FROZEN SENSORS ----------
    P("\n=== 2. IMPOSSIBLE VALUES / FROZEN STRETCHES (file order, 3h+ = 18 readings) ===")
    rng = {"temp_c": (-30, 45), "rh_pct": (0, 100), "wind_kmh": (0, 150), "ghi_wm2": (0, 1400), "rain_mm": (-0.01, 3)}
    nlines = 0
    for s in sids:
        d = raw[s]
        for ch in ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "rain_mm", "ghi_wm2"]:
            v = d[ch].to_numpy()
            out = []
            if ch in rng:
                lo, hi = rng[ch]
                for a, b in runs((v < lo) | (v > hi)):
                    out.append(f"RANGE {ft(d.ts[a])}->{ft(d.ts[b])} n={b - a + 1} vals {np.nanmin(v[a:b + 1]):g}..{np.nanmax(v[a:b + 1]):g}")
            if ch in ("temp_c", "rh_pct", "pressure_hpa", "wind_kmh"):
                eq = (v[1:] == v[:-1]) & ~np.isnan(v[1:])
                for a, b in runs(eq):
                    n = b - a + 2
                    val = v[a]
                    if n >= 18 and not (ch == "rh_pct" and 99.5 <= val <= 100):
                        out.append(f"FLAT  {ft(d.ts[a])}->{ft(d.ts[b + 1])} n={n} ({n / 6:.0f}h) val {val:g}")
            for o in out[:4]:
                P(f"{s} {ch}: {o}")
                nlines += 1
            if len(out) > 4:
                P(f"{s} {ch}: ... {len(out) - 4} more")
    if nlines == 0:
        P("none")

    # hourly means, clock-corrected
    H = {ch: pd.DataFrame({s: C[s][ch].resample("1h").mean() for s in sids}) for ch in CH}

    # ---------- 3. SUSTAINED DIFFERENCE FROM OTHER STATIONS ----------
    P("\n=== 3. HOURS A STATION SITS FAR FROM THE OTHER STATIONS (3h+, usual offset removed) ===")
    n3 = 0
    for ch, thr, nm in (("rh_pct", 25, "humidity %"), ("temp_c", 8, "temp C")):
        for s in sids:
            d = H[ch][s] - others_med(H[ch], s)
            d = d - d.median()
            rr = [(a, b) for a, b in runs((d.abs() > thr).to_numpy()) if b - a + 1 >= 3]
            for a, b in rr[:3]:
                P(f"{s} {nm}: {ft(d.index[a])}->{ft(d.index[b])} ({b - a + 1}h) mean diff {d.iloc[a:b + 1].mean():+.1f}")
                n3 += 1
            if len(rr) > 3:
                P(f"{s} {nm}: ... {len(rr) - 3} more windows")
    if n3 == 0:
        P("none")

    # ---------- 4. RAIN ----------
    P("\n=== 4. RAIN: LARGE 10-MIN VALUES AND RUNNING TOTAL VS OTHER STATIONS ===")
    R = pd.DataFrame({s: C[s]["rain_mm"].reindex(idx).fillna(0) for s in sids})
    cum = R.cumsum()
    P("total mm (and vs median of others): " + "  ".join(f"{s} {cum[s].iloc[-1]:.0f}({cum[s].iloc[-1] - others_med(cum, s).iloc[-1]:+.0f})" for s in sids))
    for s in sids:
        mx, tm = R[s].max(), R[s].idxmax()
        if mx > 3:
            P(f"{s}: biggest 10-min value {mx:.1f} mm at {ft(tm)}. End-of-day (own cumulative - others' median), mm:")
            dev = (cum[s] - others_med(cum, s)).resample("1D").last()
            w = dev.loc[tm - pd.Timedelta(days=6): tm + pd.Timedelta(days=2)]
            P("   " + "  ".join(f"{t.strftime('%m-%d')} {v:+.0f}" for t, v in w.items()))

    # ---------- 5. VISITS FROM THE MAINTENANCE LOG ----------
    P("\n=== 5. BEFORE/AFTER EACH LOG VISIT (5 days before vs 5 days after, visit day skipped) ===")
    P("cell = change in offset from others in sd units / slope ratio (1.00 = unchanged). * = notable")
    lp = os.path.join(pack, "maintenance_log.md")
    ev = []
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(S\d+)\s+(.*)", line.strip())
            if m and m.group(2) in sids:
                ev.append((m.group(1), m.group(2)))
    flagged = []
    P("date   stn   " + "".join(f"{n:>13}" for n in ("temp", "humid", "press", "wind", "sun")))
    for D, s in ev:
        t0 = pd.Timestamp(D)
        cells = []
        for ch in CH:
            own, ref = H[ch][s], others_med(H[ch], s)
            sb = slice(t0 - pd.Timedelta(days=5), t0 - pd.Timedelta(hours=1))
            sa = slice(t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=6))
            res = None
            try:
                xb, yb = ref.loc[sb], own.loc[sb]
                xa, ya = ref.loc[sa], own.loc[sa]
                mb, ma = xb.notna() & yb.notna(), xa.notna() & ya.notna()
                if mb.sum() >= 24 and ma.sum() >= 24:
                    db, da = (yb - xb)[mb], (ya - xa)[ma]
                    sd = math.sqrt((db.var() + da.var()) / 2)
                    delta = da.mean() - db.mean()
                    z = delta / sd if sd > 1e-9 else (0.0 if abs(delta) < 1e-9 else 99.0)
                    sb_ = np.polyfit(xb[mb], yb[mb], 1)[0]
                    sa_ = np.polyfit(xa[ma], ya[ma], 1)[0]
                    ratio = sa_ / sb_ if sb_ else np.nan
                    res = (z, ratio)
            except Exception:
                res = None
            if res is None:
                cells.append(f"{'n/a':>13}")
            else:
                z, ratio = res
                star = "*" if (abs(z) > 1 or abs(ratio - 1) > 0.12) else " "
                cells.append(f"{z:+6.1f}/{ratio:4.2f}{star}")
                if star == "*":
                    flagged.append((abs(z) + 5 * abs(ratio - 1), s, ch, t0))
        P(f"{D[5:]}  {s}  " + "".join(cells))
    flagged.sort(reverse=True)
    if flagged:
        P("best single step in the offset from others (timing of the change), top cells:")
        for _, s, ch, t0 in flagged[:6]:
            d = (H[ch][s] - others_med(H[ch], s)).loc[t0 - pd.Timedelta(days=5): t0 + pd.Timedelta(days=6)].dropna()
            v = d.to_numpy()
            n = len(v)
            if n < 30:
                continue
            cs = np.cumsum(v)
            best = (0, 0, 0.0)
            for i in range(6, n - 6):
                m1, m2 = cs[i - 1] / i, (cs[-1] - cs[i - 1]) / (n - i)
                sc = abs(m2 - m1) * math.sqrt(i * (n - i) / n)
                if sc > best[0]:
                    best = (sc, i, m2 - m1)
            P(f"   {s} {ch}: step {best[2]:+.2f} at {ft(d.index[best[1]])} (visit {t0.strftime('%m-%d')})")

    # ---------- 6. GPS POSITION ----------
    P("\n=== 6. GPS POSITION vs stations.csv (only stations more than 1 km off) ===")
    off = 0
    if meta is not None:
        for s in sids:
            p = os.path.join(pack, "gps", s + ".nmea")
            if not os.path.exists(p):
                continue
            la, lo = parse_gps(p)
            if not la:
                continue
            row = meta[meta["station_id"].astype(str).str.strip() == s].iloc[0]
            mla, mlo = float(np.median(la)), float(np.median(lo))
            dn = (mla - row["lat"]) * 111.19
            de = (mlo - row["lon"]) * 111.19 * math.cos(math.radians(mla))
            dist = math.hypot(dn, de)
            if dist > 1:
                off += 1
                P(f"{s}: GPS {mla:.4f},{mlo:.4f} vs sheet {row['lat']},{row['lon']} -> {dist:.1f} km "
                  f"({dn:+.1f} km north, {de:+.1f} km east)")
    if off == 0:
        P("none")

    with open("explore2_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()