#!/usr/bin/env python3
"""explore.py - first look at the weather-station data pack.
Usage: python explore.py <pack_folder>
Prints a short report and also saves it to explore_output.txt (in the folder you run it from).
Needs: pandas, numpy.
"""
import sys, os, glob, math, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

CH = ["temp_c", "rh_pct", "pressure_hpa", "wind_kmh", "rain_mm", "ghi_wm2"]
ZERO_OK = {"rain_mm", "ghi_wm2"}  # zeros are normal here, ignored for "stuck" check


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def p(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")


out = Tee("explore_output.txt")
P = out.p


def fmt_t(t):
    return "NaT" if pd.isna(t) else t.strftime("%m-%d %H:%M")


def load_station(pack, sid):
    path = os.path.join(pack, "data", sid + ".csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["ts"] = pd.to_datetime(df["timestamp_utc"].str.strip(), errors="coerce")
    for c in df.columns:
        if c not in ("timestamp_utc", "ts"):
            df[c] = pd.to_numeric(df[c].str.strip().replace("", np.nan), errors="coerce")
    return df


def main():
    if len(sys.argv) < 2:
        print("usage: python explore.py <pack_folder>")
        sys.exit(1)
    pack = sys.argv[1]
    st_path = os.path.join(pack, "stations.csv")
    meta = pd.read_csv(st_path) if os.path.exists(st_path) else None
    if meta is not None:
        ids = list(meta["station_id"].astype(str).str.strip())
    else:
        ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(pack, "data", "*.csv")))
    dfs = {s: load_station(pack, s) for s in ids}
    dfs = {s: d for s, d in dfs.items() if d is not None}

    # ---------------- 1. FILE CHECK ----------------
    P("=== 1. FILE CHECK (rows, time range, repeats, wrong steps, gaps >1h) ===")
    for s, d in dfs.items():
        ts = d["ts"]
        bad_ts = int(ts.isna().sum())
        dup = ts[ts.duplicated(keep=False) & ts.notna()]
        step = ts.diff().dt.total_seconds() / 60
        odd = step[(step != 10) & step.notna()]
        back = odd[odd < 0]
        gaps = step[step > 60]
        P(f"{s}: rows={len(d)} first={ts.min()} last={ts.max()} unparsable_ts={bad_ts} "
          f"dup_rows={len(dup)} (distinct {dup.nunique()}) odd_steps={len(odd)} backwards={len(back)} gaps>1h={len(gaps)}")
        if len(dup):
            P("   dup times e.g.:", ", ".join(fmt_t(t) for t in sorted(dup.unique())[:4]))
        if len(odd):
            ex = [f"{fmt_t(ts[i])}({odd[i]:+.0f}min)" for i in odd.index[:4]]
            P("   odd steps e.g. (time reached, step):", ", ".join(ex))
        if len(gaps):
            ex = [f"{fmt_t(ts[i - 1])}->{fmt_t(ts[i])}({gaps[i] / 60:.1f}h)" for i in gaps.index[:4]]
            P("   gaps:", ", ".join(ex))

    # ---------------- 2. PER-READING CHECK ----------------
    P("\n=== 2. READINGS (min / max / mean / empty-or-NaN count / longest exactly-flat stretch) ===")
    for c in CH:
        P(f"-- {c}" + ("  (zeros ignored for flat check)" if c in ZERO_OK else ""))
        for s, d in dfs.items():
            if c not in d:
                P(f"{s}: column missing")
                continue
            v = d[c]
            valid = v.notna()
            if c in ZERO_OK:
                valid = valid & (v != 0)
            same = v.eq(v.shift()) & valid & valid.shift(fill_value=False)
            gid = (~same).cumsum()
            sizes = valid.groupby(gid).sum()
            best = sizes.idxmax()
            n = int(sizes.max())
            first = d.index[gid == best][0]
            flat = f"{n} readings ({n * 10 / 60:.1f}h) from {fmt_t(d['ts'][first])} (val {v[first]:g})" if n > 1 else "none"
            P(f"{s}: min={v.min():.1f} max={v.max():.1f} mean={v.mean():.2f} nan={int(v.isna().sum())} flat={flat}")

    # ---------------- 3. GPS CHECK ----------------
    P("\n=== 3. GPS (logger clock minus GPS clock; position vs stations.csv) ===")
    for s in dfs:
        path = os.path.join(pack, "gps", s + ".nmea")
        if not os.path.exists(path):
            P(f"{s}: no gps file")
            continue
        diffs, lat, lon, alt = [], [], [], []
        n_bad = n_void = n_rmc = n_gga = 0
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                lts, sent = line.split(",", 1)
                lts = pd.to_datetime(lts.strip())
                body, cs = sent.strip().lstrip("$").rsplit("*", 1)
                x = 0
                for ch in body:
                    x ^= ord(ch)
                if x != int(cs.strip()[:2], 16):
                    n_bad += 1
                    continue
                f = body.split(",")
                if f[0].endswith("RMC"):
                    if f[2] != "A" or not f[3]:
                        n_void += 1
                        continue
                    g = datetime.strptime(f[9] + f[1][:6], "%d%m%y%H%M%S") + timedelta(seconds=float(f[1][6:] or 0))
                    diffs.append((lts, (lts - g).total_seconds()))
                    la = (int(f[3][:2]) + float(f[3][2:]) / 60) * (1 if f[4] == "N" else -1)
                    lo = (int(f[5][:3]) + float(f[5][3:]) / 60) * (1 if f[6] == "E" else -1)
                    lat.append(la); lon.append(lo); n_rmc += 1
                elif f[0].endswith("GGA"):
                    if f[6] in ("", "0") or not f[2]:
                        n_void += 1
                        continue
                    alt.append(float(f[9])); n_gga += 1
            except Exception:
                n_bad += 1
        msg = f"{s}: used RMC={n_rmc} GGA={n_gga} no-fix={n_void} broken={n_bad}"
        if diffs:
            dv = np.array([x[1] for x in diffs])
            msg += f" | clock diff median={np.median(dv):.0f}s (min {dv.min():.0f}, max {dv.max():.0f})"
            P(msg)
            jumps = [(diffs[i][0], dv[i - 1], dv[i]) for i in range(1, len(dv)) if abs(dv[i] - dv[i - 1]) > 60]
            for t, a, b in jumps[:6]:
                P(f"   JUMP at logger {fmt_t(t)}: {a:.0f}s -> {b:.0f}s")
            if len(jumps) > 6:
                P(f"   ... {len(jumps)} jumps total")
        else:
            P(msg + " | no usable RMC")
        if lat and meta is not None:
            row = meta[meta["station_id"].astype(str).str.strip() == s]
            if len(row):
                row = row.iloc[0]
                mla, mlo = np.median(lat), np.median(lon)
                R = 6371.0

                def hav(a1, o1, a2, o2):
                    a1, o1, a2, o2 = map(math.radians, (a1, o1, a2, o2))
                    h = math.sin((a2 - a1) / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin((o2 - o1) / 2) ** 2
                    return 2 * R * math.asin(math.sqrt(h))
                dist = hav(mla, mlo, row["lat"], row["lon"])
                spread = max(hav(mla, mlo, a, o) for a, o in zip(lat, lon))
                ad = f"{np.median(alt):.0f}m vs {row['elevation_m']}m (diff {np.median(alt) - row['elevation_m']:+.0f}m)" if alt else "no altitude"
                P(f"   position: GPS-vs-stations.csv distance={dist:.2f} km (GPS scatter max {spread * 1000:.0f} m) | altitude {ad}")

    # ---------------- 4. STATION vs OTHERS ----------------
    P("\n=== 4. DAILY: station minus median of the other stations ===")
    P("(blank '.' = no data that day. A constant offset for a station is normal (altitude/location); look for CHANGES over days.)")
    daily = {}
    for s, d in dfs.items():
        dd = d.dropna(subset=["ts"]).copy()
        dd["day"] = dd["ts"].dt.date
        g = dd.groupby("day")
        daily[s] = pd.DataFrame({
            "tmax": g["temp_c"].max(), "rh": g["rh_pct"].mean(), "p": g["pressure_hpa"].mean(),
            "wind": g["wind_kmh"].mean(), "rain": g["rain_mm"].sum(min_count=1)})
    names = {"tmax": "highest temp (C)", "rh": "mean humidity (%)", "p": "mean pressure (hPa)",
             "wind": "mean wind (km/h)", "rain": "total rain (mm)"}
    sids = list(daily)
    full = open("explore_daily_full.txt", "w")
    for m, title in names.items():
        t = pd.DataFrame({s: daily[s][m] for s in sids}).sort_index()
        dev = pd.DataFrame({s: t[s] - t.drop(columns=s).median(axis=1) for s in sids})
        base = dev.median()                       # each station's usual offset
        shift = dev - base                        # change relative to its own usual offset
        mad = (shift.abs().median() * 1.4826).clip(lower=1e-6)
        thr = np.maximum(4 * mad, {"tmax": 2.0, "rh": 10.0, "p": 2.0, "wind": 3.0, "rain": 3.0}[m])
        unusual = shift.abs() > thr
        full.write(f"-- {title}: deviation from median of others\n" + dev.round(1).to_string() + "\n\n")
        P(f"-- {title}  | row 'usual' = each station's normal offset; day rows only shown if a station is unusual (marked *)")
        P("day   " + "".join(f"{s:>8}" for s in sids))
        P("usual " + "".join(f"{v:8.1f}" for v in base))
        shown = 0
        for day in dev.index:
            if unusual.loc[day].any():
                P(f"{day.strftime('%m-%d')} " + "".join(
                    (f"{dev.loc[day, s]:7.1f}" + ("*" if unusual.loc[day, s] else " ")) if pd.notna(dev.loc[day, s]) else "       ."
                    for s in sids))
                shown += 1
        if not shown:
            P("(no unusual days)")
    full.close()
    P("\n(full numeric tables for all days saved in explore_daily_full.txt)")


if __name__ == "__main__":
    main()