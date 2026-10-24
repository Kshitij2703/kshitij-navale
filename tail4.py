import sys, pandas as pd
pack = sys.argv[1]
ids = [f"S0{i}" for i in range(1, 9)]
d = {s: pd.read_csv(f"{pack}/data/{s}.csv", parse_dates=["timestamp_utc"]).drop_duplicates("timestamp_utc").set_index("timestamp_utc") for s in ids}
h = lambda ch: pd.DataFrame({s: d[s][ch].resample("1h").mean() for s in ids})
T, W, G = h("temp_c"), h("wind_kmh"), h("ghi_wm2")
o = lambda df: df.drop(columns="S04").median(axis=1)
dT = T["S04"] - o(T)
t = pd.DataFrame({"S04_T": T.S04, "oth_T": o(T), "diff": dT, "S04_wind": W.S04, "oth_wind": o(W), "S04_sun": G.S04, "oth_sun": o(G)})
print(t.iloc[-72::3].round(1).to_string())
sun = o(G) > 150
wk = dT.groupby([dT.index.to_period("W"), sun]).mean().unstack().round(2)
wk.columns = ["night", "day"]
print(wk.to_string())