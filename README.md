Kshitij Navale
kshitijnavale2703@gmail.com
Video: <Will upload later>
Run: python check.py <path to data_pack>

# Ridgeline weather network: data check

`check.py` reads the data pack (stations.csv, data/, gps/, maintenance_log.md), looks for faulty readings, and writes a list of them to `flags.csv`. The report for the coordinator is `report.html`.

## 1. How to run

- Tested on Python 3.13.
- Install: `pip install -r requirements.txt` (pandas, numpy).
- Run: `python check.py <path to data_pack>`
- It writes `flags.csv` into the current folder and prints a short summary. It only reads the data pack and never changes the input files. The data pack is not in this repo.
- `flags.csv` columns: `station_id, channel, start_utc, end_utc, issue, action, note`. `action` is `exclude` (leave out), `correct` (fix, the note says how) or `keep` (looks odd but is real).

## 2. What each check in check.py does

1. **Logger clock.** Compares each station's sunlight curve with the middle value of the other stations, shifted by up to 12 hours. If it only lines up after a shift, the clock is off and the rows are marked `correct`. The GPS clock difference is quoted as a second opinion.
2. **Repeated times and gaps.** Identical repeated rows (a copied day) are marked `exclude`. So are repeated times with different values, and any gap longer than 1 hour.
3. **Impossible values.** Error codes (such as -40 or 85 for temperature, -999, 9999), values outside the possible range, and single readings that jump far from the other stations are marked `exclude`.
4. **Frozen sensor.** The same value for too long (about 3 hours, 4 hours for pressure) while the others keep changing is marked `exclude`. Zero wind, zero sunlight and humidity at 100 % are ignored.
5. **Rain gauge.** No rain for days while the other stations had some, followed by a large dump in a few readings, means the gauge was blocked and then released. That period is marked `exclude`. Any single reading of 3 mm or more in 10 minutes is also marked `exclude`.
6. **Far from the other stations.** A station that stays far from the others for many hours (humidity, temperature, pressure) is marked `exclude`. The comparison removes each station's usual offset and its usual time-of-day pattern, and pressure is first reduced to a common height. Humidity above 100 % outside these stretches is marked `correct` (clip to 100).
7. **Lasting steps and drifts.** Daily differences from the other stations are fitted with a step and with a slow ramp. A step in temperature, humidity or pressure is marked `correct` with the amount to subtract. A wind-speed step that matches a unit change (for example m/s read as km/h) is marked `correct` with the factor. A growing bias is marked `exclude`. Daytime temperature is tested separately from night, which catches a sensor that is only wrong in sunshine. When one station jumps, the others can show a mirror image; those are dropped.
8. **GPS position.** The GPS position and height are compared with `stations.csv`. A difference is flagged as `metadata`.
9. **Odd but real.** Offsets explained by height (temperature falls about 6.5 C per km, pressure with the usual formula) and warm-dry or cold-moist spells where temperature and humidity move opposite ways are marked `keep`, so nobody "fixes" real weather.
10. **Tidy up.** Flags inside a larger `exclude` stretch of the same station and channel are removed, neighbouring flags are merged, and the nearest maintenance log entry (within 2 days) is added to the note.

## 3. Other files

- `explore.py`, `explore2.py`, `explore3.py`: the first looks at the data (file checks, per-reading statistics, GPS, station-vs-others tables, clock and gap checks, before/after each maintenance visit). They print results only. The faults check.py looks for were found with these.
- `tail4.py`: checks the last 72 hours and weekly day/night temperature offsets of S04, which showed the daytime-only warm bias.
- `report.html`: the one-file report for the coordinator (what data to use, and where the technician goes).
- `DECISIONS.md`: my own written answers to the five decision questions.
- `ai_logs/`: logs of the AI conversations used for this work.

## 4. Assumptions

- Timestamps in the data files are UTC.
- Times written in the maintenance log are local time (UTC-6). For example "around 10" in the log is 16:00 UTC.
- Comparing stations needs at least 4 stations. With fewer, the comparison checks are skipped.
- If the GPS position and the install sheet (`stations.csv`) disagree, the sheet is assumed to be the typo.
- The thresholds were chosen by hand from this one data pack and a few made-up test packs. They are not tuned for other networks.

## 5. Known problems in flags.csv (found, not fixed)

- **S06 clock row shows the wrong times.** Its start and end are the logger's own (wrong) times, 24 Mar 10:10 to 11 Apr 17:50. In true UTC, add 6 hours: 24 Mar 16:10 to 11 Apr 23:50.
- **S02 pressure amount is too large.** The note says to subtract 8.0 hPa. That number is measured after pressure is reduced to a common height. In the station's own pressure the jump is about 6.5 hPa (8.0 x exp(-1623/8000)), and 6.5 is the amount to subtract. `report.html` uses 6.5.
- **One S05 humidity reading is marked "clip to 100".** The 11 Mar 00:00 reading (104 %) is the last point of the same faulty stretch (7 Mar 12:00 to 10 Mar 23:50) and should be `exclude`. `report.html` treats the stretch as ending at 11 Mar 00:00.
- **Start times can be up to an hour off.** The starts of lasting steps and drifts (S02 pressure, S04 temperature, S07 wind) are found from hourly and daily data, so the exact start may differ by up to an hour for steps. The S04 temperature start is a rough date, not a measured moment.