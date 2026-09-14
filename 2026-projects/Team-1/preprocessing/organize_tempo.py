"""
For each day in the date range, selects the single best TEMPO NO2 scan
(closest to 00 UTC, within a 3-hour window, meeting a 70% valid-pixel
coverage threshold over the western US) and copies it into a standardized
daily file (TEMPO_YYYYMMDD_00Z.nc4) for matching against Aurora forecasts.
 
If no scan in the window meets the coverage threshold, that day is
skipped entirely rather than falling back to a lower quality scan.
"""
 
import re
import shutil
import xarray as xr
from pathlib import Path
from datetime import date, timedelta, datetime
 
# Paths: requires TEMPO data already downloaded to TEMPO_NO2_SOURCE
TEMPO_NO2_SOURCE = Path("/umbc/rs/cybertrn/reu2026/team1/research/data/tempo_l3/ECF")  # where raw TEMPO data lives
TEMPO_DEST       = Path("/umbc/rs/cybertrn/reu2026/team1/research/testing/TEMPO_data/00_utc")  # where organized files go
 
# Settings
TARGET_HOUR_UTC = 0.0
START_DATE      = date(2024, 7, 24)
END_DATE        = date(2025, 7, 27)
 
# West coast bounding box used ONLY for coverage check, not for slicing output
LAT_MIN = 31.13
LAT_MAX = 49.51
LON_MIN = -125.85
LON_MAX = -89.06
 
# Data coverage threshold (fraction of non-NaN pixels), NOT a cloud fraction
COVERAGE_THRESHOLD = 0.7
 
# a TEMPO scan more than this many hours from TARGET_HOUR_UTC will be counted as a missing day
MAX_DELTA_HOURS = 3.0  # 180 minutes
 
TIMESTAMP_RE = re.compile(r'_(\d{8}T\d{6}Z)_S\d+_')
 
 
def parse_timestamp(filename: str) -> datetime | None:
    match = TIMESTAMP_RE.search(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
 
 
def check_coverage(filepath: Path) -> float:
    """
    Check what fraction of pixels in the California bounding box are non-NaN.
    Used only for scan selection — does NOT affect the output file contents.
    """
    try:
        ds_product = xr.open_dataset(filepath, engine="netcdf4", group="product")
        if "vertical_column_troposphere" not in ds_product:
            ds_product.close()
            return 0.0
 
        ds_root = xr.open_dataset(filepath, engine="netcdf4")
 
        da = xr.DataArray(
            ds_product["vertical_column_troposphere"].values,
            dims=("time", "latitude", "longitude"),
            coords={
                "time":      ds_root.coords["time"],
                "latitude":  ds_root.coords["latitude"],
                "longitude": ds_root.coords["longitude"],
            },
        )
 
        subset = da.sel(
            latitude=slice(LAT_MIN, LAT_MAX),
            longitude=slice(LON_MIN, LON_MAX),
        )
 
        total = subset.size
        ds_product.close()
        ds_root.close()
 
        if total == 0:
            return 0.0
 
        valid = float(subset.notnull().sum().values)
        return valid / total
 
    except Exception as e:
        print(f"Coverage check failed for {filepath.name}: {e}")
        return 0.0
 
 
def find_best_no2_scan(day_dir: Path) -> tuple[Path | None, datetime | None]:
    """
    Find the scan closest to 00 UTC that meets the coverage threshold,
    restricted to scans within MAX_DELTA_HOURS of the target hour.
    If nothing in the window meets the coverage threshold, the day is
    treated as missing — no fallback to a below-threshold scan.
    """
    candidates = []
 
    for f in sorted(day_dir.glob("*.nc4")):
        dt = parse_timestamp(f.name)
        if dt is None:
            print(f"  Could not parse timestamp: {f.name}")
            continue
        hour = dt.hour + dt.minute / 60 + dt.second / 3600
        delta = min(abs(hour - TARGET_HOUR_UTC), 24 - abs(hour - TARGET_HOUR_UTC))
        candidates.append((delta, f, dt))
 
    if not candidates:
        return None, None
 
    candidates.sort(key=lambda x: x[0])
 
    # Hard cutoff — never consider a scan more than MAX_DELTA_HOURS away.
    within_window = [c for c in candidates if c[0] <= MAX_DELTA_HOURS]
 
    if not within_window:
        print(f"  No scans within {MAX_DELTA_HOURS:.1f} hrs of {TARGET_HOUR_UTC} UTC")
        return None, None
 
    for delta, f, dt in within_window:
        coverage = check_coverage(f)
        if coverage >= COVERAGE_THRESHOLD:
            print(f"  Selected {f.name} (delta {delta:.2f} hrs, coverage {coverage:.1%})")
            return f, dt
        else:
            print(f"  Skipped {f.name} (delta {delta:.2f} hrs, coverage {coverage:.1%} below threshold)")
 
    # No candidate in the window met the coverage threshold so the day is treated as missing
    print(f"  No in-window scan met coverage threshold, day treated as missing")
    return None, None
 
 
def main():
    TEMPO_DEST.mkdir(parents=True, exist_ok=True)
 
    current = START_DATE
    days_total, days_copied, days_missing = 0, 0, 0
    missing_days = []
 
    while current <= END_DATE:
        days_total += 1
        no2_day_dir = TEMPO_NO2_SOURCE / current.strftime("%Y/%m/%d")
 
        dest_name = f"TEMPO_{current.strftime('%Y%m%d')}_00Z.nc4"
        dest_path = TEMPO_DEST / dest_name
 
        # Skip if already done
        if dest_path.exists():
            print(f"{current}  ->  already exists, skipping")
            days_copied += 1
            current += timedelta(days=1)
            continue
 
        if not no2_day_dir.exists():
            print(f"No NO2 directory for {current}")
            missing_days.append(f"{current} (no NO2 dir)")
            days_missing += 1
            current += timedelta(days=1)
            continue
 
        no2_file, no2_dt = find_best_no2_scan(no2_day_dir)
        if no2_file is None:
            print(f"No valid NO2 scans for {current}")
            missing_days.append(f"{current} (no valid scan)")
            days_missing += 1
            current += timedelta(days=1)
            continue
 
        # Copy the raw NO2 file as-is — it already has support_data/eff_cloud_fraction
        # baked in from Ray's ECF source, so no merge step is needed anymore.
        shutil.copy2(no2_file, dest_path)
        print(f"{current}  NO2 copied: {no2_file.name}  ->  {dest_name}")
 
        days_copied += 1
        current += timedelta(days=1)
 
    print("=" * 60)
    print(f"Done.  {days_copied}/{days_total} days copied,  {days_missing} missing.")
    if missing_days:
        print("Days with issues:")
        for d in missing_days:
            print(f"  {d}")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()