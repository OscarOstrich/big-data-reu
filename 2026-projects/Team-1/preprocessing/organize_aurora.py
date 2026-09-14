"""
Organizes Aurora forecast output files into standardized, per-horizon
folders for matching against TEMPO.
 
Before running, set SOURCE_DIR and DEST_DIR below.
 
SOURCE_DIR must contain three subfolders, already populated with Aurora's
forecast output (tcaurora_YYYY_MM_DD_00.nc files):
    OneDayPred/
    TwoDayPred/
    ThreeDayPred/
 
DEST_DIR does not need to exist yet, this script creates three matching
subfolders inside it automatically:
    OneDay/
    TwoDay/
    ThreeDay/
 
Each file gets renamed to a standardized format (AURORA_YYYYMMDD_00Z.nc)
as it's copied over. Organizes all three horizons in a single run.
 
Requires Python 3.10+.
"""
 
import shutil
from pathlib import Path
from datetime import date, timedelta
 
# Set these two paths before running 
SOURCE_DIR = Path("/path/to/your/aurora/forecasts")   # must contain OneDayPred/TwoDayPred/ThreeDayPred
DEST_DIR   = Path("/path/to/organized/output")         # OneDay/TwoDay/ThreeDay created here automatically

 
LEAD_DAYS_BY_HORIZON = {"OneDay": 1, "TwoDay": 2, "ThreeDay": 3}
 
# Base range matches the raw CAMS download (July 24, 2024 - July 23, 2025).
# Each horizon's actual range is this shifted forward by its lead time.
BASE_START = date(2024, 7, 24)
BASE_END   = date(2025, 7, 23)
 
 
def find_aurora_00z(day: date, source_dir: Path) -> Path | None:
    """
    Look for the 00Z Aurora forecast file for a given valid date.
    Expected filename: tcaurora_YYYY_MM_DD_00.nc
    """
    expected_name = f"tcaurora_{day.strftime('%Y_%m_%d')}_00.nc"
    candidate = source_dir / expected_name
 
    if candidate.exists():
        return candidate
 
    print(f"{day}  ->  no 00Z Aurora file found (expected {expected_name})")
    return None
 
 
def organize_horizon(horizon_label: str):
    """Organizes one forecast horizon: OneDay, TwoDay, or ThreeDay."""
    lead_days = LEAD_DAYS_BY_HORIZON[horizon_label]
    source_dir = SOURCE_DIR / f"{horizon_label}Pred"
    dest_dir = DEST_DIR / horizon_label
 
    if not source_dir.exists():
        print(f"\nSKIPPING {horizon_label}: source directory not found at {source_dir}")
        print(f"Make sure SOURCE_DIR is set correctly and contains a '{horizon_label}Pred' folder.")
        return
 
    dest_dir.mkdir(parents=True, exist_ok=True)
 
    start_date = BASE_START + timedelta(days=lead_days)
    end_date = BASE_END + timedelta(days=lead_days)
 
    print(f"\n{'=' * 60}")
    print(f"Organizing {horizon_label} horizon ({start_date} to {end_date})")
    print(f"{'=' * 60}")
 
    current = start_date
    days_total, days_copied, days_missing = 0, 0, 0
    missing_days = []
 
    while current <= end_date:
        days_total += 1
 
        aurora_file = find_aurora_00z(current, source_dir)
 
        if aurora_file is None:
            missing_days.append(str(current))
            days_missing += 1
            current += timedelta(days=1)
            continue
 
        # naming convention for organized 00Z Aurora forecast files
        dest_name = f"AURORA_{current.strftime('%Y%m%d')}_00Z.nc"
        dest_path = dest_dir / dest_name
 
        shutil.copy2(aurora_file, dest_path)
        print(f"{current}  ->  {dest_name}")
        days_copied += 1
 
        current += timedelta(days=1)
 
    print("-" * 60)
    print(f"{horizon_label} done.  {days_copied}/{days_total} days copied,  {days_missing} missing.")
    if missing_days:
        print(f"Days with no 00Z Aurora file for {horizon_label}:")
        for d in missing_days:
            print(f"  {d}")
 
 
if __name__ == "__main__":
    for horizon_label in LEAD_DAYS_BY_HORIZON:
        organize_horizon(horizon_label)