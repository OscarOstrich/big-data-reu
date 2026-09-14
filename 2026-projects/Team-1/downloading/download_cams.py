"""
Downloads one year of CAMS global atmospheric composition analysis data,
one file per day, both surface-level and atmospheric (pressure-level)
variables, at 00 UTC and 12 UTC.
 
Prerequisites:
  - source setup_env.sh (activates the conda environment and sets $CDSAPI_RC)
  - a valid .cdsapirc file with your Copernicus ADS credentials
 
Output: two files per day in DOWNLOAD_PATH,
  {date}-cams-surface-level.nc
  {date}-cams-atmospheric.nc
 
Safe to re-run: already-downloaded days are skipped automatically.
"""
 
import zipfile
from pathlib import Path
from datetime import date, timedelta
 
import cdsapi
 
# Set your date range here
START_DATE = date(2024, 7, 24)
END_DATE   = date(2025, 7, 23)
 
# Update this path if running from a different account
DOWNLOAD_PATH = Path("/home/mtogbe1/reu2026_team1/research/data/cams")
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
 
c = cdsapi.Client()  # reads credentials from $CDSAPI_RC
 
SURFACE_VARS = [
    # Meteorological surface-level variables:
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    # Pollution surface-level variables:
    "particulate_matter_1um",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_ozone",
    "total_column_sulphur_dioxide",
]
 
ATMOS_VARS = [
    # Meteorological atmospheric variables:
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "specific_humidity",
    # Pollution atmospheric variables:
    "carbon_monoxide",
    "nitrogen_dioxide",
    "nitrogen_monoxide",
    "ozone",
    "sulphur_dioxide",
]
 
PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300", "400",
    "500", "600", "700", "850", "925", "1000",
]
 
 
def download_day(day: date):
    date_str = day.strftime("%Y-%m-%d")
    zip_path = DOWNLOAD_PATH / f"{date_str}-cams.nc.zip"
    surf_path = DOWNLOAD_PATH / f"{date_str}-cams-surface-level.nc"
    atmos_path = DOWNLOAD_PATH / f"{date_str}-cams-atmospheric.nc"
 
    if surf_path.exists() and atmos_path.exists():
        print(f"{date_str}  ->  already downloaded, skipping")
        return
 
    if not zip_path.exists():
        c.retrieve(
            "cams-global-atmospheric-composition-forecasts",
            {
                "type": "forecast",
                "leadtime_hour": "0",   # analysis time step, not a forecast
                "variable": SURFACE_VARS + ATMOS_VARS,
                "pressure_level": PRESSURE_LEVELS,
                "date": date_str,
                "time": ["00:00", "12:00"],
                "format": "netcdf_zip",
            },
            str(zip_path),
        )
        print(f"{date_str}  downloaded")
 
    with zipfile.ZipFile(zip_path, "r") as zf:
        if not surf_path.exists():
            with open(surf_path, "wb") as f:
                f.write(zf.read("data_sfc.nc"))
        if not atmos_path.exists():
            with open(atmos_path, "wb") as f:
                f.write(zf.read("data_plev.nc"))
 
    print(f"{date_str}  unpacked")
    zip_path.unlink()
 
 
if __name__ == "__main__":
    current = START_DATE
    while current <= END_DATE:
        try:
            download_day(current)
        except Exception as e:
            print(f"{current}  FAILED: {e}")
        current += timedelta(days=1)