import os
import glob
import pandas as pd
import numpy as np
import xarray as xr

AURORA_BASE_DIR = "/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/AURORA_data/00_utc"
CAMS_DIR = "/umbc/rs/cybertrn/reu2026/team1/research/data/cams"
TEMPO_DIR = "/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/TEMPO_data/00_utc"

OUTPUT_DIR = "/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series"
SKIPPED_CSV = os.path.join(OUTPUT_DIR, "skipped_aurora_multilead.csv")

# Subfolders under AURORA_BASE_DIR, each containing files at a fixed
# rollout_step (lead time) from a shared initialization cycle. The
# rollout_step is re-read from each file directly rather than trusted from
# the folder name, since that's the authoritative source.
LEAD_FOLDERS = ["OneDay", "TwoDay", "ThreeDay"]

# rollout_step -> output CSV path, one file per lead time.
OUTPUT_CSV_BY_ROLLOUT_STEP = {
    1: os.path.join(OUTPUT_DIR, "residual_table_t+1_no2.csv"),
    3: os.path.join(OUTPUT_DIR, "residual_table_t+3_no2.csv"),
    5: os.path.join(OUTPUT_DIR, "residual_table_t+5_no2.csv"),
}

# Columns computed/carried during processing but not written to the final CSV
COLUMNS_TO_DROP_ON_SAVE = ["rollout_step", "target_time", "init_datetime"]

MIN_LAT = 31.31
MAX_LAT = 49.01
MIN_LON = -125.45
MAX_LON = -101.98

AVOGADRO = 6.02214076e23
NO2_MOLAR_MASS_KG_PER_MOL = 0.0460055
M2_TO_CM2 = 10000.0

ECF_THRESHOLD = 0.2

CAMS_SURFACE_VARS = [
    "u10",
    "v10",
    "t2m",
    "msl",
    "pm1",
    "pm2p5",
    "pm10",
    "tcco",
    "tc_no",
    "tcno2",
    "gtco3",
    "tcso2",
]


def cams_surface_path_for_date(date_):
    # Finds path to CAMS surface-level variables file for the given calendar date
    datestr = pd.to_datetime(date_).strftime("%Y-%m-%d")
    return os.path.join(CAMS_DIR, f"{datestr}-cams-surface-level.nc")


def tempo_path_for_target_date(target_date):
    # Finds path to the 00Z TEMPO file for target date
    datestr = pd.to_datetime(target_date).strftime("%Y%m%d")
    return os.path.join(TEMPO_DIR, f"TEMPO_{datestr}_00Z.nc4")


def convert_no2_kgm2_to_molec_cm2(da):
    return da * AVOGADRO / NO2_MOLAR_MASS_KG_PER_MOL / M2_TO_CM2


def convert_lon_to_minus180_180(da):
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.sortby("latitude")
    da = da.sortby("longitude")
    return da


def crop_360(ds):
    # crops mask to bounding box within 360 notation of lat/lon
    min_lon_360 = MIN_LON % 360
    max_lon_360 = MAX_LON % 360

    lat_mask = (ds.latitude >= MIN_LAT) & (ds.latitude <= MAX_LAT)
    lon_mask = (ds.longitude >= min_lon_360) & (ds.longitude <= max_lon_360)

    return ds.isel(latitude=lat_mask, longitude=lon_mask)


def squeeze_aurora_surface(da):
    for dim in ["batch", "history", "time"]:
        if dim in da.dims:
            da = da.isel({dim: 0})
    return da


def squeeze_cams_surface(da, forecast_reference_time_index):
    if "forecast_period" in da.dims:
        da = da.isel(forecast_period=0)
    if "forecast_reference_time" in da.dims:
        da = da.isel(forecast_reference_time=forecast_reference_time_index)
    if "time" in da.dims:
        da = da.isel(time=0)
    return da


def preprocess_func_NO2_combined(ds):
    ds_product = xr.open_dataset(ds.encoding['source'], engine='netcdf4', group='product')
    ds_support = xr.open_dataset(ds.encoding['source'], engine='netcdf4', group='support_data')
    ds_coords = xr.open_dataset(ds.encoding['source'], engine='netcdf4')

    ds_formatted = xr.Dataset(
        data_vars={
            'vertical_column_troposphere': (('time', 'latitude', 'longitude'), (ds_product['vertical_column_troposphere'].values.reshape(1, 2950, 7750))),
            'vertical_column_stratosphere': (('time', 'latitude', 'longitude'), (ds_product['vertical_column_stratosphere'].values.reshape(1, 2950, 7750))),
            'total_column_NO2': (('time', 'latitude', 'longitude'), (ds_product['vertical_column_stratosphere'].values.reshape(1, 2950, 7750) + ds_product['vertical_column_troposphere'].values.reshape(1, 2950, 7750))),
            'ECF': (('time', 'latitude', 'longitude'), (ds_support['eff_cloud_fraction'].values.reshape(1, 2950, 7750))),
            'quality': (('time', 'latitude', 'longitude'), (ds_product['main_data_quality_flag'].values.reshape(1, 2950, 7750)))
        },
        coords={
            'time': ds_coords.coords['time'],
            'latitude': ds_coords.coords['latitude'],
            'longitude': ds_coords.coords['longitude'],
        }
    )

    return ds_formatted


def apply_filter(ds_base, ecf_threshold=ECF_THRESHOLD):
    ds_filtered = ds_base.assign(
        vertical_column_troposphere=xr.where((ds_base['quality'] > 0) | (ds_base['ECF'] > ecf_threshold), np.nan, ds_base['vertical_column_troposphere']),
        vertical_column_stratosphere=xr.where((ds_base['quality'] > 0) | (ds_base['ECF'] > ecf_threshold), np.nan, ds_base['vertical_column_stratosphere']),
        total_column_NO2=xr.where((ds_base['quality'] > 0) | (ds_base['ECF'] > ecf_threshold), np.nan, ds_base['total_column_NO2']))

    return ds_filtered


def open_tempo_on_aurora_grid(tempo_path, aurora_grid):
    ds_root = xr.open_dataset(tempo_path, engine="netcdf4")
    ds_formatted = preprocess_func_NO2_combined(ds_root)
    ds_filtered = apply_filter(ds_formatted)

    tempo_no2 = ds_filtered["total_column_NO2"].isel(time=0)

    tempo_no2 = tempo_no2.sel(latitude=slice(MIN_LAT, MAX_LAT), longitude=slice(MIN_LON, MAX_LON))
    tempo_no2 = tempo_no2.sortby("latitude").sortby("longitude")

    tempo_coarsened = tempo_no2.coarsen(latitude=20, longitude=20, boundary="trim").mean()

    tempo_on_grid = tempo_coarsened.interp(
        latitude=aurora_grid.latitude,
        longitude=aurora_grid.longitude,
        method="nearest",
    )

    ds_root.close()

    return tempo_on_grid


def add_cams_surface_features(data, cams_surface_region, aurora_grid, time_index, suffix):
    for var in CAMS_SURFACE_VARS:
        if var not in cams_surface_region:
            continue

        da = squeeze_cams_surface(cams_surface_region[var], time_index)
        da = convert_lon_to_minus180_180(da)

        da = da.interp(
            latitude=aurora_grid.latitude,
            longitude=aurora_grid.longitude,
            method="nearest",
        )

        if var == "tcno2":
            da_converted = convert_no2_kgm2_to_molec_cm2(da)
            data[f"cams_tcno2_molecules_cm2_{suffix}"] = da_converted.values.ravel()
        else:
            data[f"cams_{var}_{suffix}"] = da.values.ravel()


def process_file(aurora_path):
    # Builds per pixel data table that then saves to csv, includes all CAMS variables mentioned
    if not os.path.exists(aurora_path):
        return None, f"NO_AURORA_FILE {aurora_path}"

    aurora_ds = xr.open_dataset(aurora_path)

    if "rollout_step" not in aurora_ds or "time" not in aurora_ds:
        aurora_ds.close()
        return None, "MISSING_ROLLOUT_STEP_OR_TIME"

    rollout_step = int(aurora_ds["rollout_step"].values)
    target_time = pd.to_datetime(aurora_ds["time"].values[0])
    target_date = target_time.date()

    init_datetime = target_time - pd.Timedelta(hours=12 * rollout_step)
    init_date = init_datetime.date()

    cams_surface_path = cams_surface_path_for_date(init_date)
    tempo_path = tempo_path_for_target_date(target_date)

    if not os.path.exists(cams_surface_path):
        aurora_ds.close()
        return None, f"NO_CAMS_SURFACE_FILE {cams_surface_path}"

    if not os.path.exists(tempo_path):
        aurora_ds.close()
        return None, f"NO_TEMPO_TARGET_FILE {tempo_path}"

    aurora_region = crop_360(aurora_ds)

    if "surf_tcno2" not in aurora_region:
        aurora_ds.close()
        return None, "NO_AURORA_SURF_TCNO2"

    aurora_pred_no2_raw = squeeze_aurora_surface(aurora_region["surf_tcno2"])
    aurora_pred_no2 = convert_no2_kgm2_to_molec_cm2(aurora_pred_no2_raw)
    aurora_pred_no2 = convert_lon_to_minus180_180(aurora_pred_no2)
    aurora_pred_no2.name = "aurora_pred_no2_molecules_cm2"

    tempo_no2 = open_tempo_on_aurora_grid(tempo_path, aurora_pred_no2)

    residual = tempo_no2 - aurora_pred_no2

    lat_vals = aurora_pred_no2.latitude.values
    lon_vals = aurora_pred_no2.longitude.values

    data = {
        "target_date": target_date,
        "target_time": target_time,
        "init_date": init_date,
        "init_datetime": init_datetime,
        "rollout_step": rollout_step,
        "lead_label": f"t+{rollout_step}",
        "aurora_file": os.path.basename(aurora_path),
        "cams_surface_file": os.path.basename(cams_surface_path),
        "tempo_file": os.path.basename(tempo_path),
        "latitude": np.repeat(lat_vals, len(lon_vals)),
        "longitude": np.tile(lon_vals, len(lat_vals)),
        "aurora_pred_no2_molecules_cm2": aurora_pred_no2.values.ravel(),
        "tempo_no2_molecules_cm2": tempo_no2.values.ravel(),
        "residual_molecules_cm2": residual.values.ravel(),
    }

    cams_surface_ds = xr.open_dataset(cams_surface_path)
    cams_surface_region = crop_360(cams_surface_ds)
    # forecast_reference_time index 0 = CAMS 00Z on init_date (t-1)
    # forecast_reference_time index 1 = CAMS 12Z on init_date (t)
    add_cams_surface_features(data, cams_surface_region, aurora_pred_no2, 0, "00z")
    add_cams_surface_features(data, cams_surface_region, aurora_pred_no2, 1, "12z")

    df = pd.DataFrame(data)

    rows_before = len(df)

    df = df.dropna(
        subset=[
            "aurora_pred_no2_molecules_cm2",
            "tempo_no2_molecules_cm2",
            "residual_molecules_cm2",
        ]
    )

    df = df[
        (df["tempo_no2_molecules_cm2"] >= 0) &
        (df["tempo_no2_molecules_cm2"] <= 1e16) &
        (df["aurora_pred_no2_molecules_cm2"] >= 0) &
        (df["aurora_pred_no2_molecules_cm2"] <= 1e16)
    ].copy()

    rows_after = len(df)

    aurora_ds.close()
    cams_surface_ds.close()

    return df, f"OK rollout_step={rollout_step} rows_before={rows_before} rows_after={rows_after}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = []
    for folder in LEAD_FOLDERS:
        folder_path = os.path.join(AURORA_BASE_DIR, folder)
        folder_files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))
        print(f"{folder}: {len(folder_files)} files")
        files.extend(folder_files)

    print("Total Aurora files found:", len(files))
    print("Output files:", list(OUTPUT_CSV_BY_ROLLOUT_STEP.values()))

    all_rows = []
    skipped = []

    for i, path in enumerate(files):
        print()
        print("=" * 100)
        print(f"[{i+1}/{len(files)}]", os.path.basename(path))

        try:
            df, status = process_file(path)
            print("Status:", status)

            if df is None or len(df) == 0:
                skipped.append({
                    "aurora_file": os.path.basename(path),
                    "reason": status,
                })
                continue

            all_rows.append(df)

        except Exception as e:
            print("SKIPPING error:", e)
            skipped.append({
                "aurora_file": os.path.basename(path),
                "reason": f"ERROR: {e}",
            })

    if len(all_rows) == 0:
        print("No rows created.")
        pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)
        return

    out = pd.concat(all_rows, ignore_index=True)

    out["target_date"] = pd.to_datetime(out["target_date"])
    out["init_date"] = pd.to_datetime(out["init_date"])

    out["month_target"] = out["target_date"].dt.month
    out["day_target"] = out["target_date"].dt.day
    out["dayofyear_target"] = out["target_date"].dt.dayofyear

    pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)

    print()
    print("=" * 100)
    print("Rows total:", len(out))
    print("Target dates:", out["target_date"].nunique())
    print("Lead steps present:", sorted(out["rollout_step"].unique()))
    print("Skipped:", len(skipped))
    print("Skipped log:", SKIPPED_CSV)

    print()
    print("Residual summary by lead step:")
    print(out.groupby("lead_label")["residual_molecules_cm2"].describe())

    # Save one CSV per lead time (rollout_step), dropping columns not needed downstream.
    for rollout_step, out_path in OUTPUT_CSV_BY_ROLLOUT_STEP.items():
        lead_df = out[out["rollout_step"] == rollout_step].copy()

        if lead_df.empty:
            print(f"\nNo rows for rollout_step={rollout_step}, skipping {out_path}")
            continue

        lead_df = lead_df.drop(columns=[c for c in COLUMNS_TO_DROP_ON_SAVE if c in lead_df.columns])
        lead_df.to_csv(out_path, index=False)

        print(f"\nSaved rollout_step={rollout_step} ({len(lead_df)} rows) -> {out_path}")
        print("Columns:", len(lead_df.columns))


if __name__ == "__main__":
    main()