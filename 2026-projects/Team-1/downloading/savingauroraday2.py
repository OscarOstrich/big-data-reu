#https://microsoft.github.io/aurora/example_cams.html
#Title python file: savingauroraday2.py
import datetime
import time
import sys
import aurora
print("Python Executable:", sys.executable)
print("Aurora Source Path:", aurora.__file__)
print("Available names inside this module:", dir(aurora))
import zipfile
from pathlib import Path
import cdsapi
from huggingface_hub import hf_hub_download
import pickle
import torch
import xarray as xr
from aurora import Batch, Metadata
from aurora import rollout

# Data will be downloaded here.
#download_path = Path("~/downloads/cams")
download_path = Path("/umbc/rs/cybertrn/reu2026/team1/research/data/cams")
download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

# Set to `False` to run locally and to `True` to run on Foundry.
run_on_foundry = False

if not run_on_foundry:
    from aurora import AuroraAirPollution, rollout
    model = AuroraAirPollution()
    model.load_checkpoint("microsoft/aurora", "aurora-0.4-air-pollution.ckpt")
    model.eval()
    model = model.to("cuda")

#    with torch.inference_mode():
#        predictions = [pred.to("cpu") for pred in rollout(model, batch, steps=730)]
#    model = model.to("cpu")

if run_on_foundry:
    import logging
    import os
    import warnings

    from aurora.foundry import BlobStorageChannel, FoundryClient, submit


    # In this demo, we silence all warnings.
    warnings.filterwarnings("ignore")

    # But we do want to show what's happening under the hood!
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("aurora").setLevel(logging.INFO)




    foundry_client = FoundryClient(
        endpoint=os.environ["FOUNDRY_ENDPOINT"],
        token=os.environ["FOUNDRY_TOKEN"],
    )
    channel = BlobStorageChannel(os.environ["BLOB_URL_WITH_SAS"])

# Download the static variables from HuggingFace.
static_path = hf_hub_download(
    repo_id="microsoft/aurora",
    filename="aurora-0.4-air-pollution-static.pickle",
)
print("Static variables downloaded!")

date = datetime.datetime(2024, 7, 24)

for i in range(365):
    print(date)
    # Download the surface-level variables.
    if not (download_path / f"{date.strftime('%Y-%m-%d')}-cams.nc.zip").exists():
        c = cdsapi.Client()
        c.retrieve(
            "cams-global-atmospheric-composition-forecasts",
            {
                "type": "forecast",
                "leadtime_hour": "0",
                "variable": [
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
                ],
                "pressure_level": [
                    "50",
                    "100",
                    "150",
                    "200",
                    "250",
                    "300",
                    "400",
                    "500",
                    "600",
                    "700",
                    "850",
                    "925",
                    "1000",
                ],
                "date": date.strftime('%Y-%m-%d'),
                "time": ["00:00", "12:00"],
                "format": "netcdf_zip",
            },
            str(download_path / f"{date.strftime('%Y-%m-%d')}-cams.nc.zip"),
        )
    # Unpack the ZIP. It should contain the surface-level and atmospheric data in separate
    # files.
    if not (download_path / f"{date.strftime('%Y-%m-%d')}-cams-surface-level.nc").exists():
        with zipfile.ZipFile(download_path / f"{date}-cams.nc.zip", "r") as zf, open(
            download_path / f"{date.strftime('%Y-%m-%d')}-cams-surface-level.nc", "wb"
        ) as f:
            f.write(zf.read("data_sfc.nc"))
    if not (download_path / f"{date.strftime('%Y-%m-%d')}-cams-atmospheric.nc").exists():
        with zipfile.ZipFile(download_path / f"{date.strftime('%Y-%m-%d')}-cams.nc.zip", "r") as zf, open(
            download_path / f"{date.strftime('%Y-%m-%d')}-cams-atmospheric.nc", "wb"
        ) as f:
            f.write(zf.read("data_plev.nc"))
    print("Surface-level and atmospheric variables downloaded!")
    with open(static_path, "rb") as f:
        static_vars = pickle.load(f)
    surf_vars_ds = xr.open_dataset(
        download_path / f"{date.strftime('%Y-%m-%d')}-cams-surface-level.nc", engine="netcdf4", decode_timedelta=True
    )
    atmos_vars_ds = xr.open_dataset(
        download_path / f"{date.strftime('%Y-%m-%d')}-cams-atmospheric.nc", engine="netcdf4", decode_timedelta=True
    )

    # Select the zero-hour forecast to get the analysis product.
    surf_vars_ds = surf_vars_ds.isel(forecast_period=0)
    atmos_vars_ds = atmos_vars_ds.isel(forecast_period=0)
    
    # The file has two time points: UTC 00 and UTC 12. We use both to construct the batch
    # with time 2023-07-24 UTC 12.
    batch = Batch(
        surf_vars={
            # `[None]` inserts a batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[None]),
            "pm1": torch.from_numpy(surf_vars_ds["pm1"].values[None]),
            "pm2p5": torch.from_numpy(surf_vars_ds["pm2p5"].values[None]),
            "pm10": torch.from_numpy(surf_vars_ds["pm10"].values[None]),
            "tcco": torch.from_numpy(surf_vars_ds["tcco"].values[None]),
            "tc_no": torch.from_numpy(surf_vars_ds["tc_no"].values[None]),
            "tcno2": torch.from_numpy(surf_vars_ds["tcno2"].values[None]),
            "gtco3": torch.from_numpy(surf_vars_ds["gtco3"].values[None]),
            "tcso2": torch.from_numpy(surf_vars_ds["tcso2"].values[None]),
        },
        static_vars={k: torch.from_numpy(v) for k, v in static_vars.items()},
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[None]),
            "co": torch.from_numpy(atmos_vars_ds["co"].values[None]),
            "no": torch.from_numpy(atmos_vars_ds["no"].values[None]),
            "no2": torch.from_numpy(atmos_vars_ds["no2"].values[None]),
            "go3": torch.from_numpy(atmos_vars_ds["go3"].values[None]),
            "so2": torch.from_numpy(atmos_vars_ds["so2"].values[None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(atmos_vars_ds.latitude.values),
            lon=torch.from_numpy(atmos_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s.
            time=(atmos_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[-1],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )

#    predictions = list(
#        submit(
#            batch,
#            model_name="aurora-0.4-air-pollution",
#            num_steps=730,
#            foundry_client=foundry_client,
#            channel=channel,
#        )
#    )
    
    import matplotlib.pyplot as plt




#fig, axs = plt.subplots(2, 2, figsize=(12, 7))




#for i in range(4):
#    ax = axs[i // 2, i % 2]
#    pred = predictions[i]
#    ax.imshow(pred.surf_vars["tcno2"][0, 0].numpy() / 1e-6, vmin=0, #vmax=10, cmap="Blues")
#    ax.set_title(f"TC NO${{}}_2$ {pred.metadata.time[0]}")
#    ax.set_xticks([])
#    ax.set_yticks([])


#pred.to_netcdf("/umbc/rs/cybertrn/reu2026/team1/research/testing/Addy_testing/tc24predictions.nc")

    dir = Path("/umbc/rs/cybertrn/reu2026/team1/research/testing/Addy_testing/CamsYearRedo/threedayyear/TwoDayPred")
    dir.mkdir(parents=True, exist_ok=True)


    with torch.inference_mode():
        for step, pred in enumerate(rollout(model, batch, steps=4)):
            pred = pred.to("cpu")


            timestamp = pred.metadata.time[0]
   
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            hour = timestamp.hour
            if step == 2 or step == 3:
                filename = f"tcaurora_{timestamp.year:04d}_{timestamp.month:02d}_{day:02d}_{hour:02d}.nc"
                pred.to_netcdf(dir / filename)


            del pred


    print(f"Successfully saved the Aurora prediction files for {date.strftime('%Y-%m-%d')}.")
    date += datetime.timedelta(days=1)
