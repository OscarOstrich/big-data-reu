import os
import sys
import datetime
from harmony import Client, Collection, Request, Environment

# --------------------------------------------------
# Usage:
#   python download_tempo_one_day.py YYYY-MM-DD
#
# Example:
#   python download_tempo_one_day.py 2023-08-03
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python download_tempo_one_day.py YYYY-MM-DD")
    sys.exit(1)

date_string = sys.argv[1]

try:
    target_date = datetime.datetime.strptime(date_string, "%Y-%m-%d").date()
except ValueError:
    print("Date format must be YYYY-MM-DD")
    sys.exit(1)

# NASA Harmony production environment
harmony_client = Client(env=Environment.PROD)

# TEMPO NO2 L3 collection
collection_id = "C2930763263-LARC_CLOUD"

# Base output directory on HPC research storage
base_output_dir = "/umbc/rs/cybertrn/reu2026/team1/research/data/tempo_l3/ECF"

# Output folder: raw/YYYY/MM/DD
output_dir = os.path.join(
    base_output_dir,
    target_date.strftime("%Y"),
    target_date.strftime("%m"),
    target_date.strftime("%d"),
)

os.makedirs(output_dir, exist_ok=True)

start_time = datetime.datetime.combine(target_date, datetime.time(0, 0, 0))
stop_time = start_time + datetime.timedelta(days=1)

print("=" * 80)
print(f"Target date: {target_date}")
print(f"Start time:  {start_time}")
print(f"Stop time:   {stop_time}")
print(f"Output dir:  {output_dir}")

variables = [
    "product/vertical_column_troposphere",
    "product/vertical_column_stratosphere",
    "product/main_data_quality_flag",
    "support_data/eff_cloud_fraction",
    "latitude",
    "longitude",
]

request = Request(
    collection=Collection(id=collection_id),
    temporal={
        "start": start_time,
        "stop": stop_time,
    },
    variables=variables,
)

try:
    print("Submitting Harmony request...")
    job_id = harmony_client.submit(request)
    print(f"jobID = {job_id}")

    print("Waiting for processing...")
    harmony_client.wait_for_processing(job_id, show_progress=True)

    print(f"Downloading files to: {output_dir}")
    results = harmony_client.download_all(
        job_id,
        directory=output_dir,
        overwrite=False,
    )

    all_results_stored = [f.result() for f in results]

    print(f"Number of result files: {len(all_results_stored)}")
    for f in all_results_stored:
        print(f)

    print("Download finished successfully.")

except Exception as e:
    message = str(e)

    if "No matching granules found" in message:
        print("No matching granules found for this date. Skipping.")
        sys.exit(0)

    print("Unexpected error:")
    print(e)
    sys.exit(2)
