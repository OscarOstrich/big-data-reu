import os
import numpy as np
import pandas as pd

# horizon label number for the train, testing split and residual csv
LABEL_NUM = "1"

# Point this at whichever of the three per-lead CSVs you're splitting, form: residual_table_t+(1,3,5)_no2.csv
INPUT_CSV = f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/residual_table_t+{LABEL_NUM}_no2.csv"
OUTPUT_DIR = "/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits"

TRAIN_CSV = os.path.join(OUTPUT_DIR, f"residual_train_t{LABEL_NUM}.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, f"residual_test_t{LABEL_NUM}.csv")

# Test set = the last 2 months of the actual date range present in the
# data (not fixed calendar months), by target_date. Since the data spans
# July 2024 - July 2025, this makes test = the last 2 months ending on the
# max target_date found (e.g. ~June-July 2025), train = everything before
# that cutoff. TRAIN_FRAC_TARGET is only used as a sanity-check print
N_TEST_MONTHS = 2
TRAIN_FRAC_TARGET = 0.80

# Splitting on target_date keeps every row for a given TEMPO ground-truth
# day together, and the split is done on the set of unique days, then
# applied back to all rows (pixels) for that day so pixels are never split across train/test independently of each other
SPLIT_DAY_COL = "target_date"

CAMS_SURFACE_VARS = [
    "u10", "v10", "t2m", "msl", "pm1", "pm2p5", "pm10",
    "tcco", "tc_no", "tcno2", "gtco3", "tcso2",
]


def cams_col_base(var):
    return "cams_tcno2_molecules_cm2" if var == "tcno2" else f"cams_{var}"


def day_split_last_n_months(df, day_col=SPLIT_DAY_COL, n_test_months=N_TEST_MONTHS):
    # function to split the dataframe into train and test sets based on the last n months of data, using whole calendar days

    days = pd.to_datetime(df[day_col]).dt.normalize()

    max_day = days.max()
    cutoff = max_day - pd.DateOffset(months=n_test_months)

    test_mask = days > cutoff
    train_mask = ~test_mask

    train_days = sorted(days[train_mask].unique())
    test_days = sorted(days[test_mask].unique())

    return df[train_mask].copy(), df[test_mask].copy(), train_days, test_days, cutoff


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=["target_date", "init_date"])

    # 00z/12z CAMS features are the initialization-time inputs shared across all leads; drop rows missing any of them
    cams_cols = [f"{cams_col_base(v)}_00z" for v in CAMS_SURFACE_VARS] + \
                [f"{cams_col_base(v)}_12z" for v in CAMS_SURFACE_VARS]
    cams_cols = [c for c in cams_cols if c in df.columns]

    rows_before = len(df)
    df = df.dropna(subset=cams_cols)
    print(f"Dropped {rows_before - len(df)} rows missing a CAMS feature")

    train_df, test_df, train_days, test_days, cutoff = day_split_last_n_months(df)
    assert not (set(train_days) & set(test_days)), "Day leakage detected"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    n_days = len(train_days) + len(test_days)
    train_day_frac = len(train_days) / n_days if n_days else float("nan")

    print(f"Test cutoff (exclusive, test = dates after this): {cutoff.date()}")
    print(f"Days: train={len(train_days)}, test={len(test_days)} "
          f"(train day fraction={train_day_frac:.3f}, target={TRAIN_FRAC_TARGET})")
    print(f"Train date range: {min(train_days) if train_days else 'n/a'} "
          f"to {max(train_days) if train_days else 'n/a'}")
    print(f"Test date range: {min(test_days) if test_days else 'n/a'} "
          f"to {max(test_days) if test_days else 'n/a'}")
    print(f"Rows: train={len(train_df)}, test={len(test_df)}")
    print()
    print("lead_label counts (train):")
    print(train_df["lead_label"].value_counts())
    print()
    print("lead_label counts (test):")
    print(test_df["lead_label"].value_counts())


if __name__ == "__main__":
    main()
