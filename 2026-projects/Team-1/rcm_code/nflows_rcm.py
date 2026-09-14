"""
Trains a normalizing flow to predict the residual (TEMPO - Aurora forecast)
one lead time ahead, using raw CAMS variables from two prior timestamps
(00Z and 12Z of the initialization date) as inputs, not Aurora's forecast
of those variables. Run this once per lead time by pointing TRAIN_CSV_PATH
and TEST_CSV_PATH at that lead's files and changing HORIZON_LABEL.
 
Lead times: t1 = 1 day ahead, t3 = 2 days ahead, t5 = 3 days ahead,
"""
 
import numpy as np
import pandas as pd
import torch
import zuko
import matplotlib.pyplot as plt
import cartopy
cartopy.config["data_dir"] = "/umbc/rs/cybertrn/reu2026/team1/research/testing/Oscar_testing/cartopy_data"
 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
 
#fixed seed for reproducible results  
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED) 
 
# Paths 
HORIZON_LABEL = "t1"  #<- 1day=t1 2day=t3 3day=t5
TRAIN_CSV_PATH = f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_train_{HORIZON_LABEL}.csv"
TEST_CSV_PATH = f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_test_{HORIZON_LABEL}.csv"
 
# Column that holds what we're trying to predict.
TARGET_COL = "residual_molecules_cm2"
 
# Aurora's forecast for the target/future date. NOT used as a model input,
# only used afterward to build the corrected prediction and for evaluation.
AURORA_COL = "aurora_pred_no2_molecules_cm2"
TEMPO_COL = "tempo_no2_molecules_cm2"
 
# Column that tells us which future date each row's residual belongs to.
DATE_COL = "target_date"
 
# California bounding box, used only for the map plots at the end.
LAT_MIN, LAT_MAX = 31.31, 49.01
LON_MIN, LON_MAX = -125.45, -101.98
 
# Model inputs are raw CAMS variables from two prior timestamps, 00Z and
# 12Z of the initialization date, acting as t-1 and t. These are the raw
# CAMS varibles. tcno2 here is also raw CAMS, separate from aurora_pred_no2_molecules_cm2 above.
CAMS_VARS = [
    "u10", "v10", "t2m", "msl",
    "pm1", "pm2p5", "pm10",
    "tcco", "tc_no", "tcno2_molecules_cm2", "gtco3", "tcso2",
]
 
RAW_FEATURE_COLS = (
    [f"cams_{v}_00z" for v in CAMS_VARS]
    + [f"cams_{v}_12z" for v in CAMS_VARS]
    + ["latitude", "longitude"]
)
 
EPOCHS = 50
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3  # <- 0.001 learning rate
 
MODEL_SAVE_PATH = f"flow_{HORIZON_LABEL}.pt"
SCALER_SAVE_PATH = f"scaler_{HORIZON_LABEL}.npz"
MAPS_DIR = f"/umbc/rs/cybertrn/reu2026/team1/research/testing/Michael_testing/nflows_model/map/{HORIZON_LABEL}"
PLOTS_DIR = f"/umbc/rs/cybertrn/reu2026/team1/research/testing/Michael_testing/nflows_model/plots/{HORIZON_LABEL}"
 
#Loading train and test data
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
 
#Converting to real dates needed for day.date() later in the plotting section
train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])
 
print(f"Loaded {len(train_df)} train rows from {TRAIN_CSV_PATH}")
print(f"Loaded {len(test_df)} test rows from {TEST_CSV_PATH}")
print(f"Columns available: {list(train_df.columns)}")
 
feature_cols = RAW_FEATURE_COLS
print(f"Using {len(feature_cols)} input features")
 
#standardizing features and residuals due to a wide range of values and extreme outliers
#rescaling to mean=0  and std=1
feature_mean = train_df[feature_cols].mean()
feature_std = train_df[feature_cols].std()

# standardizing residuals
target_mean = train_df[TARGET_COL].mean()
target_std = train_df[TARGET_COL].std()
 
 
def make_tensors(subset):
    # converts parameter into a tensor and applies the standardization
    x = (subset[feature_cols] - feature_mean) / feature_std
    x = torch.tensor(x.values, dtype=torch.float32)
    y = (subset[[TARGET_COL]] - target_mean) / target_std
    y = torch.tensor(y.values, dtype=torch.float32)
    return x, y
 
#assigning train and test 
x_train, y_train = make_tensors(train_df)
x_test, y_test = make_tensors(test_df)
 
np.savez(SCALER_SAVE_PATH, mean=feature_mean.values, std=feature_std.values, columns=feature_cols,
         target_mean=target_mean, target_std=target_std)
 
# training an ensemble of normalizing flows
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
 
ensemble_predictions = []

ensemble_predictions = []
 
for seed in ENSEMBLE_SEEDS:
    torch.manual_seed(seed)
 
    print(len(feature_cols))
    flow = zuko.flows.NSF(
        features=1,  # <- residual value we're predicting
        context=len(feature_cols),  # <- number of input features the model conditions on
        transforms=3,  # <- number of internal transformation steps
        hidden_features=(64, 64),  # <- size of the internal neural network layers
    )
 
    # updates the model's parameters based on how wrong it was after each batch.
    # the adjustments are set by the learning_rate
    optimizer = torch.optim.Adam(flow.parameters(), lr=LEARNING_RATE)
 
    print(f"\nStarting training, seed {seed}...")
    for epoch in range(EPOCHS):
        # shuffles rows of training data after each epoch
        shuffled_indices = torch.randperm(x_train.shape[0])
        total_loss = 0.0
 
        for start in range(0, x_train.shape[0], BATCH_SIZE):
            batch_indices = shuffled_indices[start:start + BATCH_SIZE]
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]
 
            loss = -flow(x_batch).log_prob(y_batch).mean()
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item()
 
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}   loss {total_loss:.4f}")
 
    # saving this ensemble member's model
    torch.save(flow.state_dict(), f"flow_{HORIZON_LABEL}_seed{seed}.pt")
 
    # this member's prediction on the test set, converted to real units
    with torch.no_grad():
        samples = flow(x_test).sample((200,))
        predicted_scaled = samples.mean(dim=0)
    ensemble_predictions.append(predicted_scaled * target_std + target_mean)
 
print(f"\nSaved feature scaling to {SCALER_SAVE_PATH}")
 
# averaging the ensemble, since this is regression
predicted_residual = torch.stack(ensemble_predictions).mean(dim=0)
y_test_real = y_test * target_std + target_mean
 
mae = torch.abs(predicted_residual - y_test_real).mean().item()
rmse = torch.sqrt(((predicted_residual - y_test_real) ** 2).mean()).item()
 
print(f"\nTest set performance ({len(test_df)} pixels across {test_df[DATE_COL].nunique()} days):")
print(f"  MAE:  {mae:.4e}")
print(f"  RMSE: {rmse:.4e}")
 
# builds the corrected prediction: aurora's forecast plus predicted residual
test_df["predicted_residual"] = predicted_residual.numpy()
test_df["corrected_no2"] = test_df[AURORA_COL] + test_df["predicted_residual"]
 
RESULTS_SAVE_PATH = f"results_residual_{HORIZON_LABEL}.csv"
# saving predictions so nflows_direct.py can load them for comparison
test_df[[DATE_COL, "latitude", "longitude", TEMPO_COL, AURORA_COL, "corrected_no2"]].rename(
    columns={TEMPO_COL: "tempo_no2", AURORA_COL: "aurora_baseline_no2", "corrected_no2": "residual_corrected_no2"}
).to_csv(RESULTS_SAVE_PATH, index=False)
print(f"Saved test predictions to {RESULTS_SAVE_PATH}")
 
 
 
# ── Visualize results on a map, one per test day ───────────────────────────────
def _pivot_to_grid(day_df, value_col):
    grid = day_df.pivot_table(index="latitude", columns="longitude", values=value_col)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid.columns.values, grid.index.values, grid.values
 
 
def plot_spatial_comparison_maps(test_df, lat_min, lat_max, lon_min, lon_max, save_dir=MAPS_DIR):
    """
    Same idea as Oscar's XGBoost version: one 4-panel map per test day,
    showing TEMPO (truth), Aurora's forecast for that day, our flow-corrected
    prediction, and the correction we actually applied.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    for day, day_df in test_df.groupby(DATE_COL):
        lons, lats, tempo_grid = _pivot_to_grid(day_df, TEMPO_COL)
        _, _, cams_grid = _pivot_to_grid(day_df, AURORA_COL)
        _, _, corrected_grid = _pivot_to_grid(day_df, "corrected_no2")
        correction_grid = corrected_grid - cams_grid
 
        # All four panels now share one fixed scale, 0 to 6 (x10^15 molecules/cm^2),
        # per mentor feedback. Previously the correction panel got its own
        # inflated scale, which stretched a small correction across the whole
        # colorbar and made it look bigger than it actually was. Sharing one
        # honest scale means small corrections show up small.
        SHARED_VMIN = 0
        SHARED_VMAX = 6e15
 
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 10),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
 
        panels = [
            (axes[0, 0], tempo_grid,
             f"TEMPO ground truth, {day.date()}",
             "viridis", SHARED_VMIN, SHARED_VMAX),
            (axes[0, 1], cams_grid,
             f"Aurora forecast for {day.date()}",
             "viridis", SHARED_VMIN, SHARED_VMAX),
            (axes[1, 0], corrected_grid,
             "Corrected forecast (Aurora + flow correction)",
             "viridis", SHARED_VMIN, SHARED_VMAX),
            (axes[1, 1], correction_grid,
             "Corrected forecast minus Aurora forecast",
             "viridis", SHARED_VMIN, SHARED_VMAX),
        ]
 
        for ax, grid, title, cmap, pmin, pmax in panels:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.6)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
 
            mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap, vmin=pmin, vmax=pmax,
                                  transform=ccrs.PlateCarree(), shading="auto")
 
            ax.set_title(title, fontsize=10)
            plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, label="mol/cm^2")
 
        plt.suptitle(f"Time series correction ({HORIZON_LABEL}), {day.date()}", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / f"map_comparison_{day.date()}.png", dpi=150, bbox_inches="tight")
        plt.close()
 
    print(f"Saved spatial comparison maps to {save_dir}")
 
 
plot_spatial_comparison_maps(test_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
 
# ── Build a per-day bias/RMSE table, baseline vs flow-corrected ───────────────
daily_rows = []
for day, day_df in test_df.groupby(DATE_COL):
    tempo_mean = day_df[TEMPO_COL].mean()
 
    baseline_diff = day_df[AURORA_COL] - day_df[TEMPO_COL]
    corrected_diff = day_df["corrected_no2"] - day_df[TEMPO_COL]
 
    daily_rows.append({
        "date": day,
        "variant": "Baseline (Aurora)",
        "mean_bias_pct": (baseline_diff.mean() / tempo_mean * 100) if tempo_mean != 0 else np.nan,
        "rmse": np.sqrt((baseline_diff ** 2).mean()),
    })
    daily_rows.append({
        "date": day,
        "variant": "Flow-Corrected",
        "mean_bias_pct": (corrected_diff.mean() / tempo_mean * 100) if tempo_mean != 0 else np.nan,
        "rmse": np.sqrt((corrected_diff ** 2).mean()),
    })
 
daily_metrics_df = pd.DataFrame(daily_rows)
 
 
def plot_bias_rmse_comparison(daily_metrics_df, save_dir=PLOTS_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    baseline = daily_metrics_df[daily_metrics_df["variant"] == "Baseline (Aurora)"]
    corrected = daily_metrics_df[daily_metrics_df["variant"] == "Flow-Corrected"]
 
    # Box and whisker: bias % and RMSE, baseline vs corrected
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
 
    axes[0].boxplot(
        [baseline["mean_bias_pct"].dropna(), corrected["mean_bias_pct"].dropna()],
        tick_labels=["Baseline", "Flow-Corrected"], patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1, label="Zero bias")
    axes[0].set_ylabel("Mean Bias (%)")
    axes[0].set_title("Mean Bias: Baseline vs Corrected")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)
 
    axes[1].boxplot(
        [baseline["rmse"].dropna(), corrected["rmse"].dropna()],
        tick_labels=["Baseline", "Flow-Corrected"], patch_artist=True,
        boxprops=dict(facecolor="darkorange", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[1].set_ylabel("RMSE (mol/cm^2)")
    axes[1].set_title("RMSE: Baseline vs Corrected")
    axes[1].grid(axis="y", alpha=0.4)
 
    plt.suptitle("Aurora vs TEMPO NO2, Baseline vs Flow-Corrected", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "boxplot_bias_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    # Overlaid histograms: bias % and RMSE, baseline vs corrected
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    axes[0].hist(baseline["mean_bias_pct"].dropna(), bins=15, color="steelblue", alpha=0.5,
                 edgecolor="white", label="Baseline")
    axes[0].hist(corrected["mean_bias_pct"].dropna(), bins=15, color="seagreen", alpha=0.5,
                 edgecolor="white", label="Flow-Corrected")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Mean Bias (%)")
    axes[0].set_ylabel("Number of Days")
    axes[0].set_title("Distribution of Mean Bias")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
 
    axes[1].hist(baseline["rmse"].dropna(), bins=15, color="darkorange", alpha=0.5,
                 edgecolor="white", label="Baseline")
    axes[1].hist(corrected["rmse"].dropna(), bins=15, color="seagreen", alpha=0.5,
                 edgecolor="white", label="Flow-Corrected")
    axes[1].set_xlabel("RMSE (mol/cm^2)")
    axes[1].set_ylabel("Number of Days")
    axes[1].set_title("Distribution of RMSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
 
    plt.suptitle("Aurora vs TEMPO NO2, Metric Distributions, Baseline vs Flow-Corrected", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "histogram_bias_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    print(f"Saved bias/RMSE comparison plots to {save_dir}")
 
 
plot_bias_rmse_comparison(daily_metrics_df)