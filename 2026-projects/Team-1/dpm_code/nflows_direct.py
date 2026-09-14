"""
Trains a normalizing flow to predict TEMPO NO2 directly from raw CAMS
inputs, one lead time ahead, no Aurora forecast involved at all. Then
loads the residual model's saved results and builds a 3-way comparison:
Aurora baseline, residual-corrected, and direct flow prediction.
 
Run nflows_time_series.py first so its results csv exists before this.
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
HORIZON_LABEL = "t1"
TRAIN_CSV_PATH = f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_train_{HORIZON_LABEL}.csv"
TEST_CSV_PATH = f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_test_{HORIZON_LABEL}.csv"
RESIDUAL_RESULTS_PATH = f"results_residual_{HORIZON_LABEL}.csv"
 
# predicting tempo directly, not a residual
TARGET_COL = "tempo_no2_molecules_cm2"
 
# not a model input, only used for comparison after training
AURORA_COL = "aurora_pred_no2_molecules_cm2"
 
DATE_COL = "target_date"
 
LAT_MIN, LAT_MAX = 31.31, 49.01
LON_MIN, LON_MAX = -125.45, -101.98
 
# same raw CAMS inputs as the residual model, t-1 (00z) and t (12z)
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
 
MODEL_SAVE_PATH = f"flow_direct_{HORIZON_LABEL}.pt"
SCALER_SAVE_PATH = f"scaler_direct_{HORIZON_LABEL}.npz"
MAPS_DIR = f"/umbc/rs/cybertrn/reu2026/team1/research/testing/Michael_testing/nflows_model/comparison_maps/{HORIZON_LABEL}"
PLOTS_DIR = f"/umbc/rs/cybertrn/reu2026/team1/research/testing/Michael_testing/nflows_model/comparison_plots/{HORIZON_LABEL}"
 
# loading train and test data
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)
 
# converting to real dates needed for day.date() later in the plotting section
train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])
 
print(f"Loaded {len(train_df)} train rows from {TRAIN_CSV_PATH}")
print(f"Loaded {len(test_df)} test rows from {TEST_CSV_PATH}")
print(f"Columns available: {list(train_df.columns)}")
 
feature_cols = RAW_FEATURE_COLS
print(f"Using {len(feature_cols)} input features")
 
# standardizing features and target due to a wide range of values and extreme outliers
# rescaling to mean=0 and std=1
feature_mean = train_df[feature_cols].mean()
feature_std = train_df[feature_cols].std()
 
target_mean = train_df[TARGET_COL].mean()
target_std = train_df[TARGET_COL].std()
 
 
def make_tensors(subset):
    # converts parameter into a tensor and applies the standardization
    x = (subset[feature_cols] - feature_mean) / feature_std
    x = torch.tensor(x.values, dtype=torch.float32)
    y = (subset[[TARGET_COL]] - target_mean) / target_std
    y = torch.tensor(y.values, dtype=torch.float32)
    return x, y
 
 
# assigning train and test
x_train, y_train = make_tensors(train_df)
x_test, y_test = make_tensors(test_df)
 
np.savez(SCALER_SAVE_PATH, mean=feature_mean.values, std=feature_std.values, columns=feature_cols,
         target_mean=target_mean, target_std=target_std)
 
# training an ensemble of normalizing flows
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]
 
ensemble_predictions = []

for seed in ENSEMBLE_SEEDS:
    torch.manual_seed(seed)
 
    print(len(feature_cols))
    flow = zuko.flows.NSF(
        features=1,  # <- number of features used to predict
        context=len(feature_cols),  # <- number of input values nflows gets to look at
        transforms=3,  # <- number of internal transformation steps
        hidden_features=(64, 64),  # <- size of the internal neural network layers
    )
 
    # updates the model's parameters based on how wrong it was after each batch
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
    torch.save(flow.state_dict(), f"flow_direct_{HORIZON_LABEL}_seed{seed}.pt")
 
    # this member's prediction on the test set, converted to real units
    with torch.no_grad():
        samples = flow(x_test).sample((200,))
        predicted_scaled = samples.mean(dim=0)
    ensemble_predictions.append(predicted_scaled * target_std + target_mean)
 
print(f"\nSaved feature scaling to {SCALER_SAVE_PATH}")
 
# averaging the ensemble, since this is regression
direct_prediction = torch.stack(ensemble_predictions).mean(dim=0)
y_test_real = y_test * target_std + target_mean
 
mae = torch.abs(direct_prediction - y_test_real).mean().item()
rmse = torch.sqrt(((direct_prediction - y_test_real) ** 2).mean()).item()
 
print(f"\nTest set performance ({len(test_df)} pixels across {test_df[DATE_COL].nunique()} days):")
print(f"  MAE:  {mae:.4e}")
print(f"  RMSE: {rmse:.4e}")
 
test_df["direct_flow_no2"] = direct_prediction.numpy()
 
# loading residual model's saved results to bring in aurora baseline and residual-corrected columns
residual_df = pd.read_csv(RESIDUAL_RESULTS_PATH)
residual_df[DATE_COL] = pd.to_datetime(residual_df[DATE_COL])
 
merged_df = test_df.rename(columns={TARGET_COL: "tempo_no2", AURORA_COL: "aurora_baseline_no2"}).merge(
    residual_df[[DATE_COL, "latitude", "longitude", "residual_corrected_no2"]],
    on=[DATE_COL, "latitude", "longitude"],
    how="inner",
)
print(f"Merged into {len(merged_df)} rows common to both models")
 
TEMPO_COL = "tempo_no2"
AURORA_BASELINE_COL = "aurora_baseline_no2"
RESIDUAL_CORRECTED_COL = "residual_corrected_no2"
DIRECT_COL = "direct_flow_no2"
 
 
 
# maps, one per test day: TEMPO truth, Aurora baseline, residual-corrected, direct flow
def _pivot_to_grid(day_df, value_col):
    grid = day_df.pivot_table(index="latitude", columns="longitude", values=value_col)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid.columns.values, grid.index.values, grid.values
 
 
def plot_model_comparison_maps(merged_df, lat_min, lat_max, lon_min, lon_max, save_dir=MAPS_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    # all 4 panels share one scale, 0 to 6 (x10^15 mol/cm^2)
    SHARED_VMIN = 0
    SHARED_VMAX = 6e15
 
    for day, day_df in merged_df.groupby(DATE_COL):
        lons, lats, tempo_grid = _pivot_to_grid(day_df, TEMPO_COL)
        _, _, aurora_grid = _pivot_to_grid(day_df, AURORA_BASELINE_COL)
        _, _, corrected_grid = _pivot_to_grid(day_df, RESIDUAL_CORRECTED_COL)
        _, _, direct_grid = _pivot_to_grid(day_df, DIRECT_COL)
 
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 10),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
 
        panels = [
            (axes[0, 0], tempo_grid, f"TEMPO ground truth, {day.date()}"),
            (axes[0, 1], aurora_grid, f"Aurora baseline, {day.date()}"),
            (axes[1, 0], corrected_grid, "Residual-corrected Aurora"),
            (axes[1, 1], direct_grid, "Direct flow prediction"),
        ]
 
        for ax, grid, title in panels:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.6)
            ax.add_feature(cfeature.STATES, linewidth=0.3)
 
            mesh = ax.pcolormesh(lons, lats, grid, cmap="viridis",
                                  vmin=SHARED_VMIN, vmax=SHARED_VMAX,
                                  transform=ccrs.PlateCarree(), shading="auto")
 
            ax.set_title(title, fontsize=10)
            plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, label="mol/cm^2")
 
        plt.suptitle(f"Model comparison ({HORIZON_LABEL}), {day.date()}", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / f"model_comparison_{day.date()}.png", dpi=150, bbox_inches="tight")
        plt.close()
 
    print(f"Saved comparison maps to {save_dir}")
 
 
plot_model_comparison_maps(merged_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
 
# per-day bias/RMSE table across all 3 variants
daily_rows = []
for day, day_df in merged_df.groupby(DATE_COL):
    tempo_mean = day_df[TEMPO_COL].mean()
 
    variants = {
        "Aurora Baseline": day_df[AURORA_BASELINE_COL],
        "Residual-Corrected": day_df[RESIDUAL_CORRECTED_COL],
        "Direct Flow": day_df[DIRECT_COL],
    }
 
    for name, values in variants.items():
        diff = values - day_df[TEMPO_COL]
        daily_rows.append({
            "date": day,
            "variant": name,
            "mean_bias_pct": (diff.mean() / tempo_mean * 100) if tempo_mean != 0 else np.nan,
            "rmse": np.sqrt((diff ** 2).mean()),
        })
 
daily_metrics_df = pd.DataFrame(daily_rows)
 
 
def plot_three_way_comparison(daily_metrics_df, save_dir=PLOTS_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    variant_order = ["Aurora Baseline", "Residual-Corrected", "Direct Flow"]
    colors = ["steelblue", "seagreen", "darkorange"]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
    bias_data = [daily_metrics_df[daily_metrics_df["variant"] == v]["mean_bias_pct"].dropna() for v in variant_order]
    bp0 = axes[0].boxplot(bias_data, tick_labels=variant_order, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp0["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1, label="Zero bias")
    axes[0].set_ylabel("Mean Bias (%)")
    axes[0].set_title("Mean Bias by Model")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)
 
    rmse_data = [daily_metrics_df[daily_metrics_df["variant"] == v]["rmse"].dropna() for v in variant_order]
    bp1 = axes[1].boxplot(rmse_data, tick_labels=variant_order, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("RMSE (mol/cm^2)")
    axes[1].set_title("RMSE by Model")
    axes[1].grid(axis="y", alpha=0.4)
 
    plt.suptitle(f"Aurora Baseline vs Residual-Corrected vs Direct Flow ({HORIZON_LABEL})", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "three_way_bias_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    # overlaid histograms, bias % and RMSE, all 3 variants
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    for variant, color in zip(variant_order, colors):
        subset = daily_metrics_df[daily_metrics_df["variant"] == variant]
        axes[0].hist(subset["mean_bias_pct"].dropna(), bins=15, color=color, alpha=0.5,
                     edgecolor="white", label=variant)
        axes[1].hist(subset["rmse"].dropna(), bins=15, color=color, alpha=0.5,
                     edgecolor="white", label=variant)
 
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Mean Bias (%)")
    axes[0].set_ylabel("Number of Days")
    axes[0].set_title("Distribution of Mean Bias")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
 
    axes[1].set_xlabel("RMSE (mol/cm^2)")
    axes[1].set_ylabel("Number of Days")
    axes[1].set_title("Distribution of RMSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
 
    plt.suptitle(f"Distribution of Mean Bias and RMSE by Model ({HORIZON_LABEL})", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "three_way_histogram_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
 
    print(f"Saved three-way comparison plot to {save_dir}")
 
 
plot_three_way_comparison(daily_metrics_df)