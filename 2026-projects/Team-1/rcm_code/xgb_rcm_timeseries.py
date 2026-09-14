import os
import cartopy
cartopy.config["data_dir"] = '/umbc/rs/cybertrn/reu2026/team1/research/testing/Oscar_testing/cartopy_data'

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


output_dir = Path("/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/Plots")

HORIZON_LABEL = "t5"  # match whatever horizon this run is for

maps_dir = output_dir / f"xgb_ts_{HORIZON_LABEL}" / f"maps_xgb_{HORIZON_LABEL}"          # spatial, bounding-box visualizations
metrics_dir = output_dir / f"xgb_ts_{HORIZON_LABEL}" / f"metrics_xgb_{HORIZON_LABEL}"     # bias / RMSE distribution visualizations
misc_dir = output_dir / f"xgb_ts_{HORIZON_LABEL}" / f"misc_{HORIZON_LABEL}"             # csv of all computed values
misc_dir.mkdir(parents=True, exist_ok=True)
maps_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Column names in the new per-lead train/test CSVs
DATE_COL = "target_date"
TEMPO_COL = "tempo_no2_molecules_cm2"
AURORA_COL = "aurora_pred_no2_molecules_cm2"
RESIDUAL_COL = "residual_molecules_cm2"

CAMS_SURFACE_VARS = [
    "u10", "v10", "t2m", "msl", "pm1", "pm2p5", "pm10",
    "tcco", "tc_no", "tcno2", "gtco3", "tcso2",
]


def cams_col_base(var):
    return "cams_tcno2_molecules_cm2" if var == "tcno2" else f"cams_{var}"


def build_feature_cols(extra_cols=("latitude", "longitude")):
    # Builds the time steps 
    feature_cols = list(extra_cols)
    for var in CAMS_SURFACE_VARS:
        base = cams_col_base(var)
        feature_cols.append(f"{base}_00z")
        feature_cols.append(f"{base}_12z")
    return feature_cols


# Simple function to take the training and testing csvs into dataframes and prints to see columns
def pull_train_test(train_path, test_path):

    train_df = pd.read_csv(train_path, parse_dates=[DATE_COL])
    test_df = pd.read_csv(test_path, parse_dates=[DATE_COL])

    print(train_df.head())
    print(test_df.head())

    return train_df, test_df

# train_residual_model: trains an XGBoost regressor to predict the residual (TEMPO - Aurora) using specified features, evaluates its performance, and augments the test DataFrame with predictions and corrected values
def train_residual_model(train_df, test_df,
                          feature_cols=None,
                          target_col=RESIDUAL_COL,
                          model_path=None):
    if feature_cols is None:
        feature_cols = build_feature_cols()
    feature_cols = list(feature_cols)

    # Drop rows missing any required feature/target before fitting, since a missing CAMS lag file upstream would otherwise feed NaNs into XGBoost.
    required_cols = feature_cols + [target_col]
    train_df = train_df.dropna(subset=required_cols).copy()
    test_df = test_df.dropna(subset=required_cols).copy()

    # Split each dataframe into the feature matrix (X) and the target (residual) not tempo_no2/aurora_no2 directly
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Train an XGBoost regressor to predict the residual (TEMPO - Aurora) and edit hyperparammeters here
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,          
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="mae",
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Augment test_df with predictions so downstream plotting has everything raw Aurora, TEMPO truth, and the corrected Aurora value.
    test_df = test_df.copy()
    test_df["predicted_residual"] = model.predict(X_test)
    test_df["corrected_no2"] = test_df[AURORA_COL] + test_df["predicted_residual"]

    # Compute metrics for both the baseline (raw Aurora) and the corrected values
    baseline_mae = mean_absolute_error(test_df[TEMPO_COL], test_df[AURORA_COL])
    baseline_rmse = np.sqrt(mean_squared_error(test_df[TEMPO_COL], test_df[AURORA_COL]))

    # Compute metrics for the corrected values
    corrected_mae = mean_absolute_error(test_df[TEMPO_COL], test_df["corrected_no2"])
    corrected_rmse = np.sqrt(mean_squared_error(test_df[TEMPO_COL], test_df["corrected_no2"]))
    corrected_r2 = r2_score(test_df[TEMPO_COL], test_df["corrected_no2"])

    # Store results in a dictionary for easy access and potential saving to CSV
    results = {
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "corrected_mae": corrected_mae,
        "corrected_rmse": corrected_rmse,
        "corrected_r2": corrected_r2,
        "mae_improvement_pct": 100 * (baseline_mae - corrected_mae) / baseline_mae,
        "rmse_improvement_pct": 100 * (baseline_rmse - corrected_rmse) / baseline_rmse,
    }

    print("\n--- Test set performance (uncorrected Aurora vs TEMPO) ---")
    print(f"MAE:  {baseline_mae:.4e}   RMSE: {baseline_rmse:.4e}")
    print("--- Test set performance (XGBoost-corrected Aurora vs TEMPO) ---")
    print(f"MAE:  {corrected_mae:.4e}   RMSE: {corrected_rmse:.4e}   R2: {corrected_r2:.4f}")
    print(f"Improvement: MAE {results['mae_improvement_pct']:.1f}%, RMSE {results['rmse_improvement_pct']:.1f}%")

    if model_path is not None:
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    return model, results, test_df


def compute_daily_metrics(test_df):
    # Gathers the daily metrics and computes them for both the baseline (raw Aurora) and the XGBoost-corrected values

    records = []
    for date, group in test_df.groupby(DATE_COL):
        tempo = group[TEMPO_COL]
        tempo_mean = tempo.mean()

        for variant, pred in [("Baseline (Aurora)", group[AURORA_COL]),
                               ("XGB-Corrected", group["corrected_no2"])]:
            diff = tempo - pred
            records.append({
                "date": date,
                "variant": variant,
                "mean_diff": diff.mean(),
                "mae": diff.abs().mean(),
                "rmse": np.sqrt((diff ** 2).mean()),
                "mean_bias_pct": (diff.mean() / tempo_mean * 100) if tempo_mean != 0 else np.nan,
            })

    return pd.DataFrame(records)

# Visualizations

# plot_feature_importance: generates a horizontal bar chart of feature importances from the trained XGBoost model
def plot_feature_importance(model, feature_cols, save_path=None):
    # Generates a horizontal bar chart of feature importances from the trained XGBoost model.
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1], color="steelblue")
    ax.set_xlabel("Feature importance")
    ax.set_title("XGBoost residual model — feature importance")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# Helper function to pivot the DataFrame into a grid format suitable for spatial plotting
def _pivot_to_grid(day_df, value_col):
    grid = day_df.pivot_table(index="latitude", columns="longitude", values=value_col)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid.columns.values, grid.index.values, grid.values

# plot_spatial_comparison_maps: generates side-by-side spatial comparison maps for each test day, showing TEMPO (truth), raw Aurora, XGBoost-corrected Aurora, and the correction applied
def plot_spatial_comparison_maps(test_df, lat_min, lat_max, lon_min, lon_max, save_dir=maps_dir):
    # Generates a 4-panel spatial comparison map for each individual test day

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each unique date in the test DataFrame and create a 4-panel map for that day
    for date, day_df in test_df.groupby(DATE_COL):
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        lons, lats, tempo_grid = _pivot_to_grid(day_df, TEMPO_COL)
        _, _, aurora_grid = _pivot_to_grid(day_df, AURORA_COL)
        _, _, corrected_grid = _pivot_to_grid(day_df, "corrected_no2")
        correction_grid = corrected_grid - aurora_grid

        # all 4 panels share one scale, 0 to 6 (x10^15 mol/cm^2)
        SHARED_VMIN = 0
        SHARED_VMAX = 6e15

        # Create a 2x2 subplot layout for the four panels: TEMPO, raw Aurora, corrected Aurora, and the correction applied
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 10),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        panels = [
            (axes[0, 0], tempo_grid, "TEMPO (truth)", "viridis", SHARED_VMIN, SHARED_VMAX),
            (axes[0, 1], aurora_grid, "Aurora (uncorrected)", "viridis", SHARED_VMIN, SHARED_VMAX),
            (axes[1, 0], corrected_grid, "Aurora + XGBoost correction", "viridis", SHARED_VMIN, SHARED_VMAX),
            (axes[1, 1], correction_grid, "Correction applied (corrected - raw)", "viridis", SHARED_VMIN, SHARED_VMAX),
        ]

        for ax, grid, title, cmap, pmin, pmax in panels:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.6)
            ax.add_feature(cfeature.STATES, linewidth=0.3)

            if pmin is None:
                # symmetric diverging scale centered on zero for the correction panel
                bound = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
                mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap, vmin=-bound, vmax=bound,
                                      transform=ccrs.PlateCarree(), shading="auto")
            else:
                mesh = ax.pcolormesh(lons, lats, grid, cmap=cmap, vmin=pmin, vmax=pmax,
                                      transform=ccrs.PlateCarree(), shading="auto")

            ax.set_title(title, fontsize=11)
            plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8, label="mol/cm^2" if pmin is not None else "Δ mol/cm^2")

        plt.suptitle(f"Aurora vs TEMPO NO2 — {date_str}", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f"map_comparison_{date_str}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved spatial comparison maps to {save_dir}")

# plot_bias_rmse_comparison: generates boxplots and histograms comparing the bias and RMSE distributions for the baseline (raw Aurora) and XGBoost-corrected values
def plot_bias_rmse_comparison(daily_metrics_df, save_dir=metrics_dir):
    # Saves to metrics_dir by default, but can be overridden for testing or other purposes

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    baseline = daily_metrics_df[daily_metrics_df["variant"] == "Baseline (Aurora)"]
    corrected = daily_metrics_df[daily_metrics_df["variant"] == "XGB-Corrected"]

    # Box and whisker: bias % and RMSE, baseline vs corrected
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].boxplot(
        [baseline["mean_bias_pct"].dropna(), corrected["mean_bias_pct"].dropna()],
        tick_labels=["Baseline", "XGB-Corrected"], patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    # Add a horizontal line at zero bias for reference
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1, label="Zero bias")
    axes[0].set_ylabel("Mean Bias (%)")
    axes[0].set_title("Mean Bias: Baseline vs Corrected")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.4)

    axes[1].boxplot(
        [baseline["rmse"].dropna(), corrected["rmse"].dropna()],
        tick_labels=["Baseline", "XGB-Corrected"], patch_artist=True,
        boxprops=dict(facecolor="darkorange", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[1].set_ylabel("RMSE (mol/cm^2)")
    axes[1].set_title("RMSE: Baseline vs Corrected")
    axes[1].grid(axis="y", alpha=0.4)

    plt.suptitle("Aurora vs TEMPO NO2 — Baseline vs XGB-Corrected", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "boxplot_bias_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Overlaid histograms: bias % and RMSE, baseline vs corrected
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(baseline["mean_bias_pct"].dropna(), bins=15, color="steelblue", alpha=0.5,
                 edgecolor="white", label="Baseline")
    axes[0].hist(corrected["mean_bias_pct"].dropna(), bins=15, color="seagreen", alpha=0.5,
                 edgecolor="white", label="XGB-Corrected")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Mean Bias (%)")
    axes[0].set_ylabel("Number of Days")
    axes[0].set_title("Distribution of Mean Bias")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(baseline["rmse"].dropna(), bins=15, color="darkorange", alpha=0.5,
                 edgecolor="white", label="Baseline")
    axes[1].hist(corrected["rmse"].dropna(), bins=15, color="seagreen", alpha=0.5,
                 edgecolor="white", label="XGB-Corrected")
    axes[1].set_xlabel("RMSE (mol/cm^2)")
    axes[1].set_ylabel("Number of Days")
    axes[1].set_title("Distribution of RMSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Aurora vs TEMPO NO2 — Metric Distributions, Baseline vs XGB-Corrected", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "histogram_bias_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved bias/RMSE comparison plots to {save_dir}")

# Main execution block: orchestrates the entire workflow, from loading the split csvs to training the model and generating visualizations
if __name__ == "__main__":
    LON_MIN, LON_MAX = -125.45, -101.98
    LAT_MIN, LAT_MAX = 31.31, 49.01

    # Define paths to training and testing sets produced by the day-based
    # splitter (one lead time's residual_train_t(1,3,5).csv / residual_test_t(1,3,5).csv)
    train_path = Path(f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_train_{HORIZON_LABEL}.csv")
    test_path = Path(f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_test_{HORIZON_LABEL}.csv")

    # Feature columns: raw Aurora prediction, spatial coords, and both
    # initialization-time CAMS timesteps (00Z = t-1, 12Z = t) per variable.
    feature_cols = build_feature_cols()

    # Training and testing block
    train_df, test_df = pull_train_test(train_path, test_path)

    model, results, test_df = train_residual_model(
        train_df, test_df, feature_cols=feature_cols, model_path=output_dir / "xgb_residual_model.joblib")
 
    RESIDUAL_RESULTS_PATH = misc_dir / f"residual_results_{HORIZON_LABEL}.csv"
    test_df[[DATE_COL, "latitude", "longitude", TEMPO_COL, AURORA_COL, "corrected_no2"]].rename(
        columns={AURORA_COL: "aurora_baseline_no2", "corrected_no2": "residual_corrected_no2"}).to_csv(RESIDUAL_RESULTS_PATH, index=False)
    print(f"Saved per-pixel residual results to {RESIDUAL_RESULTS_PATH}")

    # Visualizations
    plot_feature_importance(model, feature_cols, save_path=metrics_dir / "feature_importance.png")

    # Spatial, bounding-box visualizations
    plot_spatial_comparison_maps(test_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, save_dir=maps_dir)

    # Bias / RMSE distribution visualizations
    daily_metrics_df = compute_daily_metrics(test_df)

    daily_metrics_df.to_csv(misc_dir / "daily_metrics_comparison.csv", index=False)
    plot_bias_rmse_comparison(daily_metrics_df, save_dir=metrics_dir)

    pd.DataFrame([results]).to_csv(misc_dir / "xgb_test_results.csv", index=False)
    print("Done.")