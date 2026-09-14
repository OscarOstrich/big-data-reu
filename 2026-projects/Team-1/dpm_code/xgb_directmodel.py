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


HORIZON_LABEL = "t5"

output_dir = Path("/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/Plots")

maps_dir = output_dir / f"xgb_dir_{HORIZON_LABEL}" / f"maps_xgb_{HORIZON_LABEL}"       # spatial, bounding-box visualizations
metrics_dir = output_dir / f"xgb_dir_{HORIZON_LABEL}" / f"metrics_xgb_{HORIZON_LABEL}"
misc_dir = output_dir / f"xgb_dir_{HORIZON_LABEL}" / f"misc_{HORIZON_LABEL}"           # csv of all computed values
misc_dir.mkdir(parents=True, exist_ok=True)
maps_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

# Per-pixel results saved by the residual/XGB-corrected script
RESIDUAL_RESULTS_PATH = output_dir / f"xgb_ts_{HORIZON_LABEL}" / f"misc_{HORIZON_LABEL}" / f"residual_results_{HORIZON_LABEL}.csv"

# Column names in the per-lead train/test CSVs (residual_train.csv /
# residual_test.csv produced by the day-based splitter).
DATE_COL = "target_date"
TEMPO_COL = "tempo_no2_molecules_cm2"
AURORA_COL = "aurora_pred_no2_molecules_cm2"

CAMS_SURFACE_VARS = [
    "u10", "v10", "t2m", "msl", "pm1", "pm2p5", "pm10",
    "tcco", "tc_no", "tcno2", "gtco3", "tcso2",
]


def cams_col_base(var):
    return "cams_tcno2_molecules_cm2" if var == "tcno2" else f"cams_{var}"


def build_feature_cols(extra_cols=("latitude", "longitude")):
    # Builds the feature list: spatial coords plus both initialization-time
    # CAMS timesteps (00Z = t-1, 12Z = t) per variable. No Aurora or
    # residual columns are used and this predicts TEMPO NO2 directly from CAMS inputs alone.
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

# train_direct_model: trains an XGBoost regressor to predict TEMPO NO2 directly from CAMS features, evaluates its performance, and augments the test DataFrame with predictions
def train_direct_model(train_df, test_df,
                        feature_cols=None,
                        target_col=TEMPO_COL,
                        model_path=None):
    if feature_cols is None:
        feature_cols = build_feature_cols()
    feature_cols = list(feature_cols)

    # Drop rows missing any required feature/target before fitting, since a missing CAMS lag file upstream would otherwise feed NaNs into XGBoost.
    required_cols = feature_cols + [target_col]
    train_df = train_df.dropna(subset=required_cols).copy()
    test_df = test_df.dropna(subset=required_cols).copy()

    # Split each dataframe into the feature matrix (X) and the target (TEMPO NO2)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Train an XGBoost regressor to predict TEMPO NO2 directly and edit hyperparammeters here
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

    # Augment test_df with predictions so downstream plotting has both
    # the TEMPO truth and the model's direct prediction.
    test_df = test_df.copy()
    test_df["predicted_no2"] = model.predict(X_test)

    # Compute metrics for the direct prediction vs TEMPO truth
    mae = mean_absolute_error(test_df[TEMPO_COL], test_df["predicted_no2"])
    rmse = np.sqrt(mean_squared_error(test_df[TEMPO_COL], test_df["predicted_no2"]))
    r2 = r2_score(test_df[TEMPO_COL], test_df["predicted_no2"])

    results = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    print("\n--- Test set performance (XGBoost direct prediction vs TEMPO) ---")
    print(f"MAE:  {mae:.4e}   RMSE: {rmse:.4e}   R2: {r2:.4f}")

    if model_path is not None:
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

    return model, results, test_df


# load_and_merge_residual_results: brings in the residual/XGB-corrected model's
# saved per-pixel predictions so the direct model can be compared against
# both Aurora baseline and the corrected model on the same rows.
def load_and_merge_residual_results(test_df, residual_results_path):
    residual_df = pd.read_csv(residual_results_path)
    residual_df[DATE_COL] = pd.to_datetime(residual_df[DATE_COL])

    merged_df = test_df.merge(
        residual_df[[DATE_COL, "latitude", "longitude", "aurora_baseline_no2", "residual_corrected_no2"]],
        on=[DATE_COL, "latitude", "longitude"],
        how="inner",
    )
    print(f"Merged into {len(merged_df)} rows common to both models")
    return merged_df


AURORA_BASELINE_COL = "aurora_baseline_no2"
RESIDUAL_CORRECTED_COL = "residual_corrected_no2"
DIRECT_COL = "predicted_no2"


# Visualizations

# plot_feature_importance: generates a horizontal bar chart of feature importances from the trained XGBoost model
def plot_feature_importance(model, feature_cols, save_path=None):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1], color="steelblue")
    ax.set_xlabel("Feature importance")
    ax.set_title("XGBoost direct NO2 model — feature importance")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# Helper function to pivot the DataFrame into a grid format suitable for spatial plotting
def _pivot_to_grid(day_df, value_col):
    grid = day_df.pivot_table(index="latitude", columns="longitude", values=value_col)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid.columns.values, grid.index.values, grid.values

# plot_model_comparison_maps: same structure as nflows_direct.py's map —
# TEMPO truth, Aurora baseline, XGB-corrected (residual model), XGB-direct
# (this model), all sharing one 0-6e15 scale.
def plot_model_comparison_maps(merged_df, lat_min, lat_max, lon_min, lon_max, save_dir=maps_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # all 4 panels share one scale, 0 to 6 (x10^15 mol/cm^2)
    SHARED_VMIN = 0
    SHARED_VMAX = 6e15

    for date, day_df in merged_df.groupby(DATE_COL):
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        lons, lats, tempo_grid = _pivot_to_grid(day_df, TEMPO_COL)
        _, _, aurora_grid = _pivot_to_grid(day_df, AURORA_BASELINE_COL)
        _, _, corrected_grid = _pivot_to_grid(day_df, RESIDUAL_CORRECTED_COL)
        _, _, direct_grid = _pivot_to_grid(day_df, DIRECT_COL)

        fig, axes = plt.subplots(
            2, 2, figsize=(13, 10),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        panels = [
            (axes[0, 0], tempo_grid, f"TEMPO ground truth, {date_str}"),
            (axes[0, 1], aurora_grid, f"Aurora baseline, {date_str}"),
            (axes[1, 0], corrected_grid, "XGB-corrected Aurora"),
            (axes[1, 1], direct_grid, "XGB direct prediction"),
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

        plt.suptitle(f"Model comparison ({HORIZON_LABEL}), {date_str}", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / f"model_comparison_{date_str}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved comparison maps to {save_dir}")


# compute_daily_metrics: per-day bias/RMSE table across all 3 variants,
def compute_daily_metrics(merged_df):
    daily_rows = []
    for date, day_df in merged_df.groupby(DATE_COL):
        tempo_mean = day_df[TEMPO_COL].mean()

        variants = {
            "Aurora Baseline": day_df[AURORA_BASELINE_COL],
            "XGB-Corrected": day_df[RESIDUAL_CORRECTED_COL],
            "XGB-Direct": day_df[DIRECT_COL],
        }

        for name, values in variants.items():
            diff = values - day_df[TEMPO_COL]
            daily_rows.append({
                "date": date,
                "variant": name,
                "mean_bias_pct": (diff.mean() / tempo_mean * 100) if tempo_mean != 0 else np.nan,
                "rmse": np.sqrt((diff ** 2).mean()),
            })

    return pd.DataFrame(daily_rows)


# plot_three_way_comparison: boxplots and histograms comparing Aurora
# Baseline, XGB-Corrected, and XGB-Direct 
def plot_three_way_comparison(daily_metrics_df, save_dir=metrics_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    variant_order = ["Aurora Baseline", "XGB-Corrected", "XGB-Direct"]
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

    plt.suptitle(f"Aurora Baseline vs XGB-Corrected vs XGB-Direct ({HORIZON_LABEL})", fontsize=13)
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

    print(f"Saved three-way comparison plots to {save_dir}")


# Main execution block
if __name__ == "__main__":
    LON_MIN, LON_MAX = -125.45, -101.98
    LAT_MIN, LAT_MAX = 31.31, 49.01

    train_path = Path(f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_train_{HORIZON_LABEL}.csv")
    test_path = Path(f"/umbc/rs/cybertrn/reu2026/team1/research/data/rcm_data/csv/time_series/splits/residual_test_{HORIZON_LABEL}.csv")

    feature_cols = build_feature_cols()

    # Training and testing block
    train_df, test_df = pull_train_test(train_path, test_path)

    model, results, test_df = train_direct_model(
        train_df, test_df, feature_cols=feature_cols, model_path=output_dir / "xgb_direct_model.joblib")

    # Feature importance is specific to this model, unaffected by the merge
    plot_feature_importance(model, feature_cols, save_path=metrics_dir / "feature_importance.png")

    # Bring in the residual/XGB-corrected model's saved per-pixel results
    merged_df = load_and_merge_residual_results(test_df, RESIDUAL_RESULTS_PATH)

    # Spatial, bounding-box visualizations — 4-panel: TEMPO, Aurora, XGB-corrected, XGB-direct
    plot_model_comparison_maps(merged_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, save_dir=maps_dir)

    # Bias / RMSE distribution visualizations — 3-way: Aurora Baseline, XGB-Corrected, XGB-Direct
    daily_metrics_df = compute_daily_metrics(merged_df)
    daily_metrics_df.to_csv(misc_dir / "daily_metrics_comparison.csv", index=False)
    plot_three_way_comparison(daily_metrics_df, save_dir=metrics_dir)

    pd.DataFrame([results]).to_csv(misc_dir / "xgb_test_results.csv", index=False)
    print("Done.")