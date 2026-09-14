from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_DIR = Path(
    "/umbc/rs/cybertrn/reu2026/team1/research/data/"
    "rcm_data/csv/time_series/splits"
)

OUTPUT_DIR = Path(
    "/umbc/rs/cybertrn/reu2026/team1/research/testing/"
    "Aryan_testing/correction_model/oscar_results_no_tempo_aurora_inputs"
)

LEAD_NAMES = {
    "t1": "OneDay",
    "t3": "TwoDay",
    "t5": "ThreeDay",
}

AURORA = "aurora_pred_no2_molecules_cm2"
TEMPO = "tempo_no2_molecules_cm2"
RESIDUAL = "residual_molecules_cm2"


CAMS_FEATURES = [
    "cams_u10_00z",
    "cams_v10_00z",
    "cams_t2m_00z",
    "cams_msl_00z",
    "cams_pm1_00z",
    "cams_pm2p5_00z",
    "cams_pm10_00z",
    "cams_tcco_00z",
    "cams_tc_no_00z",
    "cams_tcno2_molecules_cm2_00z",
    "cams_gtco3_00z",
    "cams_tcso2_00z",

    "cams_u10_12z",
    "cams_v10_12z",
    "cams_t2m_12z",
    "cams_msl_12z",
    "cams_pm1_12z",
    "cams_pm2p5_12z",
    "cams_pm10_12z",
    "cams_tcco_12z",
    "cams_tc_no_12z",
    "cams_tcno2_molecules_cm2_12z",
    "cams_gtco3_12z",
    "cams_tcso2_12z",
]


COMMON_FEATURES = [
    "latitude",
    "longitude",
    *CAMS_FEATURES,
    "month_target",
    "day_target",
    "dayofyear_target",
]


OUTPUT_METADATA = [
    "target_date",
    "init_date",
    "lead_label",
    "latitude",
    "longitude",
    AURORA,
    TEMPO,
    RESIDUAL,
]


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """
    Bias is defined as TEMPO minus prediction.

    Positive bias means the prediction is too low.
    Negative bias means the prediction is too high.
    """

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    truth,
                    prediction,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                truth,
                prediction,
            )
        ),
        "bias_tempo_minus_prediction": float(
            np.mean(truth - prediction)
        ),
        "r2": float(
            r2_score(
                truth,
                prediction,
            )
        ),
    }


def detect_residual_definition(
    dataframe: pd.DataFrame,
) -> tuple[str, float, float]:
    """
    Determine whether Oscar's residual is:

        TEMPO - Aurora

    or:

        Aurora - TEMPO
    """

    tempo = dataframe[TEMPO].to_numpy(
        dtype=np.float64
    )

    aurora = dataframe[AURORA].to_numpy(
        dtype=np.float64
    )

    residual = dataframe[RESIDUAL].to_numpy(
        dtype=np.float64
    )

    tempo_minus_aurora_error = float(
        np.mean(
            np.abs(
                residual - (tempo - aurora)
            )
        )
    )

    aurora_minus_tempo_error = float(
        np.mean(
            np.abs(
                residual - (aurora - tempo)
            )
        )
    )

    residual_scale = max(
        float(np.mean(np.abs(residual))),
        1.0,
    )

    best_relative_error = (
        min(
            tempo_minus_aurora_error,
            aurora_minus_tempo_error,
        )
        / residual_scale
    )

    if best_relative_error > 1.0e-4:
        raise ValueError(
            "Could not verify the residual definition. "
            f"Relative mismatch: {best_relative_error:.3e}"
        )

    if (
        tempo_minus_aurora_error
        <= aurora_minus_tempo_error
    ):
        definition = "tempo_minus_aurora"
    else:
        definition = "aurora_minus_tempo"

    return (
        definition,
        tempo_minus_aurora_error,
        aurora_minus_tempo_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest models using Oscar's "
            "official train and test splits."
        )
    )

    parser.add_argument(
        "lead",
        choices=["t1", "t3", "t5"],
        help=(
            "t1 = OneDay, "
            "t3 = TwoDay, "
            "t5 = ThreeDay"
        ),
    )

    parser.add_argument(
        "mode",
        choices=["residual", "direct"],
        help=(
            "residual predicts Aurora error; "
            "direct predicts TEMPO NO2 directly"
        ),
    )

    args = parser.parse_args()

    lead = args.lead
    mode = args.mode
    horizon = LEAD_NAMES[lead]

    train_file = DATA_DIR / (
        f"residual_train_{lead}.csv"
    )

    test_file = DATA_DIR / (
        f"residual_test_{lead}.csv"
    )

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    if not test_file.exists():
        raise FileNotFoundError(test_file)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_prefix = OUTPUT_DIR / (
        f"rf_official_{lead}_{mode}"
    )

    print("=" * 80)
    print(f"Horizon: {horizon} ({lead})")
    print(f"Mode: {mode}")
    print(f"Training file: {train_file}")
    print(f"Testing file:  {test_file}")
    print("=" * 80)

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    print(
        f"Training rows loaded: "
        f"{len(train):,}"
    )

    print(
        f"Testing rows loaded: "
        f"{len(test):,}"
    )

    print(
        f"Columns loaded: "
        f"{len(train.columns)}"
    )

    required_columns = set(
        COMMON_FEATURES
        + [
            AURORA,
            TEMPO,
            RESIDUAL,
            "target_date",
            "init_date",
            "lead_label",
        ]
    )

    missing_train = sorted(
        required_columns
        - set(train.columns)
    )

    missing_test = sorted(
        required_columns
        - set(test.columns)
    )

    if missing_train:
        raise KeyError(
            f"Missing training columns: "
            f"{missing_train}"
        )

    if missing_test:
        raise KeyError(
            f"Missing testing columns: "
            f"{missing_test}"
        )

    train_dates = pd.to_datetime(
        train["target_date"],
        errors="raise",
    )

    test_dates = pd.to_datetime(
        test["target_date"],
        errors="raise",
    )

    train_unique_dates = set(
        train_dates.dt.date
    )

    test_unique_dates = set(
        test_dates.dt.date
    )

    overlapping_dates = (
        train_unique_dates
        & test_unique_dates
    )

    if overlapping_dates:
        raise ValueError(
            "Training and testing contain overlapping "
            f"target dates: "
            f"{sorted(overlapping_dates)[:5]}"
        )

    print("\nOfficial date split")

    print(
        "Training dates:",
        train_dates.min().date(),
        "to",
        train_dates.max().date(),
    )

    print(
        "Training unique dates:",
        train_dates.dt.date.nunique(),
    )

    print(
        "Testing dates:",
        test_dates.min().date(),
        "to",
        test_dates.max().date(),
    )

    print(
        "Testing unique dates:",
        test_dates.dt.date.nunique(),
    )

    (
        residual_definition,
        tempo_minus_aurora_error,
        aurora_minus_tempo_error,
    ) = detect_residual_definition(test)

    print("\nResidual verification")

    print(
        "Mismatch with TEMPO - Aurora:",
        f"{tempo_minus_aurora_error:.6e}",
    )

    print(
        "Mismatch with Aurora - TEMPO:",
        f"{aurora_minus_tempo_error:.6e}",
    )

    print(
        "Detected residual definition:",
        residual_definition,
    )

    # Both models use only CAMS, location, and calendar inputs.
    # TEMPO, Aurora, and the residual are excluded from X.
    if mode == "residual":
        features = COMMON_FEATURES
        target_column = RESIDUAL

    else:
        features = COMMON_FEATURES
        target_column = TEMPO

    train_required = (
        features
        + [target_column]
    )

    test_required = (
        features
        + [
            target_column,
            AURORA,
            TEMPO,
            RESIDUAL,
        ]
    )

    train_before = len(train)
    test_before = len(test)

    train = train.dropna(
        subset=train_required
    ).copy()

    test = test.dropna(
        subset=test_required
    ).copy()

    print("\nRows after dropping missing values")

    print(
        f"Training rows: {len(train):,} "
        f"(removed "
        f"{train_before - len(train):,})"
    )

    print(
        f"Testing rows: {len(test):,} "
        f"(removed "
        f"{test_before - len(test):,})"
    )

    X_train = train[features].astype(
        np.float32
    )

    X_test = test[features].astype(
        np.float32
    )

    y_train = train[target_column].astype(
        np.float64
    )

    print(
        f"\nFeatures used: {len(features)}"
    )

    for feature in features:
        print("-", feature)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print(
        "\nTraining Random Forest using the "
        "complete official training split..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nPredicting the official test split..."
    )

    model_output = model.predict(
        X_test
    )

    tempo_truth = test[TEMPO].to_numpy(
        dtype=np.float64
    )

    aurora_prediction = test[
        AURORA
    ].to_numpy(
        dtype=np.float64
    )

    raw_aurora_metrics = metrics(
        tempo_truth,
        aurora_prediction,
    )

    predictions = test[
        OUTPUT_METADATA
    ].copy()

    results = {
        "lead": lead,
        "horizon": horizon,
        "mode": mode,
        "train_file": train_file.name,
        "test_file": test_file.name,
        "training_rows": len(train),
        "testing_rows": len(test),
        "training_unique_dates": (
            train_dates.dt.date.nunique()
        ),
        "testing_unique_dates": (
            test_dates.dt.date.nunique()
        ),
        "feature_count": len(features),
        "residual_definition": (
            residual_definition
        ),
        "aurora_rmse": (
            raw_aurora_metrics["rmse"]
        ),
        "aurora_mae": (
            raw_aurora_metrics["mae"]
        ),
        "aurora_bias_tempo_minus_prediction": (
            raw_aurora_metrics[
                "bias_tempo_minus_prediction"
            ]
        ),
        "aurora_r2": (
            raw_aurora_metrics["r2"]
        ),
    }

    if mode == "residual":
        predicted_residual = model_output

        if (
            residual_definition
            == "tempo_minus_aurora"
        ):
            corrected_prediction = (
                aurora_prediction
                + predicted_residual
            )
        else:
            corrected_prediction = (
                aurora_prediction
                - predicted_residual
            )

        corrected_metrics = metrics(
            tempo_truth,
            corrected_prediction,
        )

        residual_r2 = float(
            r2_score(
                test[RESIDUAL].to_numpy(
                    dtype=np.float64
                ),
                predicted_residual,
            )
        )

        predictions[
            "predicted_residual_molecules_cm2"
        ] = predicted_residual

        predictions[
            "corrected_no2_molecules_cm2"
        ] = corrected_prediction

        predictions[
            "aurora_error_tempo_minus_prediction"
        ] = (
            tempo_truth
            - aurora_prediction
        )

        predictions[
            "corrected_error_tempo_minus_prediction"
        ] = (
            tempo_truth
            - corrected_prediction
        )

        results.update(
            {
                "corrected_rmse": (
                    corrected_metrics["rmse"]
                ),
                "corrected_mae": (
                    corrected_metrics["mae"]
                ),
                "corrected_bias_tempo_minus_prediction": (
                    corrected_metrics[
                        "bias_tempo_minus_prediction"
                    ]
                ),
                "corrected_r2": (
                    corrected_metrics["r2"]
                ),
                "residual_r2": residual_r2,
                "rmse_improvement": (
                    raw_aurora_metrics["rmse"]
                    - corrected_metrics["rmse"]
                ),
                "mae_improvement": (
                    raw_aurora_metrics["mae"]
                    - corrected_metrics["mae"]
                ),
                "rmse_improvement_percent": (
                    100.0
                    * (
                        raw_aurora_metrics["rmse"]
                        - corrected_metrics["rmse"]
                    )
                    / raw_aurora_metrics["rmse"]
                ),
                "mae_improvement_percent": (
                    100.0
                    * (
                        raw_aurora_metrics["mae"]
                        - corrected_metrics["mae"]
                    )
                    / raw_aurora_metrics["mae"]
                ),
            }
        )

        print("\n" + "=" * 80)
        print(
            f"{horizon} OFFICIAL RESIDUAL RESULTS"
        )
        print("=" * 80)

        print("\nRaw Aurora versus TEMPO")

        print(
            "RMSE:",
            f"{raw_aurora_metrics['rmse']:.6e}",
        )

        print(
            "MAE:",
            f"{raw_aurora_metrics['mae']:.6e}",
        )

        print(
            "Bias (TEMPO - Aurora):",
            f"{raw_aurora_metrics['bias_tempo_minus_prediction']:.6e}",
        )

        print(
            "\nCorrected forecast versus TEMPO"
        )

        print(
            "RMSE:",
            f"{corrected_metrics['rmse']:.6e}",
        )

        print(
            "MAE:",
            f"{corrected_metrics['mae']:.6e}",
        )

        print(
            "Bias (TEMPO - corrected):",
            f"{corrected_metrics['bias_tempo_minus_prediction']:.6e}",
        )

        print(
            "Residual R2:",
            f"{residual_r2:.6f}",
        )

    else:
        direct_prediction = model_output

        direct_metrics = metrics(
            tempo_truth,
            direct_prediction,
        )

        predictions[
            "direct_predicted_tempo_no2_molecules_cm2"
        ] = direct_prediction

        predictions[
            "aurora_error_tempo_minus_prediction"
        ] = (
            tempo_truth
            - aurora_prediction
        )

        predictions[
            "direct_error_tempo_minus_prediction"
        ] = (
            tempo_truth
            - direct_prediction
        )

        results.update(
            {
                "direct_rmse": (
                    direct_metrics["rmse"]
                ),
                "direct_mae": (
                    direct_metrics["mae"]
                ),
                "direct_bias_tempo_minus_prediction": (
                    direct_metrics[
                        "bias_tempo_minus_prediction"
                    ]
                ),
                "direct_r2": (
                    direct_metrics["r2"]
                ),
                "rmse_improvement_vs_aurora": (
                    raw_aurora_metrics["rmse"]
                    - direct_metrics["rmse"]
                ),
                "mae_improvement_vs_aurora": (
                    raw_aurora_metrics["mae"]
                    - direct_metrics["mae"]
                ),
                "rmse_improvement_percent_vs_aurora": (
                    100.0
                    * (
                        raw_aurora_metrics["rmse"]
                        - direct_metrics["rmse"]
                    )
                    / raw_aurora_metrics["rmse"]
                ),
                "mae_improvement_percent_vs_aurora": (
                    100.0
                    * (
                        raw_aurora_metrics["mae"]
                        - direct_metrics["mae"]
                    )
                    / raw_aurora_metrics["mae"]
                ),
            }
        )

        print("\n" + "=" * 80)
        print(
            f"{horizon} OFFICIAL DIRECT RESULTS"
        )
        print("=" * 80)

        print("\nRaw Aurora versus TEMPO")

        print(
            "RMSE:",
            f"{raw_aurora_metrics['rmse']:.6e}",
        )

        print(
            "MAE:",
            f"{raw_aurora_metrics['mae']:.6e}",
        )

        print(
            "\nDirect prediction versus TEMPO"
        )

        print(
            "RMSE:",
            f"{direct_metrics['rmse']:.6e}",
        )

        print(
            "MAE:",
            f"{direct_metrics['mae']:.6e}",
        )

        print(
            "Bias (TEMPO - direct):",
            f"{direct_metrics['bias_tempo_minus_prediction']:.6e}",
        )

        print(
            "Direct R2:",
            f"{direct_metrics['r2']:.6f}",
        )

    predictions_file = Path(
        f"{output_prefix}_predictions.csv"
    )

    importance_file = Path(
        f"{output_prefix}_feature_importances.csv"
    )

    metrics_file = Path(
        f"{output_prefix}_metrics.csv"
    )

    predictions.to_csv(
        predictions_file,
        index=False,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        importance_file,
        index=False,
    )

    pd.DataFrame(
        [results]
    ).to_csv(
        metrics_file,
        index=False,
    )

    print("\nTop 20 features")

    print(
        feature_importance.head(20).to_string(
            index=False
        )
    )

    print("\nSaved files")

    print(predictions_file)
    print(importance_file)
    print(metrics_file)


if __name__ == "__main__":
    main()
