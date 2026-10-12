"""Train and evaluate individual machine-learning models with 20 selected features."""
# ================================================================
# single_ml_model_train_20_features.py

# ================================================================

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from traditional_ml_model import (
    get_model,
    get_model_names,
    count_model_parameters,
    get_model_size_mb,
    get_best_params_readable
)


warnings.filterwarnings("ignore")


# ================================================================
# Configuration
# ================================================================

DATA_ROOT = Path(r"G:\PhyMaize")

TRAIN_VAL_ROOT = DATA_ROOT / "features" / "feature_selection_results"
TEST_FILE = DATA_ROOT / "features" / "features-Pn (test).xlsx"
TEST_SPLIT_FILE = DATA_ROOT / "features" / "splits" / "test.csv"

ID_COL = "Sample_ID"
TARGET = "Pn"

RANDOM_STATE = 42
CV_FOLDS = 5

SCORING = "neg_root_mean_squared_error"

N_JOBS = -1

MODELS_TO_RUN = [
    "MLR",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVR",
    "BPNN",
    "RF",
    "ExtraTrees",
    "GBR",
    "HistGBR",
    "XGB"
]

SAVE_MODEL = True
SAVE_PREDICTIONS = True
SAVE_CV_RESULTS = True
SAVE_PLOTS = True


# ================================================================
# Data
# ================================================================

def load_train_val_data(path):
    """Retrieve or load train val data for subsequent analysis."""
    df = pd.read_excel(path)
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in {path}")
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in {path}")
    if df[ID_COL].duplicated().any():
        duplicate_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate Sample_ID detected: {duplicate_ids[:10]}")
    feature_cols = [
        c for c in df.columns
        if c not in [ID_COL, TARGET]
    ]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found.")
    df = df[[ID_COL] + feature_cols + [TARGET]].copy()
    df = df.dropna().reset_index(drop=True)
    return df, feature_cols


def load_test_data(path, feature_cols):
    """Retrieve or load test data for subsequent analysis."""
    df = pd.read_excel(path)
    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in {path}")
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in {path}")
    missing_features = [
        feature
        for feature in feature_cols
        if feature not in df.columns
    ]
    if missing_features:
        raise ValueError(
            f"Test data missing selected features: {missing_features}"
        )
    if df[ID_COL].duplicated().any():
        duplicate_ids = df.loc[df[ID_COL].duplicated(), ID_COL].tolist()
        raise ValueError(f"Duplicate Sample_ID detected in test: {duplicate_ids[:10]}")
    return df[[ID_COL] + feature_cols + [TARGET]].copy()


def load_test_ids(path):
    """Retrieve or load test ids for subsequent analysis."""
    split_df = pd.read_csv(path)
    if ID_COL not in split_df.columns:
        raise ValueError(f"{ID_COL} not found in {path}")
    test_ids = split_df[ID_COL].astype(str).tolist()
    if len(test_ids) == 0:
        raise ValueError("Test split contains no Sample_ID.")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate Sample_ID detected in test split.")
    return test_ids


def prepare_data(train_val_file):
    """Perform prepare data."""
    train_val_df, feature_cols = load_train_val_data(train_val_file)
    test_df = load_test_data(TEST_FILE, feature_cols)
    test_ids = load_test_ids(TEST_SPLIT_FILE)
    train_val_ids = set(train_val_df[ID_COL].astype(str))
    test_ids_set = set(test_ids)
    overlap = train_val_ids.intersection(test_ids_set)
    if overlap:
        raise ValueError(
            f"Train/Val and Test overlap detected: {list(overlap)[:10]}"
        )
    test_df[ID_COL] = test_df[ID_COL].astype(str)
    train_val_df[ID_COL] = train_val_df[ID_COL].astype(str)
    test_df = test_df[
        test_df[ID_COL].isin(test_ids_set)
    ].copy()
    if len(test_df) != len(test_ids):
        missing_ids = test_ids_set - set(test_df[ID_COL])
        raise ValueError(
            f"Some test Sample_IDs were not found in test feature file: "
            f"{list(missing_ids)[:10]}"
        )
    test_order = {sample_id: i for i, sample_id in enumerate(test_ids)}
    test_df["_split_order"] = test_df[ID_COL].map(test_order)
    test_df = (
        test_df
        .sort_values("_split_order")
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )
    X_train = train_val_df[feature_cols].copy()
    y_train = train_val_df[TARGET].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET].copy()
    return (
        train_val_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols
    )


# ================================================================
# Metrics
# ================================================================

def calculate_metrics(y_true, y_pred):
    """Compute metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_y = np.mean(np.abs(y_true))
    if mean_y == 0:
        rrmse = np.nan
    else:
        rrmse = rmse / mean_y * 100
    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "rRMSE": float(rrmse),
        "MAE": float(mae)
    }


# ================================================================
# Plot
# ================================================================

def save_prediction_plot(
    y_train,
    pred_train,
    y_test,
    pred_test,
    model_name,
    output_dir
):
    """Generate or save prediction plot."""
    plt.figure(figsize=(7, 7))
    plt.scatter(
        y_train,
        pred_train,
        alpha=0.65,
        label="Train"
    )
    plt.scatter(
        y_test,
        pred_test,
        alpha=0.8,
        label="Test"
    )
    all_values = np.concatenate([
        np.asarray(y_train),
        np.asarray(pred_train),
        np.asarray(y_test),
        np.asarray(pred_test)
    ])
    min_value = all_values.min()
    max_value = all_values.max()
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )
    plt.xlabel("Observed Pn")
    plt.ylabel("Predicted Pn")
    plt.title(f"{model_name} Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "prediction_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_residual_plot(
    y_test,
    pred_test,
    model_name,
    output_dir
):
    """Generate or save residual plot."""
    residuals = np.asarray(y_test) - np.asarray(pred_test)
    plt.figure(figsize=(7, 5))
    plt.scatter(
        pred_test,
        residuals,
        alpha=0.8
    )
    plt.axhline(
        0,
        linestyle="--"
    )
    plt.xlabel("Predicted Pn")
    plt.ylabel("Residual")
    plt.title(f"{model_name} Test Residuals")
    plt.tight_layout()
    plt.savefig(
        output_dir / "test_residuals.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_cv_plot(cv_results, model_name, output_dir):
    """Generate or save cv plot."""
    results = pd.DataFrame(cv_results)
    if "mean_test_score" not in results.columns:
        return
    scores = results["mean_test_score"].to_numpy()
    scores = -scores
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(scores) + 1),
        scores,
        marker="o",
        markersize=3
    )
    plt.xlabel("Grid Search Candidate")
    plt.ylabel("5-Fold CV RMSE")
    plt.title(f"{model_name} Grid Search")
    plt.tight_layout()
    plt.savefig(
        output_dir / "grid_search_cv.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ================================================================
# Result Saving
# ================================================================

def save_predictions(
    train_ids,
    test_ids,
    y_train,
    pred_train,
    y_test,
    pred_test,
    output_dir
):
    """Generate or save predictions."""
    train_df = pd.DataFrame({
        ID_COL: train_ids,
        "Observed_Pn": y_train,
        "Predicted_Pn": pred_train,
        "Residual": np.asarray(y_train) - np.asarray(pred_train)
    })
    test_df = pd.DataFrame({
        ID_COL: test_ids,
        "Observed_Pn": y_test,
        "Predicted_Pn": pred_test,
        "Residual": np.asarray(y_test) - np.asarray(pred_test)
    })
    with pd.ExcelWriter(
        output_dir / "predictions.xlsx",
        engine="openpyxl"
    ) as writer:
        train_df.to_excel(
            writer,
            sheet_name="Train",
            index=False
        )
        test_df.to_excel(
            writer,
            sheet_name="Test",
            index=False
        )


def save_best_params(best_params, output_dir):
    """Generate or save best params."""
    readable_params = get_best_params_readable(best_params)
    with open(
        output_dir / "best_params.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            readable_params,
            f,
            ensure_ascii=False,
            indent=4,
            default=str
        )


# ================================================================
# Single Model Training
# ================================================================

def train_single_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    train_ids,
    test_ids,
    output_root
):
    """Perform train single model."""
    output_dir = output_root / model_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    print()
    print("=" * 72)
    print(f"Model: {model_name}")
    print("=" * 72)
    model, param_grid = get_model(model_name)
    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=True,
        verbose=1
    )
    start_time = time.perf_counter()
    grid_search.fit(
        X_train,
        y_train
    )
    training_time = time.perf_counter() - start_time
    best_model = grid_search.best_estimator_
    pred_train = best_model.predict(X_train)
    pred_test = best_model.predict(X_test)
    train_metrics = calculate_metrics(
        y_train,
        pred_train
    )
    test_metrics = calculate_metrics(
        y_test,
        pred_test
    )
    parameter_count = count_model_parameters(best_model)
    model_size = get_model_size_mb(best_model)
    result = {
        "Model": model_name,
        "Train_R2": train_metrics["R2"],
        "Train_RMSE": train_metrics["RMSE"],
        "Train_rRMSE": train_metrics["rRMSE"],
        "Train_MAE": train_metrics["MAE"],
        "Test_R2": test_metrics["R2"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_rRMSE": test_metrics["rRMSE"],
        "Test_MAE": test_metrics["MAE"],
        "CV_Best_RMSE": -grid_search.best_score_,
        "Training_Time_s": training_time,
        "Model_Size_MB": model_size,
        "Parameter_Count": parameter_count,
        "Feature_Count": X_train.shape[1],
        "Train_Samples": X_train.shape[0],
        "Test_Samples": X_test.shape[0],
        "Best_Params": str(
            get_best_params_readable(
                grid_search.best_params_
            )
        )
    }
    result_df = pd.DataFrame([result])
    result_df.to_csv(
        output_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )
    result_df.to_excel(
        output_dir / "metrics.xlsx",
        index=False
    )
    save_best_params(
        grid_search.best_params_,
        output_dir
    )
    if SAVE_MODEL:
        joblib.dump(
            best_model,
            output_dir / "best_model.joblib"
        )
    if SAVE_PREDICTIONS:
        save_predictions(
            train_ids,
            test_ids,
            y_train,
            pred_train,
            y_test,
            pred_test,
            output_dir
        )
    if SAVE_PLOTS:
        save_prediction_plot(
            y_train,
            pred_train,
            y_test,
            pred_test,
            model_name,
            output_dir
        )
        save_residual_plot(
            y_test,
            pred_test,
            model_name,
            output_dir
        )
        if SAVE_CV_RESULTS:
            save_cv_plot(
                grid_search.cv_results_,
                model_name,
                output_dir
            )
    if SAVE_CV_RESULTS:
        cv_results = pd.DataFrame(
            grid_search.cv_results_
        )
        cv_results.to_csv(
            output_dir / "grid_search_results.csv",
            index=False,
            encoding="utf-8-sig"
        )
        cv_results.to_excel(
            output_dir / "grid_search_results.xlsx",
            index=False
        )
    print()
    print(f"Best parameters: {get_best_params_readable(grid_search.best_params_)}")
    print(f"Train R²       : {train_metrics['R2']:.4f}")
    print(f"Train RMSE     : {train_metrics['RMSE']:.4f}")
    print(f"Train rRMSE    : {train_metrics['rRMSE']:.2f}%")
    print(f"Train MAE      : {train_metrics['MAE']:.4f}")
    print(f"Test R²        : {test_metrics['R2']:.4f}")
    print(f"Test RMSE      : {test_metrics['RMSE']:.4f}")
    print(f"Test rRMSE     : {test_metrics['rRMSE']:.2f}%")
    print(f"Test MAE       : {test_metrics['MAE']:.4f}")
    print(f"CV RMSE        : {-grid_search.best_score_:.4f}")
    print(f"Training time  : {training_time:.2f} s")
    print(f"Model size     : {model_size:.4f} MB")
    print(f"Parameter count: {parameter_count}")
    return result


# ================================================================
# Comparison Results
# ================================================================

def save_comparison(results, output_root):
    """Generate or save comparison."""
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(
        "Test_R2",
        ascending=False
    ).reset_index(drop=True)
    comparison_df.to_excel(
        output_root / "model_comparison.xlsx",
        index=False
    )
    comparison_df.to_csv(
        output_root / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )
    return comparison_df


def save_comparison_plot(comparison_df, output_root):
    """Generate or save comparison plot."""
    plt.figure(figsize=(10, 6))
    plt.bar(
        comparison_df["Model"],
        comparison_df["Test_R2"]
    )
    plt.xlabel("Model")
    plt.ylabel("Test R²")
    plt.title("Traditional ML Model Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        output_root / "model_comparison_R2.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def save_metric_comparison_plot(comparison_df, output_root):
    """Generate or save metric comparison plot."""
    x = np.arange(len(comparison_df))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2,
        comparison_df["Test_RMSE"],
        width,
        label="RMSE"
    )
    plt.bar(
        x + width / 2,
        comparison_df["Test_MAE"],
        width,
        label="MAE"
    )
    plt.xticks(
        x,
        comparison_df["Model"],
        rotation=30
    )
    plt.ylabel("Error")
    plt.title("Traditional ML Error Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_root / "model_comparison_error.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ================================================================
# Main
# ================================================================

def main():
    """Run the main analysis workflow."""
    print("=" * 72)
    print("Traditional ML Pn Prediction")
    print("=" * 72)
    print(f"Train/Val root  : {TRAIN_VAL_ROOT}")
    print(f"Test file       : {TEST_FILE}")
    print(f"Test split      : {TEST_SPLIT_FILE}")
    print(f"Feature range   : 3-25")
    print(f"CV folds        : {CV_FOLDS}")
    print(f"Random state    : {RANDOM_STATE}")
    print()
    for feature_num in range(20, 21):
        feature_dir = TRAIN_VAL_ROOT / f"{feature_num}_features"
        train_val_file = feature_dir / "selected_features.xlsx"
        output_root = DATA_ROOT / "traditional_ml_results" / f"{feature_num}_features"
        output_root.mkdir(
            parents=True,
            exist_ok=True
        )
        print()
        print("#" * 72)
        print(f"# Processing {feature_num}_features")
        print("#" * 72)
        print(f"Train/Val file : {train_val_file}")
        print(f"Output root    : {output_root}")
        if not train_val_file.exists():
            print(f"[ERROR] File not found: {train_val_file}")
            continue
        try:
            (
                train_val_df,
                test_df,
                X_train,
                y_train,
                X_test,
                y_test,
                feature_cols
            ) = prepare_data(train_val_file)
            print(f"Train/Val samples: {len(X_train)}")
            print(f"Test samples     : {len(X_test)}")
            print(f"Feature count    : {len(feature_cols)}")
            print(f"Features         : {feature_cols}")
            print()
            print("Models to run:")
            for model_name in MODELS_TO_RUN:
                print(f"  - {model_name}")
            print()
            train_ids = train_val_df[ID_COL].tolist()
            test_ids = test_df[ID_COL].tolist()
            results = []
            progress = tqdm(
                MODELS_TO_RUN,
                desc=f"{feature_num}_features",
                unit="model"
            )
            for model_name in progress:
                progress.set_postfix(
                    model=model_name
                )
                try:
                    result = train_single_model(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        train_ids=train_ids,
                        test_ids=test_ids,
                        output_root=output_root
                    )
                    results.append(result)
                except Exception as e:
                    print()
                    print(f"[ERROR] {model_name}: {e}")
                    error_df = pd.DataFrame([{
                        "Model": model_name,
                        "Status": "FAILED",
                        "Error": str(e)
                    }])
                    error_df.to_csv(
                        output_root / f"{model_name}_error.csv",
                        index=False,
                        encoding="utf-8-sig"
                    )
            if not results:
                print(f"[ERROR] No model was successfully trained for {feature_num}_features.")
                continue
            comparison_df = save_comparison(
                results,
                output_root
            )
            save_comparison_plot(
                comparison_df,
                output_root
            )
            save_metric_comparison_plot(
                comparison_df,
                output_root
            )
            print()
            print("=" * 72)
            print(f"Final Model Comparison - {feature_num}_features")
            print("=" * 72)
            display_cols = [
                "Model",
                "Train_R2",
                "Train_RMSE",
                "Train_rRMSE",
                "Train_MAE",
                "Test_R2",
                "Test_RMSE",
                "Test_rRMSE",
                "Test_MAE",
                "CV_Best_RMSE",
                "Training_Time_s",
                "Model_Size_MB",
                "Parameter_Count"
            ]
            print(
                comparison_df[display_cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}"
                )
            )
            print()
            print(f"Results saved to:")
            print(output_root)
        except Exception as e:
            print()
            print(f"[ERROR] {feature_num}_features: {e}")
    print()
    print("=" * 72)
    print("All feature-number experiments completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()