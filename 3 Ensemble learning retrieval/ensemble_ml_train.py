"""Train, evaluate, and save ensemble machine-learning models."""
import warnings
warnings.filterwarnings("ignore")
import io
import random
from pathlib import Path
from itertools import combinations
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_ROOT = Path(r"G:\PhyMaize")
TRAIN_VAL_FILE = DATA_ROOT / "features" / "features-Pn (train and val).xlsx"
TEST_FILE = DATA_ROOT / "features" / "features-Pn (test).xlsx"
SPLIT_DIR = DATA_ROOT / "features" / "splits"
SELECTED_FEATURE_FILE = DATA_ROOT / "features" / "feature_selection_results" / "20_features" / "selected_features.xlsx"
MODEL_ROOT = DATA_ROOT / "traditional_ml_results" / "20_features"
OUTPUT_ROOT = MODEL_ROOT / "ensemble_v5"

N_SPLITS = 5
RANDOM_STATE = 42
TARGET = "Pn"
ID_COL = "Sample_ID"
ET_NAME = "ExtraTrees"

RIDGE_ALPHA = 10.0
ELASTIC_ALPHA = 0.01
ELASTIC_L1_RATIO = 0.5

MAX_COMPLEMENTARY_MODELS = 3
MIN_MODEL_R2_FRACTION = 0.80

ALPHA_MIN = 0.0
ALPHA_MAX = 1.0

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    """Perform set seed."""
    random.seed(seed)
    np.random.seed(seed)


def metrics(y_true, y_pred):
    """Perform metrics."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"R2": r2_score(y_true, y_pred), "RMSE": rmse, "MAE": mean_absolute_error(y_true, y_pred), "rRMSE": rmse / np.mean(np.abs(y_true)) * 100}


def serialized_size_mb(obj):
    """Perform serialized size mb."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer, compress=0)
    return buffer.tell() / (1024 ** 2)


def build_ensemble_bundle(name, fitted_models, metadata):
    """Perform build ensemble bundle."""
    if name == ET_NAME:
        return {"Model_Type": "Base", "Strategy": ET_NAME, "Model": fitted_models[ET_NAME]}
    info = metadata[name]
    bundle = {"Model_Type": "ExtraTrees_Anchored_Ensemble", "Strategy": name, "Method": info["Method"], "Selected": tuple(info["Selected"]), "ExtraTrees": fitted_models[ET_NAME]}
    for model_name in info["Selected"]: bundle[model_name] = fitted_models[model_name]
    for key in ["Alpha", "Weights", "Meta_Model"]:
        if key in info: bundle[key] = info[key]
    return bundle


def strategy_size_mb(name, fitted_models, metadata):
    """Perform strategy size mb."""
    return serialized_size_mb(build_ensemble_bundle(name, fitted_models, metadata))


def save_best_ensemble_model(best_strategy, fitted_models, metadata, path):
    """Generate or save best ensemble model."""
    bundle = build_ensemble_bundle(best_strategy, fitted_models, metadata)
    joblib.dump(bundle, path, compress=3)
    return path


def load_selected_features():
    """Retrieve or load selected features for subsequent analysis."""
    df = pd.read_excel(SELECTED_FEATURE_FILE)
    cols = list(df.columns)
    if "Feature" in cols: features = df["Feature"].dropna().astype(str).tolist()
    elif "Selected_Feature" in cols: features = df["Selected_Feature"].dropna().astype(str).tolist()
    elif "Feature_Name" in cols: features = df["Feature_Name"].dropna().astype(str).tolist()
    else: features = [c for c in cols if c not in {ID_COL, TARGET}]
    features = list(dict.fromkeys(features))
    if len(features) != 20: raise ValueError(f"Expected 20 selected features, but found {len(features)}.")
    return features


def load_split_ids():
    """Retrieve or load split ids for subsequent analysis."""
    train_file = SPLIT_DIR / "train.csv"
    val_file = SPLIT_DIR / "val.csv"
    if not val_file.exists(): val_file = SPLIT_DIR / "validation.csv"
    test_file = SPLIT_DIR / "test.csv"
    train_ids = pd.read_csv(train_file)[ID_COL].astype(str).tolist()
    val_ids = pd.read_csv(val_file)[ID_COL].astype(str).tolist()
    test_ids = pd.read_csv(test_file)[ID_COL].astype(str).tolist()
    return set(train_ids), set(val_ids), set(test_ids)


def load_data(features):
    """Retrieve or load data for subsequent analysis."""
    dev_df = pd.read_excel(TRAIN_VAL_FILE)
    test_df = pd.read_excel(TEST_FILE)
    dev_df[ID_COL] = dev_df[ID_COL].astype(str)
    test_df[ID_COL] = test_df[ID_COL].astype(str)
    train_ids, val_ids, test_ids = load_split_ids()
    dev_df = dev_df[dev_df[ID_COL].isin(train_ids | val_ids)].copy()
    test_df = test_df[test_df[ID_COL].isin(test_ids)].copy()
    missing_dev = [c for c in features + [TARGET] if c not in dev_df.columns]
    missing_test = [c for c in features + [TARGET] if c not in test_df.columns]
    if missing_dev: raise ValueError(f"Missing columns in development data: {missing_dev}")
    if missing_test: raise ValueError(f"Missing columns in test data: {missing_test}")
    if dev_df[ID_COL].duplicated().any(): raise ValueError("Duplicate Sample_ID detected in development data.")
    if test_df[ID_COL].duplicated().any(): raise ValueError("Duplicate Sample_ID detected in test data.")
    if set(dev_df[ID_COL]).intersection(set(test_df[ID_COL])): raise ValueError("Train/Val and Test overlap detected.")
    dev_df = dev_df.dropna(subset=features + [TARGET]).reset_index(drop=True)
    X_dev = dev_df[features].copy()
    y_dev = dev_df[TARGET].to_numpy(dtype=float)
    ids_dev = dev_df[ID_COL].to_numpy()
    X_test = test_df[features].copy()
    y_test = test_df[TARGET].to_numpy(dtype=float)
    ids_test = test_df[ID_COL].to_numpy()
    X_dev = X_dev.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_dev = X_dev.fillna(X_dev.median())
    X_test = X_test.fillna(X_dev.median())
    return X_dev, y_dev, ids_dev, X_test, y_test, ids_test


def load_base_models():
    """Retrieve or load base models for subsequent analysis."""
    model_names = ["ExtraTrees", "RF", "GBR", "HistGBR", "SVR", "BPNN", "XGB"]
    models = {}
    for name in model_names:
        model_dir = MODEL_ROOT / name
        joblib_file = model_dir / "best_model.joblib"
        pkl_file = model_dir / "best_model.pkl"
        if joblib_file.exists():
            models[name] = joblib.load(joblib_file)
            print(f"  {name}: best_model.joblib")
        elif pkl_file.exists():
            models[name] = joblib.load(pkl_file)
            print(f"  {name}: best_model.pkl")
        else:
            print(f"[Warning] Model file not found: {model_dir}")
    if ET_NAME not in models: raise FileNotFoundError("ExtraTrees model not found.")
    return models


def crossfit_oof(X, y, models, n_splits=5, seed=42):
    """Perform crossfit oof."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {name: np.zeros(len(X)) for name in models}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        print(f"    OOF fold {fold}/{n_splits}")
        X_tr, X_va, y_tr = X.iloc[tr_idx], X.iloc[va_idx], y[tr_idx]
        for name, model in models.items():
            m = clone(model)
            m.fit(X_tr, y_tr)
            oof[name][va_idx] = m.predict(X_va)
    return pd.DataFrame(oof)


def fit_full_models(X, y, X_test, models):
    """Perform fit full models."""
    test_predictions = {}
    fitted_models = {}
    for name, model in models.items():
        print(f"    Full-development fit: {name}")
        m = clone(model)
        m.fit(X, y)
        fitted_models[name] = m
        test_predictions[name] = m.predict(X_test)
    return fitted_models, pd.DataFrame(test_predictions)


def evaluate_predictions(y, prediction_dict):
    """Perform evaluate predictions."""
    rows = [{"Model": name, **metrics(y, pred)} for name, pred in prediction_dict.items()]
    return pd.DataFrame(rows).set_index("Model").sort_values("R2", ascending=False)


def get_model_r2(oof_df, y):
    """Retrieve or load model r2 for subsequent analysis."""
    return {name: r2_score(y, oof_df[name]) for name in oof_df.columns}


def residual_diversity(oof_df, y):
    """Perform residual diversity."""
    et_residual = y - oof_df[ET_NAME].to_numpy()
    rows = []
    for name in oof_df.columns:
        if name == ET_NAME: continue
        residual = y - oof_df[name].to_numpy()
        corr = np.corrcoef(et_residual, residual)[0, 1]
        rows.append({"Model": name, "R2": r2_score(y, oof_df[name]), "Residual_Correlation_with_ET": corr, "Residual_Diversity": 1 - abs(corr)})
    return pd.DataFrame(rows).sort_values(["Residual_Correlation_with_ET", "R2"], ascending=[True, False]).reset_index(drop=True)


def generate_candidate_subsets(oof_df, y):
    """Perform generate candidate subsets."""
    names = [x for x in oof_df.columns if x != ET_NAME]
    r2_table = get_model_r2(oof_df, y)
    et_r2 = r2_table[ET_NAME]
    candidates = [name for name in names if r2_table[name] >= et_r2 * MIN_MODEL_R2_FRACTION]
    if not candidates: candidates = sorted(names, key=lambda x: r2_table[x], reverse=True)
    subsets = [combo for k in range(1, min(MAX_COMPLEMENTARY_MODELS, len(candidates)) + 1) for combo in combinations(candidates, k)]
    return [tuple(x) for x in subsets]


def fit_scalar_blend(y, oof_df, predict_df, selected):
    """Perform fit scalar blend."""
    et, et_t = oof_df[ET_NAME].to_numpy(), predict_df[ET_NAME].to_numpy()
    comp, comp_t = oof_df[list(selected)].mean(axis=1).to_numpy(), predict_df[list(selected)].mean(axis=1).to_numpy()
    delta, delta_t = comp - et, comp_t - et_t
    denominator = np.sum(delta ** 2)
    alpha = 0.0 if denominator <= 1e-12 else np.sum(delta * (y - et)) / denominator
    alpha = float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))
    return et_t + alpha * delta_t, alpha


def fit_nonnegative_residual(y, oof_df, predict_df, selected):
    """Perform fit nonnegative residual."""
    et, et_t = oof_df[ET_NAME].to_numpy(), predict_df[ET_NAME].to_numpy()
    X = np.column_stack([oof_df[name].to_numpy() - et for name in selected])
    Xt = np.column_stack([predict_df[name].to_numpy() - et_t for name in selected])
    residual = y - et
    n = X.shape[1]
    objective = lambda w: mean_squared_error(residual, X @ w)
    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n, options={"maxiter": 2000, "ftol": 1e-12})
    weights = result.x if result.success else x0
    return et_t + Xt @ weights, dict(zip(selected, weights))


def fit_ridge_residual(y, oof_df, predict_df, selected):
    """Perform fit ridge residual."""
    et, et_t = oof_df[ET_NAME].to_numpy(), predict_df[ET_NAME].to_numpy()
    X = np.column_stack([oof_df[name].to_numpy() - et for name in selected])
    Xt = np.column_stack([predict_df[name].to_numpy() - et_t for name in selected])
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X, y - et)
    return et_t + model.predict(Xt), model


def fit_elastic_residual(y, oof_df, predict_df, selected):
    """Perform fit elastic residual."""
    et, et_t = oof_df[ET_NAME].to_numpy(), predict_df[ET_NAME].to_numpy()
    X = np.column_stack([oof_df[name].to_numpy() - et for name in selected])
    Xt = np.column_stack([predict_df[name].to_numpy() - et_t for name in selected])
    model = make_pipeline(StandardScaler(), ElasticNet(alpha=ELASTIC_ALPHA, l1_ratio=ELASTIC_L1_RATIO, max_iter=10000, random_state=RANDOM_STATE))
    model.fit(X, y - et)
    return et_t + model.predict(Xt), model


def candidate_methods(y, oof_df, predict_df, fixed_subsets=None):
    """Perform candidate methods."""
    candidates = {"ExtraTrees": predict_df[ET_NAME].to_numpy()}
    metadata = {}
    subsets = fixed_subsets if fixed_subsets is not None else generate_candidate_subsets(oof_df, y)
    for selected in subsets:
        tag = "+".join(selected)
        pred, alpha = fit_scalar_blend(y, oof_df, predict_df, selected)
        candidates[f"ET_Blend_{tag}"] = pred
        metadata[f"ET_Blend_{tag}"] = {"Method": "Scalar_Blend", "Selected": selected, "Alpha": alpha}
        pred, weights = fit_nonnegative_residual(y, oof_df, predict_df, selected)
        candidates[f"ET_NNResidual_{tag}"] = pred
        metadata[f"ET_NNResidual_{tag}"] = {"Method": "Nonnegative_Residual", "Selected": selected, "Weights": weights}
        pred, ridge_model = fit_ridge_residual(y, oof_df, predict_df, selected)
        candidates[f"ET_RidgeResidual_{tag}"] = pred
        metadata[f"ET_RidgeResidual_{tag}"] = {"Method": "Ridge_Residual", "Selected": selected, "Meta_Model": ridge_model}
        pred, elastic_model = fit_elastic_residual(y, oof_df, predict_df, selected)
        candidates[f"ET_ElasticResidual_{tag}"] = pred
        metadata[f"ET_ElasticResidual_{tag}"] = {"Method": "ElasticNet_Residual", "Selected": selected, "Meta_Model": elastic_model}
    return candidates, metadata


def get_fixed_candidate_names(oof_df, y):
    """Retrieve or load fixed candidate names for subsequent analysis."""
    subsets = generate_candidate_subsets(oof_df, y)
    names = ["ExtraTrees"]
    for selected in subsets:
        tag = "+".join(selected)
        names.extend([f"ET_Blend_{tag}", f"ET_NNResidual_{tag}", f"ET_RidgeResidual_{tag}", f"ET_ElasticResidual_{tag}"])
    return names, subsets


def select_best_strategy(y, oof_df):
    """Perform select best strategy."""
    candidate_names, subsets = get_fixed_candidate_names(oof_df, y)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 100)
    fold_scores = {name: [] for name in candidate_names}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(oof_df), 1):
        print(f"    Meta-selection fold {fold}/{N_SPLITS}")
        train_df, val_df = oof_df.iloc[tr_idx].reset_index(drop=True), oof_df.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va = y[tr_idx], y[va_idx]
        _, _ = candidate_methods(y_tr, train_df, train_df, subsets)
        val_candidates, _ = candidate_methods(y_tr, train_df, val_df, subsets)
        for name in candidate_names: fold_scores[name].append(r2_score(y_va, val_candidates[name]))
    rows = [{"Model": name, "Mean_R2": np.mean(scores), "Std_R2": np.std(scores, ddof=1), "Min_R2": np.min(scores), "Max_R2": np.max(scores)} for name, scores in fold_scores.items()]
    return pd.DataFrame(rows).sort_values(["Mean_R2", "Std_R2"], ascending=[False, True]).reset_index(drop=True), subsets


def nested_oof(X, y, models):
    """Perform nested oof."""
    outer_kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    names = list(models.keys())
    base_oof = pd.DataFrame(index=np.arange(len(X)), columns=names, dtype=float)
    ensemble_oof = np.full(len(X), np.nan)
    fold_records, selection_records = [], []
    for fold, (outer_tr, outer_va) in enumerate(outer_kf.split(X), 1):
        print(f"\n{'=' * 70}\nOuter Fold {fold}/{N_SPLITS}\n{'=' * 70}")
        X_tr, X_va, y_tr, y_va = X.iloc[outer_tr], X.iloc[outer_va], y[outer_tr], y[outer_va]
        inner_oof = crossfit_oof(X_tr, y_tr, models, N_SPLITS, RANDOM_STATE + fold)
        _, va_predictions = fit_full_models(X_tr, y_tr, X_va, models)
        base_oof.loc[outer_va, names] = va_predictions[names].to_numpy()
        strategy_table, subsets = select_best_strategy(y_tr, inner_oof)
        best_strategy = strategy_table.iloc[0]["Model"]
        val_candidates, _ = candidate_methods(y_tr, inner_oof, va_predictions, subsets)
        ensemble_oof[outer_va] = val_candidates[best_strategy]
        selection_records.append({"Fold": fold, "Selected_Strategy": best_strategy, "Selection_Mean_R2": strategy_table.iloc[0]["Mean_R2"], "Selection_Std_R2": strategy_table.iloc[0]["Std_R2"]})
        fold_records.append({"Fold": fold, "Selected_Strategy": best_strategy, "ExtraTrees_R2": r2_score(y_va, va_predictions[ET_NAME]), "Ensemble_R2": r2_score(y_va, val_candidates[best_strategy])})
    pd.DataFrame(fold_records).to_excel(OUTPUT_ROOT / "nested_oof_fold_performance.xlsx", index=False)
    pd.DataFrame(selection_records).to_excel(OUTPUT_ROOT / "nested_oof_strategy_selection.xlsx", index=False)
    return base_oof, ensemble_oof, names


def save_base_diagnostics(base_oof, names, y):
    """Generate or save base diagnostics."""
    result = pd.DataFrame([{"Model": name, **metrics(y, base_oof[name])} for name in names]).set_index("Model").sort_values("R2", ascending=False)
    result.to_excel(OUTPUT_ROOT / "base_model_nested_oof_performance.xlsx")
    base_oof.corr().to_excel(OUTPUT_ROOT / "base_model_prediction_correlation.xlsx")
    residual_df = pd.DataFrame({name: y - base_oof[name].to_numpy() for name in names})
    residual_df.corr().to_excel(OUTPUT_ROOT / "base_model_residual_correlation.xlsx")
    diversity_df = residual_diversity(base_oof, y)
    diversity_df.to_excel(OUTPUT_ROOT / "ET_residual_diversity.xlsx", index=False)
    return result


def save_prediction_table(ids, y_true, prediction_dict, path):
    """Generate or save prediction table."""
    df = pd.DataFrame({ID_COL: ids, "Observed_Pn": y_true})
    for name, pred in prediction_dict.items(): df[f"Predicted_{name}"] = pred
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_ensemble_metadata(best_strategy, metadata, output_path):
    """Generate or save ensemble metadata."""
    if best_strategy == ET_NAME:
        df = pd.DataFrame([{"Strategy": ET_NAME, "Method": "Base_Model", "Selected": ET_NAME}])
    else:
        info = metadata[best_strategy]
        row = {"Strategy": best_strategy, "Method": info.get("Method", ""), "Selected": "+".join(info.get("Selected", ()))}
        if "Alpha" in info: row["Alpha"] = info["Alpha"]
        if "Weights" in info: row["Weights"] = str(info["Weights"])
        if "Meta_Model" in info: row["Meta_Model"] = type(info["Meta_Model"]).__name__
        df = pd.DataFrame([row])
    df.to_excel(output_path, index=False)
    return df


def final_test_evaluation(X_dev, y_dev, ids_dev, X_test, y_test, ids_test, models):
    """Perform final test evaluation."""
    print(f"\n{'=' * 70}\nFinal Full-Development Training\n{'=' * 70}")
    fitted_models, test_pred_df = fit_full_models(X_dev, y_dev, X_test, models)
    print("\nGenerating development OOF predictions for final ensemble fitting...")
    dev_oof = crossfit_oof(X_dev, y_dev, models, N_SPLITS, RANDOM_STATE)
    strategy_table, subsets = select_best_strategy(y_dev, dev_oof)
    strategy_table.to_excel(OUTPUT_ROOT / "final_strategy_selection.xlsx", index=False)
    best_strategy = strategy_table.iloc[0]["Model"]
    print(f"\nFinal selected strategy: {best_strategy}")
    final_candidates, metadata = candidate_methods(y_dev, dev_oof, test_pred_df, subsets)
    test_results = evaluate_predictions(y_test, final_candidates)
    base_results = evaluate_predictions(y_test, {name: test_pred_df[name].to_numpy() for name in test_pred_df.columns})
    test_results["Model_Size_MB"] = [strategy_size_mb(name, fitted_models, metadata) for name in test_results.index]
    test_results.to_excel(OUTPUT_ROOT / "independent_test_performance.xlsx")
    base_results.to_excel(OUTPUT_ROOT / "base_model_independent_test_performance.xlsx")
    prediction_dict = {name: test_pred_df[name].to_numpy() for name in test_pred_df.columns}
    prediction_dict[best_strategy] = final_candidates[best_strategy]
    save_prediction_table(ids_test, y_test, prediction_dict, OUTPUT_ROOT / "independent_test_predictions.csv")
    model_records = []
    for name, model in fitted_models.items():
        model_path = OUTPUT_ROOT / f"final_{name}_model.joblib"
        joblib.dump(model, model_path, compress=3)
        model_records.append({"Model": name, "Saved_Model": str(model_path), "Model_Size_MB": serialized_size_mb(model)})
    pd.DataFrame(model_records).to_excel(OUTPUT_ROOT / "final_models.xlsx", index=False)
    best_model_path = OUTPUT_ROOT / "best_ensemble_model.joblib"
    save_best_ensemble_model(best_strategy, fitted_models, metadata, best_model_path)
    best_model_size = serialized_size_mb(joblib.load(best_model_path))
    best_info = save_ensemble_metadata(best_strategy, metadata, OUTPUT_ROOT / "best_ensemble_model_info.xlsx")
    metadata_rows = []
    for name, info in metadata.items():
        row = {"Strategy": name, "Method": info.get("Method", ""), "Selected": "+".join(info.get("Selected", ()))}
        if "Alpha" in info: row["Alpha"] = info["Alpha"]
        if "Weights" in info: row["Weights"] = str(info["Weights"])
        if "Meta_Model" in info: row["Meta_Model"] = type(info["Meta_Model"]).__name__
        metadata_rows.append(row)
    pd.DataFrame(metadata_rows).to_excel(OUTPUT_ROOT / "final_ensemble_metadata.xlsx", index=False)
    selected_metrics = metrics(y_test, final_candidates[best_strategy])
    print("\nBase-model independent test performance:")
    print(base_results.round(4))
    print("\nExtraTrees-anchored ensemble performance:")
    print(test_results.head(20).round(4))
    print("\nFinal selected strategy:")
    print(best_strategy)
    print("\nFinal selected strategy test metrics:")
    print(pd.Series(selected_metrics).round(4))
    print(f"\nBest ensemble model saved to:\n{best_model_path}")
    print(f"Best ensemble model size: {best_model_size:.4f} MB")
    return test_results, base_results, best_strategy, selected_metrics, strategy_table, dev_oof


def main():
    """Run the main analysis workflow."""
    set_seed(RANDOM_STATE)
    print("=" * 70)
    print("Ensemble Learning 5.0")
    print("ExtraTrees-Anchored Residual Ensemble Framework")
    print("=" * 70)
    print("\n[1] Loading selected features...")
    features = load_selected_features()
    print(f"Selected features: {len(features)}")
    print("\n[2] Loading data...")
    X, y, ids, X_test, y_test, ids_test = load_data(features)
    print(f"Development samples: {len(X)}")
    print(f"Independent test samples: {len(X_test)}")
    print("\n[3] Loading base models...")
    models = load_base_models()
    print(f"Loaded models: {list(models.keys())}")
    print("\n[4] Nested OOF evaluation...")
    base_oof, ensemble_oof, names = nested_oof(X, y, models)
    print("\n[5] Saving nested OOF diagnostics...")
    base_results = save_base_diagnostics(base_oof, names, y)
    print(base_results.round(4))
    pd.DataFrame({"Observed_Pn": y, "Ensemble_5.0_OOF": ensemble_oof}).to_csv(OUTPUT_ROOT / "nested_oof_ensemble_prediction.csv", index=False, encoding="utf-8-sig")
    print("\n[6] Final independent test evaluation...")
    test_results, base_test_results, best_strategy, selected_metrics, strategy_table, dev_oof = final_test_evaluation(X, y, ids, X_test, y_test, ids_test, models)
    print("\n" + "=" * 70)
    print("Completed")
    print("=" * 70)
    print(f"Final strategy: {best_strategy}")
    print(f"Test R2: {selected_metrics['R2']:.4f}")
    print(f"Results saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()