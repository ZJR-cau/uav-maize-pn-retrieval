"""Define traditional machine-learning estimators and shared model utilities."""
# ================================================================
# traditional_ml_model.py

# ================================================================

from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor


MODEL_CONFIGS = {
    "MLR": {
        "model": LinearRegression(),
        "scale": True,
        "param_grid": {
            "model__fit_intercept": [True, False]
        }
    },

    "Ridge": {
        "model": Ridge(),
        "scale": True,
        "param_grid": {
            "model__alpha": [0.01, 0.1, 1, 10, 100]
        }
    },

    "Lasso": {
        "model": Lasso(max_iter=20000),
        "scale": True,
        "param_grid": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1]
        }
    },

    "ElasticNet": {
        "model": ElasticNet(max_iter=20000),
        "scale": True,
        "param_grid": {
            "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    },

    "SVR": {
        "model": SVR(),
        "scale": True,
        "param_grid": {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1, 10, 100],
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto"]
        }
    },

    "BPNN": {
        "model": MLPRegressor(
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42
        ),
        "scale": True,
        "param_grid": {
            "model__hidden_layer_sizes": [
                (16,),
                (32,),
                (64,),
                (32, 16),
                (64, 32),
                (64, 32, 16)
            ],
            "model__activation": [
                "relu",
                "tanh"
            ],
            "model__alpha": [
                0.0001,
                0.001,
                0.01
            ],
            "model__learning_rate_init": [
                0.001,
                0.01
            ]
        }
    },

    "RF": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        "scale": False,
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 1.0]
        }
    },

    "ExtraTrees": {
        "model": ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1
        ),
        "scale": False,
        "param_grid": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 1.0]
        }
    },

    "GBR": {
        "model": GradientBoostingRegressor(
            random_state=42
        ),
        "scale": False,
        "param_grid": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 5],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    },

    "HistGBR": {
        "model": HistGradientBoostingRegressor(
            random_state=42
        ),
        "scale": False,
        "param_grid": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__max_depth": [None, 5, 10],
            "model__l2_regularization": [0, 0.1, 1]
        }
    },

    "XGB": {
        "model": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ),
        "scale": False,
        "param_grid": {
            "model__n_estimators": [500],
            "model__max_depth": [7],
            "model__learning_rate": [0.01],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
            "model__min_child_weight": [1],
            "model__gamma": [0, 0.1],
            "model__reg_alpha": [0],
            "model__reg_lambda": [1]
        }
    }
}


def get_model_names():
    """Retrieve or load model names for subsequent analysis."""
    return list(MODEL_CONFIGS.keys())


def get_model(name):
    """Retrieve or load model for subsequent analysis."""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Available models: {get_model_names()}"
        )
    config = MODEL_CONFIGS[name]
    if config["scale"]:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", config["model"])
        ])
    else:
        pipeline = Pipeline([
            ("model", config["model"])
        ])
    return pipeline, config["param_grid"]


def count_model_parameters(model):
    """Perform count model parameters."""
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "coefs_"):
        count = 0
        for coef in estimator.coefs_:
            count += coef.size
        for intercept in estimator.intercepts_:
            count += intercept.size
        return int(count)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        count = coef.size
        if hasattr(estimator, "intercept_"):
            count += np.asarray(estimator.intercept_).size
        return int(count)
    if hasattr(estimator, "support_vectors_"):
        count = estimator.support_vectors_.size
        if hasattr(estimator, "dual_coef_"):
            count += estimator.dual_coef_.size
        if hasattr(estimator, "intercept_"):
            count += np.asarray(estimator.intercept_).size
        return int(count)
    if hasattr(estimator, "estimators_"):
        total = 0
        for tree in np.asarray(
            estimator.estimators_,
            dtype=object
        ).ravel():
            if hasattr(tree, "tree_"):
                total += tree.tree_.node_count
        return int(total)
    if hasattr(estimator, "tree_"):
        return int(estimator.tree_.node_count)
    return 0


def get_model_size_mb(model):
    """Retrieve or load model size mb for subsequent analysis."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=".joblib",
        delete=False
    ) as f:
        temp_path = Path(f.name)
    try:
        joblib.dump(
            model,
            temp_path
        )
        size_mb = temp_path.stat().st_size / (1024 ** 2)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return float(size_mb)


def get_best_params_readable(best_params):
    """Retrieve or load best params readable for subsequent analysis."""
    return {
        key.replace("model__", ""): value
        for key, value in best_params.items()
    }