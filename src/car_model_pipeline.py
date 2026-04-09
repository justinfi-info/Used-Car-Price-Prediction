from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

CATEGORICAL_COLS = ["model_main", "fuel_type", "transmission", "ext_col", "int_col", "accident"]
NUMERIC_COLS = ["milage", "HP", "Engine_size", "Cylinders", "car_age"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "used_cars.csv"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "model" / "best_used_car_price_model.pkl"


@dataclass
class FeatureStats:
    top_models: list[str]
    medians: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"top_models": self.top_models, "medians": self.medians}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FeatureStats":
        return FeatureStats(top_models=list(data["top_models"]), medians=dict(data["medians"]))


def simplify_fuel(value: str) -> str:
    text = str(value)
    if text == "Gasoline":
        return "Gasoline"
    if "Hybrid" in text:
        return "Hybrid"
    if text == "Diesel":
        return "Diesel"
    return "Other"


def simplify_transmission(value: str) -> str:
    text = str(value)
    if "A/T" in text or "Automatic" in text:
        return "Automatic"
    if "M/T" in text or "Manual" in text:
        return "Manual"
    if "CVT" in text:
        return "CVT"
    return "Other"


def simplify_color(value: str) -> str:
    text = str(value).lower()
    main_colors = ["white", "black", "silver", "gray", "red", "blue"]
    for color in main_colors:
        if color in text:
            return color.capitalize()
    return "Other"


def parse_mileage(value: str) -> float:
    match = re.search(r"(\d+)", str(value).replace(",", ""))
    return float(match.group(1)) if match else np.nan


def parse_price(value: str) -> float:
    cleaned = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
    return float(cleaned) if cleaned else np.nan


def extract_engine_features(value: str) -> tuple[float, float, float]:
    text = str(value)
    hp_match = re.search(r"(\d+)\s*HP", text)
    size_match = re.search(r"(\d+\.\d+)", text)
    cylinders_match = re.search(r"(\d+)(?!.*\d)", text)

    hp = float(hp_match.group(1)) if hp_match else np.nan
    engine_size = float(size_match.group(1)) if size_match else np.nan
    cylinders = float(cylinders_match.group(1)) if cylinders_match else np.nan
    return hp, engine_size, cylinders


def base_clean(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.ffill().drop_duplicates()
    data["fuel_type"] = data["fuel_type"].fillna("Unknown")
    data["accident"] = data["accident"].replace(
        {"None reported": "No", "None ": "No", "Accident reported": "Yes"}
    )
    data["accident"] = data["accident"].fillna("No Info")
    return data


def fit_feature_stats(train_df: pd.DataFrame) -> FeatureStats:
    model_main = train_df["model"].astype(str).apply(lambda x: x.split()[0] if x else "Other")
    top_models = model_main.value_counts().nlargest(10).index.tolist()

    engine = train_df["engine"].astype(str).apply(extract_engine_features)
    engine_df = pd.DataFrame(engine.tolist(), columns=["HP", "Engine_size", "Cylinders"], index=train_df.index)

    medians = {
        "milage": float(train_df["milage"].apply(parse_mileage).median()),
        "HP": float(engine_df["HP"].median()),
        "Engine_size": float(engine_df["Engine_size"].median()),
        "Cylinders": float(engine_df["Cylinders"].median()),
        "car_age": float((datetime.now().year - pd.to_numeric(train_df["model_year"], errors="coerce")).median()),
    }

    for key, value in medians.items():
        if np.isnan(value):
            medians[key] = 0.0

    return FeatureStats(top_models=top_models, medians=medians)


def transform_features(df: pd.DataFrame, stats: FeatureStats) -> pd.DataFrame:
    data = df.copy()

    data["model_main"] = data["model"].astype(str).apply(lambda x: x.split()[0] if x else "Other")
    data["model_main"] = data["model_main"].apply(lambda x: x if x in stats.top_models else "Other")

    data["fuel_type"] = data["fuel_type"].apply(simplify_fuel)
    data["transmission"] = data["transmission"].apply(simplify_transmission)
    data["ext_col"] = data["ext_col"].apply(simplify_color)
    data["int_col"] = data["int_col"].apply(simplify_color)

    data["accident"] = data["accident"].replace(
        {"None reported": "No", "None ": "No", "Accident reported": "Yes"}
    )
    data["accident"] = data["accident"].fillna("No Info")

    data["milage"] = data["milage"].apply(parse_mileage).fillna(stats.medians["milage"])

    engine = data["engine"].astype(str).apply(extract_engine_features)
    engine_df = pd.DataFrame(engine.tolist(), columns=["HP", "Engine_size", "Cylinders"], index=data.index)
    for col in ["HP", "Engine_size", "Cylinders"]:
        data[col] = engine_df[col].fillna(stats.medians[col])

    data["car_age"] = datetime.now().year - pd.to_numeric(data["model_year"], errors="coerce")
    data["car_age"] = data["car_age"].fillna(stats.medians["car_age"])

    return data[FEATURE_COLS].copy()


def prepare_train_test(
    data_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, FeatureStats]:
    raw = pd.read_csv(data_path)
    clean = base_clean(raw)

    train_df, test_df = train_test_split(clean, test_size=test_size, random_state=random_state)

    stats = fit_feature_stats(train_df)

    y_train_price = train_df["price"].apply(parse_price)
    y_test_price = test_df["price"].apply(parse_price)

    train_mask = y_train_price.notna()
    test_mask = y_test_price.notna()

    train_df = train_df.loc[train_mask].copy()
    test_df = test_df.loc[test_mask].copy()

    y_train_price = y_train_price.loc[train_mask].astype(float)
    y_test_price = y_test_price.loc[test_mask].astype(float)

    X_train = transform_features(train_df, stats)
    X_test = transform_features(test_df, stats)

    y_train_log = np.log1p(y_train_price)
    y_test_log = np.log1p(y_test_price)

    return X_train, X_test, y_train_log, y_test_log, y_train_price, y_test_price, stats


def build_candidates() -> dict[str, Pipeline]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    return {
        "LinearRegression": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
        ),
        "SVR": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", SVR(kernel="rbf", C=10, epsilon=0.1))]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    return {
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
    }


def evaluate_price(y_true_price: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    pred_price = np.expm1(y_pred_log)
    y_true = y_true_price.to_numpy()
    ape = np.abs((y_true - pred_price) / np.clip(y_true, 1e-9, None))

    return {
        "mae_price": float(mean_absolute_error(y_true, pred_price)),
        "rmse_price": float(np.sqrt(mean_squared_error(y_true, pred_price))),
        "mape_percent": float(np.mean(ape) * 100),
        "within_20_percent": float(np.mean(ape <= 0.20) * 100),
    }


def train_and_select_best(
    data_path: str | Path = DEFAULT_DATA_PATH,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    X_train, X_test, y_train_log, y_test_log, y_train_price, y_test_price, stats = prepare_train_test(data_path)

    results: list[dict[str, Any]] = []
    model_store: dict[str, Pipeline] = {}
    best_name = ""
    best_model: Pipeline | None = None
    best_rmse = float("inf")

    for name, pipeline in build_candidates().items():
        pipeline.fit(X_train, y_train_log)
        model_store[name] = pipeline
        pred_log = pipeline.predict(X_test)

        metrics = {"model": name}
        metrics.update(evaluate_predictions(y_test_log, pred_log))
        metrics.update(evaluate_price(y_test_price, pred_log))
        results.append(metrics)

        if metrics["rmse_price"] < best_rmse:
            best_rmse = metrics["rmse_price"]
            best_name = name
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No model was trained.")

    leaderboard = sorted(results, key=lambda x: x["rmse_price"])
    best_metrics = next(item for item in leaderboard if item["model"] == best_name)

    artifact = {
        "model": best_model,
        "models": model_store,
        "model_name": best_name,
        "metrics": best_metrics,
        "leaderboard": leaderboard,
        "feature_stats": stats.to_dict(),
        "feature_cols": FEATURE_COLS,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact


def prepare_single_input(
    stats: FeatureStats,
    model_name: str,
    model_year: int,
    mileage: int,
    fuel_type: str,
    transmission: str,
    ext_col: str,
    int_col: str,
    accident: str,
    engine_desc: str,
) -> pd.DataFrame:
    row = pd.DataFrame(
        {
            "model": [model_name],
            "model_year": [int(model_year)],
            "milage": [mileage],
            "fuel_type": [fuel_type],
            "engine": [engine_desc],
            "transmission": [transmission],
            "ext_col": [ext_col],
            "int_col": [int_col],
            "accident": [accident],
            "price": [np.nan],
        }
    )
    return transform_features(base_clean(row), stats)
