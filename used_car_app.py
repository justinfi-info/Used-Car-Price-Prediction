from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from car_model_pipeline import FeatureStats, prepare_single_input, train_and_select_best

MODEL_PATH = Path("best_used_car_price_model.pkl")
DATA_PATH = Path("used_cars.csv")

st.set_page_config(page_title="Used Car Price Prediction", page_icon="car", layout="wide")


@st.cache_resource
def load_or_train_artifact() -> dict:
    if MODEL_PATH.exists():
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "model" in loaded and "feature_stats" in loaded:
            return loaded

    if not DATA_PATH.exists():
        raise FileNotFoundError("used_cars.csv not found.")

    return train_and_select_best(data_path=DATA_PATH, artifact_path=MODEL_PATH)


def input_profile(model_name: str, model_year: int, mileage: int, engine_desc: str) -> dict:
    checks = {
        "model_name": bool(str(model_name).strip()),
        "model_year": int(model_year) > 0,
        "mileage": int(mileage) >= 0,
        "engine_desc": bool(str(engine_desc).strip()),
        "engine_hp": bool(re.search(r"(\d+)\s*HP", str(engine_desc))),
        "engine_size": bool(re.search(r"(\d+\.\d+)", str(engine_desc))),
    }
    completeness = sum(1 for ok in checks.values() if ok) / len(checks)
    return {"checks": checks, "completeness": completeness}


def choose_best_model(
    models: dict,
    leaderboard: pd.DataFrame,
    best_model_name: str,
    profile: dict,
    prediction_df: pd.DataFrame,
    model_year: int,
    mileage: int,
) -> tuple[str, str, str, pd.DataFrame]:
    if not models:
        empty = pd.DataFrame(columns=["model", "score", "performance_score", "profile_fit"])
        return best_model_name, "Unavailable", "No models are loaded.", empty

    if not leaderboard.empty and "model" in leaderboard.columns:
        score_df = leaderboard[["model", "r2_log", "rmse_price"]].copy()
        rmse_max = float(score_df["rmse_price"].max())
        rmse_min = float(score_df["rmse_price"].min())
        spread = max(1e-9, rmse_max - rmse_min)
        score_df["perf_rmse"] = 1.0 - ((score_df["rmse_price"] - rmse_min) / spread)
        score_df["perf_r2"] = score_df["r2_log"].clip(lower=0.0, upper=1.0)
        score_df["performance_score"] = 0.6 * score_df["perf_rmse"] + 0.4 * score_df["perf_r2"]
    else:
        score_df = pd.DataFrame({"model": list(models.keys()), "performance_score": 0.5})

    score_df = score_df[score_df["model"].isin(models.keys())].copy()
    if score_df.empty:
        score_df = pd.DataFrame({"model": list(models.keys()), "performance_score": 0.5})

    completeness = float(profile.get("completeness", 0.0))
    checks = profile.get("checks", {})
    has_engine_hp = bool(checks.get("engine_hp"))
    has_engine_size = bool(checks.get("engine_size"))
    has_engine_features = has_engine_hp and has_engine_size
    high_mileage = int(mileage) > 100000
    very_old = int(model_year) < 2008

    profile_fit = []
    for name in score_df["model"]:
        fit = completeness
        # Tree models are more robust to partial/noisy inputs.
        if not has_engine_features and name == "RandomForest":
            fit += 0.10
        if not has_engine_features and name in ("SVR", "LinearRegression"):
            fit -= 0.10
        if high_mileage and name == "LinearRegression":
            fit += 0.08
        if very_old and name == "RandomForest":
            fit += 0.05
        fit = min(max(fit, 0.0), 1.0)
        profile_fit.append(fit)

    score_df["profile_fit"] = profile_fit

    if not prediction_df.empty and "predicted_price" in prediction_df.columns:
        pred_ref = prediction_df[["model", "predicted_price"]].copy()
        pred_ref["abs_dev"] = (pred_ref["predicted_price"] - pred_ref["predicted_price"].median()).abs()
        dev_max = max(1e-9, float(pred_ref["abs_dev"].max()))
        pred_ref["agreement_score"] = 1.0 - (pred_ref["abs_dev"] / dev_max)
        score_df = score_df.merge(pred_ref[["model", "agreement_score"]], on="model", how="left")
        score_df["agreement_score"] = score_df["agreement_score"].fillna(0.5)
    else:
        score_df["agreement_score"] = 0.5

    score_df["score"] = (
        0.50 * score_df["performance_score"]
        + 0.25 * score_df["profile_fit"]
        + 0.25 * score_df["agreement_score"]
    )
    score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

    selected_name = str(score_df["model"].iloc[0])
    selected_score = float(score_df["score"].iloc[0])

    if selected_score < 0.35:
        fallback = best_model_name if best_model_name in models else list(models.keys())[0]
        return fallback, "Ready", "Weak profile-model match; fallback default applied.", score_df

    return selected_name, "Ready", "Auto-selected best-fit model from current input profile.", score_df


st.title("Used Car Price Prediction")
st.caption("Leakage-safe model training, fair comparison, and best-model inference.")

try:
    artifact = load_or_train_artifact()
except Exception as exc:
    st.error(f"Unable to load or train model artifact: {exc}")
    st.stop()

model = artifact["model"]
best_model_name = artifact.get("model_name", "Unknown")
metrics = artifact.get("metrics", {})
leaderboard = pd.DataFrame(artifact.get("leaderboard", []))
stats = FeatureStats.from_dict(artifact["feature_stats"])
models = artifact.get("models", {})
if not isinstance(models, dict) or not models:
    models = {best_model_name: model}

st.sidebar.header("Enter Car Details")
user_model_name = st.sidebar.text_input("Model Name", value="Corolla Altis")
model_year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2100, value=2017)
mileage = st.sidebar.number_input("Mileage (miles)", min_value=0, value=50000, step=1000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "CVT"])
ext_col = st.sidebar.selectbox("Exterior Color", ["White", "Black", "Silver", "Gray", "Red", "Blue", "Other"])
int_col = st.sidebar.selectbox("Interior Color", ["Black", "White", "Gray", "Beige", "Red", "Blue", "Other"])
accident = st.sidebar.selectbox("Accident History", ["No", "Yes", "No Info"])
engine_desc = st.sidebar.text_input("Engine Description", value="180 HP 2.0L V4")
predict_clicked = st.sidebar.button("Get Price Estimate", type="primary", use_container_width=True)

if "show_prediction" not in st.session_state:
    st.session_state["show_prediction"] = False

if predict_clicked:
    st.session_state["show_prediction"] = True

input_row = prepare_single_input(
    stats=stats,
    model_name=user_model_name,
    model_year=int(model_year),
    mileage=int(mileage),
    fuel_type=fuel_type,
    transmission=transmission,
    ext_col=ext_col,
    int_col=int_col,
    accident=accident,
    engine_desc=engine_desc,
)

prediction_rows = []
for name, model_obj in models.items():
    pred_log = float(model_obj.predict(input_row)[0])
    pred_price = max(0.0, float(np.expm1(pred_log)))
    prediction_rows.append({"model": name, "predicted_price": pred_price, "predicted_log_price": pred_log})

prediction_df = pd.DataFrame(prediction_rows).sort_values("predicted_price").reset_index(drop=True)
profile = input_profile(user_model_name, int(model_year), int(mileage), engine_desc)
selected_model_name, model_status, model_status_note, model_scores = choose_best_model(
    models=models,
    leaderboard=leaderboard,
    best_model_name=best_model_name,
    profile=profile,
    prediction_df=prediction_df,
    model_year=int(model_year),
    mileage=int(mileage),
)
selected_row = prediction_df[prediction_df["model"] == selected_model_name]
selected_price = float(selected_row["predicted_price"].iloc[0]) if not selected_row.empty else float(
    prediction_df["predicted_price"].iloc[0]
)

if st.session_state["show_prediction"]:
    st.success(f"Estimated Price ({selected_model_name}): ${selected_price:,.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Summary")
        st.write(f"Model: {user_model_name}")
        st.write(f"Year: {int(model_year)}")
        st.write(f"Mileage: {int(mileage):,} miles")
        st.write(f"Fuel: {fuel_type}")
        st.write(f"Transmission: {transmission}")
        st.write(f"Accident: {accident}")

    with col2:
        upper = max(5000.0, selected_price * 1.5)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=selected_price,
                number={"prefix": "$", "valueformat": ",.0f"},
                title={"text": f"Estimated Value ({selected_model_name})"},
                gauge={
                    "axis": {"range": [0, upper]},
                    "bar": {"color": "#e74c3c"},
                    "steps": [
                        {"range": [0, upper * 0.4], "color": "#f7f7f7"},
                        {"range": [upper * 0.4, upper * 0.7], "color": "#ffe9b3"},
                        {"range": [upper * 0.7, upper], "color": "#d4f4dd"},
                    ],
                },
            )
        )
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Model Status")
st.write(
    f"Applied Model: **{selected_model_name}**  |  Status: **{model_status}**  |  Best Overall: **{best_model_name}**"
)
st.caption(model_status_note)

price_min = float(prediction_df["predicted_price"].min())
price_max = float(prediction_df["predicted_price"].max())
price_spread = price_max - price_min
mean_price = float(prediction_df["predicted_price"].mean())

s1, s2, s3, s4 = st.columns(4)
s1.metric("Selected Prediction", f"${selected_price:,.0f}")
s2.metric("Average Across Models", f"${mean_price:,.0f}")
s3.metric("Prediction Spread", f"${price_spread:,.0f}")
s4.metric("Model Count", f"{len(prediction_df)}")

if not leaderboard.empty:
    selected_eval = leaderboard[leaderboard["model"] == selected_model_name]
    if not selected_eval.empty:
        eval_row = selected_eval.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R2 (log target)", f"{float(eval_row['r2_log']):.3f}")
        m2.metric("MAE (price)", f"${float(eval_row['mae_price']):,.0f}")
        m3.metric("RMSE (price)", f"${float(eval_row['rmse_price']):,.0f}")
        m4.metric("Within 20%", f"{float(eval_row['within_20_percent']):.1f}%")
elif metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R2 (log target)", f"{metrics.get('r2_log', float('nan')):.3f}")
    m2.metric("MAE (price)", f"${metrics.get('mae_price', 0.0):,.0f}")
    m3.metric("RMSE (price)", f"${metrics.get('rmse_price', 0.0):,.0f}")
    m4.metric("Within 20%", f"{metrics.get('within_20_percent', 0.0):.1f}%")

if not leaderboard.empty:
    st.markdown("### Fair Model Comparison")
    merge_df = leaderboard.merge(prediction_df[["model", "predicted_price"]], on="model", how="left")
    merge_df["delta_vs_selected"] = merge_df["predicted_price"] - selected_price
    if not model_scores.empty:
        merge_df = merge_df.merge(model_scores[["model", "score", "profile_fit"]], on="model", how="left")
    show_cols = [
        "model",
        "predicted_price",
        "delta_vs_selected",
        "score",
        "profile_fit",
        "r2_log",
        "mae_price",
        "rmse_price",
        "mape_percent",
        "within_20_percent",
    ]
    st.dataframe(merge_df[show_cols], use_container_width=True)
