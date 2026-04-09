# Used Car Price Prediction

End-to-end machine learning project for estimating used car prices using multiple models, leakage-safe training, and an interactive Streamlit UI.

## Features

- Leakage-safe train/test split and preprocessing pipeline
- Fair model comparison across:
  - Linear Regression
  - SVR
  - Random Forest
- Automatic best-model selection and artifact saving
- Streamlit app with:
  - Live price prediction
  - Auto model selection based on input profile
  - Dynamic model status and comparison metrics

## Tech Stack

- Python
- pandas, numpy, scikit-learn
- Streamlit
- Plotly
- joblib

## Project Files

Main files used in this project:

- `data/used_cars.csv` - dataset
- `src/car_model_pipeline.py` - preprocessing, feature engineering, training, evaluation, and model selection
- `src/train_models.py` - retrain and compare all models, save best artifact
- `src/used_car_app.py` - Streamlit UI for inference and model status
- `models/best_used_car_price_model.pkl` - saved model artifact (generated after training)
- `requirements.txt` - dependencies

## Project Structure

```text
Used-Car-Price-Prediction/
│
├── data/
│   └── used_cars.csv
│
├── notebooks/
│   └── used_car_price_prediction.ipynb
│
├── src/
│   ├── car_model_pipeline.py
│   ├── train_models.py
│   └── used_car_app.py
│
├── models/
│   └── best_used_car_price_model.pkl
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Setup

1. Create and activate virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train Models

Run the training and comparison script:

```powershell
python train_models.py
```

This will:

- Train Linear Regression, SVR, and Random Forest on the same split
- Evaluate each model with fair metrics
- Select the best model (by `rmse_price`)
- Save the artifact to `best_used_car_price_model.pkl`

## Run App

Start the Streamlit app:

```powershell
python -m streamlit run used_car_app.py
```

Open the shown local URL in your browser.

## How Auto Model Selection Works

- User input is converted into a feature profile.
- All available trained models generate predictions for the same input.
- The app scores model suitability using:
  - historical performance metrics
  - input-profile fit
  - agreement/confidence from current predictions
- The best-fit model is applied automatically.
- Model status updates in the UI.

## Notes

- If the model artifact is missing, the app can train from `used_cars.csv`.
- Keep `used_cars.csv` in the working directory before training/running.
