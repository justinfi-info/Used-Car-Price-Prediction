from pathlib import Path

import pandas as pd

from car_model_pipeline import train_and_select_best


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "used_cars.csv"
    artifact_path = project_root / "model" / "best_used_car_price_model.pkl"
    artifact = train_and_select_best(data_path=data_path, artifact_path=artifact_path)

    leaderboard = pd.DataFrame(artifact["leaderboard"])
    print("Best model:", artifact["model_name"])
    print("Saved artifact:", artifact_path)
    print("\nLeaderboard (sorted by rmse_price):")
    print(
        leaderboard[
            [
                "model",
                "r2_log",
                "mae_log",
                "rmse_log",
                "mae_price",
                "rmse_price",
                "mape_percent",
                "within_20_percent",
            ]
        ].to_string(index=False)
    )
