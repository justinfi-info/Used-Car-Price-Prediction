from pathlib import Path

import pandas as pd

from car_model_pipeline import train_and_select_best


if __name__ == "__main__":
    artifact_path = Path("best_used_car_price_model.pkl")
    artifact = train_and_select_best(data_path="used_cars.csv", artifact_path=artifact_path)

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
