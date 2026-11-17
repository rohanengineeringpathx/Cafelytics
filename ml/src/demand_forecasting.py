
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "daily_product_demand.csv"
FORECAST_FILE = DATA_DIR / "demand_forecast.csv"
EVALUATION_FILE = DATA_DIR / "model_evaluation.csv"
METRICS_FILE = DATA_DIR / "evaluation_metrics.csv"
GRAPH_FILE = DATA_DIR / "actual_vs_predicted.png"

FORECAST_DAYS = 7

print("\n========== CAFELYTICS DEMAND FORECASTING ==========\n")

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"])

all_forecasts = []
evaluation_rows = []
metrics_rows = []

for product in df["product"].unique():

    product_df = (
        df[df["product"] == product]
        .groupby("date")["total_quantity"]
        .sum()
        .sort_index()
    )

    full_dates = pd.date_range(
        product_df.index.min(),
        product_df.index.max(),
        freq="D"
    )

    product_df = product_df.reindex(full_dates, fill_value=0)

    data = pd.DataFrame({"quantity": product_df})
    data["day_of_week"] = data.index.day_of_week
    data["lag_1"] = data["quantity"].shift(1)
    data["lag_2"] = data["quantity"].shift(2)
    data["lag_3"] = data["quantity"].shift(3)

    train = data.dropna()

    if len(train) < 3:
        print(f"Insufficient data for {product}. Skipping.")
        continue

    features = ["day_of_week", "lag_1", "lag_2", "lag_3"]

    split = max(1, int(len(train) * 0.8))

    X_train = train[features].iloc[:split]
    y_train = train["quantity"].iloc[:split]

    X_test = train[features].iloc[split:]
    y_test = train["quantity"].iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    if len(X_test) > 0:

        predictions = np.maximum(0, model.predict(X_test))

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        denominator = np.sum(np.abs(y_test))

        wape = (
            np.sum(np.abs(y_test.values - predictions))
            / denominator * 100
            if denominator != 0 else 0
        )

        print(
            f"{product} | MAE: {mae:.2f} | "
            f"RMSE: {rmse:.2f} | WAPE: {wape:.2f}%"
        )

        metrics_rows.append({
            "product": product,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "WAPE_Percent": round(wape, 2)
        })

        for date, actual, predicted in zip(
            y_test.index,
            y_test.values,
            predictions
        ):
            evaluation_rows.append({
                "date": date.date(),
                "product": product,
                "actual_demand": int(actual),
                "predicted_demand": round(float(predicted), 2)
            })

    # Train final model on all available historical data
    model.fit(train[features], train["quantity"])

    history = list(product_df.values.astype(float))

    for day in range(FORECAST_DAYS):

        next_date = product_df.index[-1] + pd.Timedelta(days=day + 1)

        row = pd.DataFrame([{
            "day_of_week": next_date.day_of_week,
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_3": history[-3]
        }])

        predicted = max(
            0,
            float(model.predict(row[features])[0])
        )

        predicted = round(predicted)

        all_forecasts.append({
            "date": next_date.date(),
            "product": product,
            "predicted_demand": predicted
        })

        history.append(predicted)

if not all_forecasts:
    raise ValueError("No forecasts generated. Check the historical dataset.")

# Save forecast
forecast_df = pd.DataFrame(all_forecasts)
forecast_df.to_csv(FORECAST_FILE, index=False)

# Save evaluation results
if evaluation_rows:
    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_df.to_csv(EVALUATION_FILE, index=False)

    # Actual vs Predicted graph
    labels = (
        evaluation_df["product"].astype(str)
        + "\n"
        + evaluation_df["date"].astype(str)
    )

    x = np.arange(len(evaluation_df))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        evaluation_df["actual_demand"],
        width,
        label="Actual Demand"
    )

    plt.bar(
        x + width / 2,
        evaluation_df["predicted_demand"],
        width,
        label="Predicted Demand"
    )

    plt.xlabel("Product and Test Date")
    plt.ylabel("Quantity Sold")
    plt.title("Cafelytics - Actual vs Predicted Demand")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(GRAPH_FILE, dpi=150)
    plt.close()

# Save metrics
if metrics_rows:
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(METRICS_FILE, index=False)

print("\n--- 7-DAY DEMAND FORECAST ---")
print(forecast_df.to_string(index=False))

print("\nFiles saved:")
print("Forecast:", FORECAST_FILE)
print("Evaluation:", EVALUATION_FILE)
print("Metrics:", METRICS_FILE)
print("Graph:", GRAPH_FILE)

print("\n========== FORECASTING COMPLETE ==========")
