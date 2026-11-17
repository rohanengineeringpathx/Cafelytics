
from pathlib import Path
import pandas as pd

# Load sales dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sales.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Calculate revenue
df["revenue"] = df["quantity"] * df["unit_price"]

# Basic business metrics
total_revenue = df["revenue"].sum()
total_orders = df["quantity"].sum()
unique_products = df["product"].nunique()

# Daily sales
daily_sales = (
    df.groupby("date")["revenue"]
    .sum()
    .sort_index()
)

# Best-selling products
product_sales = (
    df.groupby("product")
    .agg(
        total_quantity=("quantity", "sum"),
        total_revenue=("revenue", "sum")
    )
    .sort_values("total_quantity", ascending=False)
)

# Category performance
category_sales = (
    df.groupby("category")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

# Display results
print("\n========== CAFELYTICS SALES ANALYTICS ==========")

print(f"\nTotal Revenue: Rs. {total_revenue:,.2f}")
print(f"Total Items Sold: {total_orders}")
print(f"Unique Products: {unique_products}")

print("\n--- DAILY REVENUE ---")
print(daily_sales.to_string())

print("\n--- BEST SELLING PRODUCTS ---")
print(product_sales.to_string())

print("\n--- CATEGORY PERFORMANCE ---")
print(category_sales.to_string())

print("\n========== ANALYSIS COMPLETE ==========")
