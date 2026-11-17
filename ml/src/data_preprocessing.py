
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data" / "sales.csv"
CLEAN_FILE = BASE_DIR / "data" / "cleaned_sales.csv"
DEMAND_FILE = BASE_DIR / "data" / "daily_product_demand.csv"

print("\n========== CAFELYTICS DATA PREPROCESSING ==========\n")

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))
print("Original columns:", list(df.columns))

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Validate required columns
required = ["date", "product", "category", "quantity", "unit_price"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Remove duplicate records
df = df.drop_duplicates()

# Convert data types
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

# Clean text columns
df["product"] = df["product"].astype("string").str.strip()
df["category"] = df["category"].astype("string").str.strip()

# Remove invalid or incomplete records
df = df.dropna(subset=required)
df = df[
    (df["quantity"] > 0) &
    (df["unit_price"] >= 0) &
    (df["product"] != "") &
    (df["category"] != "")
].copy()

# Calculate revenue
df["revenue"] = df["quantity"] * df["unit_price"]

# Save cleaned dataset
df.to_csv(CLEAN_FILE, index=False)

# Aggregate daily product demand
daily_demand = (
    df.groupby(["date", "product"], as_index=False)
    .agg(
        total_quantity=("quantity", "sum"),
        total_revenue=("revenue", "sum")
    )
)

daily_demand.to_csv(DEMAND_FILE, index=False)

print("\nCleaned records:", len(df))
print("Removed records:", "See original and cleaned counts")
print("\nCleaned dataset:", CLEAN_FILE)
print("Daily demand dataset:", DEMAND_FILE)

print("\n========== PREPROCESSING COMPLETE ==========")
