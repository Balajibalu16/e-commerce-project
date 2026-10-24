# ============================================================
# E-COMMERCE SALES & CUSTOMER ANALYTICS
# RUNNABLE VERSION
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. FIND / CREATE DATASET
# ============================================================

USER_CSV = Path(r"C:\Users\grand\Downloads\ecommerce_sales_customer_analytics.csv")
LOCAL_CSV = Path("ecommerce_sales_customer_analytics.csv")


def create_demo_dataset(path: Path, rows: int = 100) -> pd.DataFrame:
    """Create a small demo dataset so the project can run even without the CSV."""
    rng = np.random.default_rng(42)

    names = [
        "Rahul", "Priya", "Arun", "Kiran", "Sneha", "Vijay", "Anjali",
        "Ravi", "Divya", "Suresh", "Neha", "Manoj", "Pooja", "Amit", "Kavya"
    ]
    cities = ["Hyderabad", "Bengaluru", "Chennai", "Vijayawada", "Mumbai", "Delhi"]
    states = ["Telangana", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Maharashtra", "Delhi"]
    categories = ["Electronics", "Clothing", "Home", "Beauty"]
    subcategories = ["Mobiles", "Laptops", "Shirts", "Shoes", "Kitchen", "Furniture", "Skincare"]
    products = [
        "Smartphone", "Laptop", "Headphones", "T-Shirt", "Jeans", "Running Shoes",
        "Mixer", "Chair", "Face Wash", "Smart Watch"
    ]
    payments = ["UPI", "Credit Card", "Debit Card", "Cash on Delivery"]
    statuses = ["Delivered", "Shipped", "Cancelled"]
    genders = ["Male", "Female"]

    # Make repeated customers possible so RFM analysis is meaningful.
    customer_ids = rng.integers(1001, 1031, size=rows)
    product_idx = rng.integers(0, len(products), size=rows)

    data = {
        "Order_ID": np.arange(50001, 50001 + rows),
        "Customer_ID": customer_ids,
        "Customer_Name": [names[(cid - 1001) % len(names)] for cid in customer_ids],
        "Gender": rng.choice(genders, size=rows),
        "Age": rng.integers(18, 60, size=rows),
        "City": rng.choice(cities, size=rows),
        "State": rng.choice(states, size=rows),
        "Category": rng.choice(categories, size=rows),
        "Sub_Category": rng.choice(subcategories, size=rows),
        "Product": [products[i] for i in product_idx],
        "Quantity": rng.integers(1, 5, size=rows),
        "Unit_Price": rng.integers(300, 50000, size=rows).astype(float),
        "Discount": rng.choice([0, 5, 10, 15, 20], size=rows),
        "Cost": rng.integers(200, 35000, size=rows).astype(float),
        "Rating": np.round(rng.uniform(2.5, 5.0, size=rows), 1),
        "Payment_Mode": rng.choice(payments, size=rows),
        "Order_Status": rng.choice(statuses, size=rows, p=[0.75, 0.15, 0.10]),
        "Order_Date": pd.date_range("2025-01-01", periods=rows, freq="3D"),
    }

    demo = pd.DataFrame(data)
    demo.to_csv(path, index=False)
    return demo


if USER_CSV.exists():
    data_file = USER_CSV
elif LOCAL_CSV.exists():
    data_file = LOCAL_CSV
else:
    data_file = Path("ecommerce_sales_customer_analytics_demo.csv")
    create_demo_dataset(data_file)
    print(f"CSV not found at {USER_CSV}")
    print(f"Demo dataset created: {data_file.resolve()}")


df = pd.read_csv(data_file)

print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Order_ID", "Customer_ID", "Order_Date", "Quantity", "Unit_Price",
    "Discount", "Cost", "Product", "Category", "Sub_Category", "State",
    "City", "Payment_Mode", "Gender", "Order_Status"
]

missing_required = [col for col in required_columns if col not in df.columns]

if missing_required:
    raise ValueError(
        "The CSV is missing these required columns: " + ", ".join(missing_required)
    )


# ============================================================
# 3. DATA CLEANING
# ============================================================

df = df.drop_duplicates().copy()

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

categorical_columns = [
    "Customer_Name", "Gender", "City", "State", "Category",
    "Sub_Category", "Product", "Payment_Mode", "Order_Status"
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("string").fillna("Unknown")

numerical_columns = ["Age", "Quantity", "Unit_Price", "Discount", "Cost", "Rating"]

for col in numerical_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(median) else median)

# Remove rows for which dates are invalid because date-based analysis needs valid dates.
df = df.dropna(subset=["Order_Date"]).copy()

if df.empty:
    raise ValueError("No valid rows remain after cleaning the Order_Date column.")


# ============================================================
# 4. CALCULATED COLUMNS
# ============================================================

if "Revenue" in df.columns:
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0)
else:
    df["Revenue"] = df["Quantity"] * df["Unit_Price"]

# Keep discount between 0% and 100% to avoid nonsensical net sales.
df["Discount"] = df["Discount"].clip(lower=0, upper=100)

df["Net_Sales"] = df["Revenue"] * (1 - df["Discount"] / 100)

df["Profit"] = df["Net_Sales"] - (df["Quantity"] * df["Cost"])

df["Profit_Margin"] = np.where(
    df["Net_Sales"].ne(0),
    df["Profit"] / df["Net_Sales"] * 100,
    0,
)


# ============================================================
# 5. DATE FEATURES
# ============================================================

df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
df["Quarter"] = df["Order_Date"].dt.quarter


# ============================================================
# 6. BASIC KPIs
# ============================================================

total_sales = df["Net_Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
total_products = df["Product"].nunique()
average_order_value = total_sales / total_orders if total_orders else 0
profit_margin = total_profit / total_sales * 100 if total_sales else 0

print("\n" + "=" * 55)
print("E-COMMERCE KPIs")
print("=" * 55)
print(f"Total Sales:        {total_sales:,.2f}")
print(f"Total Profit:       {total_profit:,.2f}")
print(f"Total Orders:       {total_orders:,}")
print(f"Total Customers:    {total_customers:,}")
print(f"Total Products:     {total_products:,}")
print(f"Average Order Value:{average_order_value:,.2f}")
print(f"Profit Margin:      {profit_margin:.2f}%")


# ============================================================
# 7. ANALYSIS TABLES
# ============================================================

monthly_sales = (
    df.groupby(df["Order_Date"].dt.to_period("M"), observed=True)["Net_Sales"]
    .sum()
    .rename_axis("Order_Month")
    .reset_index()
)
monthly_sales["Order_Month"] = monthly_sales["Order_Month"].astype(str)

monthly_profit = (
    df.groupby(df["Order_Date"].dt.to_period("M"), observed=True)["Profit"]
    .sum()
    .rename_axis("Order_Month")
    .reset_index()
)
monthly_profit["Order_Month"] = monthly_profit["Order_Month"].astype(str)

category_analysis = (
    df.groupby("Category", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Profit=("Profit", "sum"),
         Quantity=("Quantity", "sum"), Orders=("Order_ID", "nunique"))
    .sort_values("Sales", ascending=False)
)

subcategory_analysis = (
    df.groupby("Sub_Category", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Profit=("Profit", "sum"),
         Quantity=("Quantity", "sum"))
    .sort_values("Sales", ascending=False)
)

top_products = (
    df.groupby("Product", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Profit=("Profit", "sum"),
         Quantity=("Quantity", "sum"))
    .sort_values("Sales", ascending=False)
    .head(10)
)

low_products = (
    df.groupby("Product", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Profit=("Profit", "sum"))
    .sort_values("Sales", ascending=True)
    .head(10)
)

state_analysis = (
    df.groupby("State", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Profit=("Profit", "sum"),
         Orders=("Order_ID", "nunique"), Customers=("Customer_ID", "nunique"))
    .sort_values("Sales", ascending=False)
)

city_analysis = (
    df.groupby("City", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Profit=("Profit", "sum"),
         Orders=("Order_ID", "nunique"))
    .sort_values("Sales", ascending=False)
    .head(20)
)

payment_analysis = (
    df.groupby("Payment_Mode", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Orders=("Order_ID", "nunique"))
    .sort_values("Sales", ascending=False)
)

gender_analysis = (
    df.groupby("Gender", observed=True)
    .agg(Sales=("Net_Sales", "sum"), Orders=("Order_ID", "nunique"),
         Customers=("Customer_ID", "nunique"))
)

status_analysis = (
    df.groupby("Order_Status", observed=True)
    .agg(Orders=("Order_ID", "nunique"), Sales=("Net_Sales", "sum"))
    .sort_values("Orders", ascending=False)
)


# ============================================================
# 8. CUSTOMER / RFM ANALYSIS
# ============================================================

customer_analysis = (
    df.groupby("Customer_ID", observed=True)
    .agg(
        Orders=("Order_ID", "nunique"),
        Total_Spending=("Net_Sales", "sum"),
        Average_Order_Value=("Net_Sales", "mean"),
        Total_Quantity=("Quantity", "sum"),
        Last_Purchase=("Order_Date", "max"),
        First_Purchase=("Order_Date", "min"),
    )
)

analysis_date = df["Order_Date"].max()
customer_analysis["Recency"] = (
    analysis_date - customer_analysis["Last_Purchase"]
).dt.days
customer_analysis["Frequency"] = customer_analysis["Orders"]
customer_analysis["Monetary"] = customer_analysis["Total_Spending"]

# Robust RFM scoring: works with fewer than five customers too.
def qcut_score(series: pd.Series, reverse: bool = False) -> pd.Series:
    n = len(series)
    if n == 0:
        return pd.Series(index=series.index, dtype="int64")
    if n == 1:
        return pd.Series(1, index=series.index, dtype="int64")
    bins = min(5, n)
    ranks = series.rank(method="first")
    labels = list(range(bins, 0, -1)) if reverse else list(range(1, bins + 1))
    return pd.qcut(ranks, q=bins, labels=labels).astype(int)


customer_analysis["R_Score"] = qcut_score(customer_analysis["Recency"], reverse=True)
customer_analysis["F_Score"] = qcut_score(customer_analysis["Frequency"])
customer_analysis["M_Score"] = qcut_score(customer_analysis["Monetary"])
customer_analysis["RFM_Score"] = (
    customer_analysis["R_Score"]
    + customer_analysis["F_Score"]
    + customer_analysis["M_Score"]
)


def customer_segment(score: int) -> str:
    if score >= 13:
        return "Champions"
    if score >= 10:
        return "Loyal Customers"
    if score >= 8:
        return "Potential Loyalists"
    if score >= 5:
        return "At Risk"
    return "Dormant"


customer_analysis["Segment"] = customer_analysis["RFM_Score"].apply(customer_segment)

segment_analysis = (
    customer_analysis.groupby("Segment", observed=True)
    .agg(Customers=("Segment", "count"), Revenue=("Total_Spending", "sum"),
         Average_Spending=("Total_Spending", "mean"), Average_Orders=("Orders", "mean"))
    .sort_values("Revenue", ascending=False)
)

orders_per_customer = df.groupby("Customer_ID", observed=True)["Order_ID"].nunique()
new_customers = int((orders_per_customer == 1).sum())
returning_customers = int((orders_per_customer > 1).sum())
repeat_purchase_rate = (
    returning_customers / total_customers * 100 if total_customers else 0
)

top_customers = customer_analysis.sort_values("Total_Spending", ascending=False).head(10)
at_risk_customers = customer_analysis[
    customer_analysis["Segment"] == "At Risk"
].sort_values("Total_Spending", ascending=False)
champion_customers = customer_analysis[
    customer_analysis["Segment"] == "Champions"
].sort_values("Total_Spending", ascending=False)


# ============================================================
# 9. PRINT MAIN OUTPUT
# ============================================================

print("\nMonthly Sales:\n", monthly_sales.to_string(index=False))
print("\nCategory Analysis:\n", category_analysis.to_string())
print("\nTop 10 Products:\n", top_products.to_string())
print("\nState Analysis:\n", state_analysis.to_string())
print("\nPayment Mode Analysis:\n", payment_analysis.to_string())
print("\nCustomer Segments:\n", customer_analysis[[
    "Orders", "Total_Spending", "Recency", "Frequency", "Monetary", "RFM_Score", "Segment"
]].head(20).to_string())
print("\nSegment Analysis:\n", segment_analysis.to_string())
print(f"\nNew Customers: {new_customers}")
print(f"Returning Customers: {returning_customers}")
print(f"Repeat Purchase Rate: {repeat_purchase_rate:.2f}%")


# ============================================================
# 10. SAVE RESULTS
# ============================================================

output_dir = Path("ecommerce_analysis_output")
output_dir.mkdir(exist_ok=True)

df.to_csv(output_dir / "cleaned_ecommerce_sales.csv", index=False)
customer_analysis.to_csv(output_dir / "customer_analysis.csv")
customer_analysis[[
    "Orders", "Total_Spending", "Recency", "Frequency", "Monetary", "RFM_Score", "Segment"
]].to_csv(output_dir / "customer_rfm_segmentation.csv")
category_analysis.to_csv(output_dir / "category_analysis.csv")
monthly_sales.to_csv(output_dir / "monthly_sales.csv", index=False)
monthly_profit.to_csv(output_dir / "monthly_profit.csv", index=False)
segment_analysis.to_csv(output_dir / "segment_analysis.csv")


# ============================================================
# 11. CHARTS
# ============================================================

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales["Order_Month"], monthly_sales["Net_Sales"], marker="o")
plt.title("Monthly E-Commerce Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "monthly_sales.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(category_analysis.index.astype(str), category_analysis["Sales"])
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "sales_by_category.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
plt.barh(top_products.index.astype(str)[::-1], top_products["Sales"].to_numpy()[::-1])
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(output_dir / "top_products.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
top_states = state_analysis.head(10)
plt.bar(top_states.index.astype(str), top_states["Sales"])
plt.title("Top 10 States by Sales")
plt.xlabel("State")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "top_states.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 8))
plt.pie(payment_analysis["Sales"], labels=payment_analysis.index.astype(str), autopct="%1.1f%%")
plt.title("Sales by Payment Mode")
plt.tight_layout()
plt.savefig(output_dir / "payment_mode.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
segment_counts = customer_analysis["Segment"].value_counts()
plt.bar(segment_counts.index.astype(str), segment_counts.values)
plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output_dir / "customer_segments.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(category_analysis.index.astype(str), category_analysis["Profit"])
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "profit_by_category.png", dpi=150)
plt.close()


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("E-COMMERCE ANALYTICS COMPLETED SUCCESSFULLY")
print("=" * 55)
print(f"Total Sales:         {total_sales:,.2f}")
print(f"Total Profit:        {total_profit:,.2f}")
print(f"Total Orders:        {total_orders:,}")
print(f"Total Customers:     {total_customers:,}")
print(f"Average Order Value: {average_order_value:,.2f}")
print(f"Profit Margin:       {profit_margin:.2f}%")
print(f"Repeat Purchase Rate:{repeat_purchase_rate:.2f}%")
print(f"Top Category:        {category_analysis.index[0] if len(category_analysis) else 'N/A'}")
print(f"Top Product:         {top_products.index[0] if len(top_products) else 'N/A'}")
print(f"Top State:           {state_analysis.index[0] if len(state_analysis) else 'N/A'}")
print(f"Top Customer:        {top_customers.index[0] if len(top_customers) else 'N/A'}")
print(f"\nResults saved in: {output_dir.resolve()}")
