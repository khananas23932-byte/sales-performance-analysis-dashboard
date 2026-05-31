# =========================================================
# AI POWERED BUSINESS ANALYTICS SYSTEM
# CREATED BY ANAS KHAN
# =========================================================

# INSTALL REQUIRED LIBRARIES FIRST
# pip install pandas numpy streamlit plotly scikit-learn xgboost openpyxl

# RUN COMMAND
# streamlit run sales.py

# =========================================================

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Business Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

h1,h2,h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("📊 AI-Powered Business Analytics System")
st.markdown("### Advanced Sales & Profit Analytics Dashboard")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "data.csv",
    encoding="latin1"
)

# =========================================================
# CLEAN DATA
# =========================================================

df.drop_duplicates(inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'])

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("📌 Dashboard Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

segment = st.sidebar.multiselect(
    "Select Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique()
)

# =========================================================
# FILTER DATA
# =========================================================

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment))
]

# =========================================================
# KPI SECTION
# =========================================================

total_sales = filtered_df["Sales"].sum()

total_profit = filtered_df["Profit"].sum()

total_orders = filtered_df["Order ID"].nunique()

total_customers = filtered_df["Customer ID"].nunique()

profit_margin = (total_profit / total_sales) * 100

avg_order_value = total_sales / total_orders

# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "💰 Total Sales",
    f"₹ {total_sales:,.0f}"
)

col2.metric(
    "📈 Total Profit",
    f"₹ {total_profit:,.0f}"
)

col3.metric(
    "🛒 Total Orders",
    total_orders
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "👥 Customers",
    total_customers
)

col5.metric(
    "📊 Profit Margin",
    f"{profit_margin:.2f}%"
)

col6.metric(
    "💵 Avg Order Value",
    f"₹ {avg_order_value:,.0f}"
)

st.divider()

# =========================================================
# SALES BY REGION
# =========================================================

sales_region = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig_region = px.bar(
    sales_region,
    x="Region",
    y="Sales",
    text_auto=True,
    title="📍 Sales by Region",
    color="Region"
)

st.plotly_chart(
    fig_region,
    use_container_width=True
)

# =========================================================
# SALES BY CATEGORY
# =========================================================

sales_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig_category = px.pie(
    sales_category,
    names="Category",
    values="Sales",
    title="📦 Sales by Category",
    hole=0.5
)

st.plotly_chart(
    fig_category,
    use_container_width=True
)

# =========================================================
# MONTHLY SALES TREND
# =========================================================

filtered_df["Month"] = (
    filtered_df["Order Date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    filtered_df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig_trend = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    markers=True,
    title="📈 Monthly Sales Trend"
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)

# =========================================================
# TOP PRODUCTS
# =========================================================

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_products = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="🏆 Top 10 Products",
    color="Sales"
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)

# =========================================================
# PROFIT BY CATEGORY
# =========================================================

profit_category = (
    filtered_df.groupby("Category")["Profit"]
    .sum()
    .reset_index()
)

fig_profit = px.bar(
    profit_category,
    x="Category",
    y="Profit",
    text_auto=True,
    title="💹 Profit by Category",
    color="Category"
)

st.plotly_chart(
    fig_profit,
    use_container_width=True
)

# =========================================================
# TOP CUSTOMERS
# =========================================================

top_customers = (
    filtered_df.groupby("Customer Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_customers = px.bar(
    top_customers,
    x="Sales",
    y="Customer Name",
    orientation="h",
    title="👑 Top Customers",
    color="Sales"
)

st.plotly_chart(
    fig_customers,
    use_container_width=True
)

# =========================================================
# MACHINE LEARNING FORECAST
# =========================================================

st.subheader("🤖 AI Sales Forecast")

forecast_df = monthly_sales.copy()

forecast_df["Month_Num"] = np.arange(len(forecast_df))

X = forecast_df[["Month_Num"]]
y = forecast_df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X)

forecast_df["Predicted Sales"] = predictions

score = r2_score(y, predictions)

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df["Month"],
        y=forecast_df["Sales"],
        mode='lines+markers',
        name='Actual Sales'
    )
)

fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df["Month"],
        y=forecast_df["Predicted Sales"],
        mode='lines+markers',
        name='Predicted Sales'
    )
)

fig_forecast.update_layout(
    title="📊 Sales Forecasting",
    xaxis_title="Month",
    yaxis_title="Sales"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

st.success(f"✅ AI Model Accuracy Score: {score:.2f}")

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.subheader("📌 Business Insights")

highest_region = sales_region.sort_values(
    by="Sales",
    ascending=False
).iloc[0]["Region"]

best_category = sales_category.sort_values(
    by="Sales",
    ascending=False
).iloc[0]["Category"]

top_customer = top_customers.iloc[0]["Customer Name"]

st.info(
    f"🏆 Highest Sales Region: {highest_region}"
)

st.info(
    f"📦 Best Performing Category: {best_category}"
)

st.info(
    f"👑 Top Customer: {top_customer}"
)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("📋 Dataset Preview")

st.dataframe(filtered_df)

# =========================================================
# DOWNLOAD OPTION
# =========================================================

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    "⬇ Download Filtered Data",
    csv,
    "filtered_data.csv",
    "text/csv"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    "### 🚀 Developed by Anas Khan"
)