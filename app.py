
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# 0. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="E-Commerce Profit Overview Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 E-Commerce Profit Decision Dashboard")
st.caption(
    "Superstore dataset | Exploratory Analytics • Customer Segmentation • "
    "Profit Prediction • Discount Scenario Analysis"
)

# ============================================================
# 1. LOAD DATA
# ============================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Order ID", "Customer ID", "Sales", "Quantity",
                           "Discount", "Profit", "Order Date"]).copy()

    return df


# Change this if the CSV is stored somewhere else.
DATA_PATH = "Sample - Superstore.csv"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Không tìm thấy file `{DATA_PATH}`. "
        "Hãy đặt file CSV cùng thư mục với app.py hoặc sửa biến DATA_PATH."
    )
    st.stop()

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================
def money(x):
    return f"${x:,.2f}"

def pct(x):
    return f"{x:.2f}%"

def make_discount_group(x):
    if x == 0:
        return "0%"
    elif x <= 0.10:
        return ">0–10%"
    elif x <= 0.20:
        return ">10–20%"
    elif x <= 0.30:
        return ">20–30%"
    else:
        return ">30%"

df["Discount Group"] = df["Discount"].apply(make_discount_group)

# Order-level data
order_df = (
    df.groupby("Order ID", as_index=False)
      .agg(
          Order_Sales=("Sales", "sum"),
          Order_Profit=("Profit", "sum"),
          Order_Quantity=("Quantity", "sum"),
          Avg_Discount=("Discount", "mean")
      )
)

order_df["Profit_Margin"] = (
    order_df["Order_Profit"] / order_df["Order_Sales"] * 100
)

# Monthly data
monthly_df = (
    df.set_index("Order Date")
      .resample("MS")
      .agg(
          Sales=("Sales", "sum"),
          Profit=("Profit", "sum"),
          Quantity=("Quantity", "sum"),
          Avg_Discount=("Discount", "mean")
      )
      .reset_index()
)

monthly_df["Profit_Margin"] = (
    monthly_df["Profit"] / monthly_df["Sales"] * 100
)

# Customer-level data
customer_df = (
    df.groupby("Customer ID", as_index=False)
      .agg(
          Avg_Discount=("Discount", "mean"),
          Total_Sales=("Sales", "sum"),
          Total_Profit=("Profit", "sum"),
          Order_Count=("Order ID", "nunique")
      )
)

customer_df["Profit_Margin"] = (
    customer_df["Total_Profit"] / customer_df["Total_Sales"] * 100
)

# ============================================================
# 3. RANDOM FOREST MODEL
# ============================================================
@st.cache_resource
def train_model(data):
    ml_df = data[
        [
            "Sales",
            "Quantity",
            "Discount",
            "Category",
            "Sub-Category",
            "Segment",
            "Region",
            "Ship Mode",
            "Profit"
        ]
    ].dropna().copy()

    X_raw = ml_df.drop(columns=["Profit"])
    y = ml_df["Profit"]

    X = pd.get_dummies(
        X_raw,
        columns=[
            "Category",
            "Sub-Category",
            "Segment",
            "Region",
            "Ship Mode"
        ],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    baseline_r2 = r2_score(y_test, baseline_pred)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, X.columns.tolist(), X_test, y_test, y_pred, rmse, mae, r2, baseline_r2, importance


(
    model,
    feature_columns,
    X_test,
    y_test,
    y_pred,
    rmse,
    mae,
    r2,
    baseline_r2,
    feature_importance
) = train_model(df)

# ============================================================
# 4. SIDEBAR
# ============================================================
st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio(
    "Display options",
    [
        "Overview",
        "Discount Analysis",
        "Customer Segmentation",
        "Profit Simulator",
        "Model Performance"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Note: The Profit Simulator is a decision-support tool based on the Random Forest model. The projected results do not account for the impact of price discounts."
)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":

    st.header("1. Executive Overview")
    st.write(
        "The dashboard answers the question: How is the business performing?**"
    )

    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_quantity = df["Quantity"].sum()
    aov = total_sales / total_orders
    profit_margin = total_profit / total_sales * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Total Sales", money(total_sales))
    c2.metric("Total Profit", money(total_profit))
    c3.metric("Profit Margin", pct(profit_margin))
    c4.metric("Orders", f"{total_orders:,}")
    c5.metric("Quantity", f"{total_quantity:,}")
    c6.metric("AOV", money(aov))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Sales")
        chart_sales = monthly_df.set_index("Order Date")[["Sales"]]
        st.line_chart(chart_sales)

    with col2:
        st.subheader("Monthly Profit")
        chart_profit = monthly_df.set_index("Order Date")[["Profit"]]
        st.line_chart(chart_profit)

    best_sales = monthly_df.loc[monthly_df["Sales"].idxmax()]
    best_profit = monthly_df.loc[monthly_df["Profit"].idxmax()]
    worst_profit = monthly_df.loc[monthly_df["Profit"].idxmin()]

    st.subheader("Key observations")

    obs1, obs2, obs3 = st.columns(3)

    obs1.info(
        f"**Sales peak**\n\n"
        f"{best_sales['Order Date'].strftime('%Y-%m')}: "
        f"{money(best_sales['Sales'])}"
    )

    obs2.success(
        f"**Profit peak**\n\n"
        f"{best_profit['Order Date'].strftime('%Y-%m')}: "
        f"{money(best_profit['Profit'])}"
    )

    obs3.warning(
        f"**Lowest monthly profit**\n\n"
        f"{worst_profit['Order Date'].strftime('%Y-%m')}: "
        f"{money(worst_profit['Profit'])}"
    )

# ============================================================
# PAGE 2 — DISCOUNT ANALYSIS
# ============================================================
elif page == "Discount Analysis":

    st.header("2. Discount Analysis")
    st.write(
        "This page focuses on the relationship between the discount level and profitability."
    )

    discount_summary = (
        df.groupby("Discount Group", sort=False)
          .agg(
              Orders=("Order ID", "nunique"),
              Quantity=("Quantity", "sum"),
              Sales=("Sales", "sum"),
              Profit=("Profit", "sum")
          )
          .reset_index()
    )

    discount_summary["AOV"] = (
        discount_summary["Sales"] / discount_summary["Orders"]
    )

    discount_summary["Profit Margin"] = (
        discount_summary["Profit"] / discount_summary["Sales"] * 100
    )

    order_discount = (
        order_df.assign(
            Discount_Group=order_df["Avg_Discount"].apply(make_discount_group)
        )
        .groupby("Discount_Group", sort=False)
        .agg(
            Orders=("Order_Sales", "count"),
            Sales=("Order_Sales", "sum"),
            Profit=("Order_Profit", "sum")
        )
        .reset_index()
    )

    selected_group = st.selectbox(
        "Chọn Discount Group",
        discount_summary["Discount Group"].tolist()
    )

    selected = discount_summary[
        discount_summary["Discount Group"] == selected_group
    ].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", f"{int(selected['Orders']):,}")
    c2.metric("Sales", money(selected["Sales"]))
    c3.metric("Profit", money(selected["Profit"]))
    c4.metric("AOV", money(selected["AOV"]))
    c5.metric("Profit Margin", pct(selected["Profit Margin"]))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profit Margin by Discount")
        margin_chart = discount_summary.set_index("Discount Group")[["Profit Margin"]]
        st.bar_chart(margin_chart)

    with col2:
        st.subheader("Sales vs Profit")
        sales_profit_chart = discount_summary.set_index("Discount Group")[
            ["Sales", "Profit"]
        ]
        st.bar_chart(sales_profit_chart)

    st.subheader("Discount KPI Table")

    display_table = discount_summary.copy()
    display_table["Sales"] = display_table["Sales"].map(lambda x: f"${x:,.2f}")
    display_table["Profit"] = display_table["Profit"].map(lambda x: f"${x:,.2f}")
    display_table["AOV"] = display_table["AOV"].map(lambda x: f"${x:,.2f}")
    display_table["Profit Margin"] = display_table["Profit Margin"].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.subheader("Decision signal")

    high_discount = discount_summary[
        discount_summary["Discount Group"] == ">30%"
    ].iloc[0]

    if high_discount["Profit"] < 0:
    st.markdown(
        f"""
        <div style="
            background-color: #fffde7;
            color: #8a6500;
            padding: 16px;
            border-radius: 8px;
            font-size: 16px;
            line-height: 1.6;
        ">
            Discount exceeding 30% generated Sales of
            {money(high_discount['Sales'])}, but Profit was negative at
            {money(high_discount['Profit'])}, with a Profit Margin of
            {pct(high_discount['Profit Margin'])}.
            This is a signal to review discount policies carefully.
        </div>
        """,
        unsafe_allow_html=True
    )

    corr_discount_profit = order_df["Avg_Discount"].corr(
        order_df["Order_Profit"]
    )
    corr_discount_quantity = order_df["Avg_Discount"].corr(
        order_df["Order_Quantity"]
    )

    c1, c2 = st.columns(2)
    c1.metric("Discount ↔ Profit correlation", f"{corr_discount_profit:.4f}")
    c2.metric("Discount ↔ Quantity correlation", f"{corr_discount_quantity:.4f}")

    st.caption(
        "Correlation only reflects a relationship within observed data; "
        "it does not prove a causal relationship."
    )

# ============================================================
# PAGE 3 — CUSTOMER SEGMENTATION
# ============================================================
elif page == "Customer Segmentation":

    st.header("3. Customer Segmentation")
    st.write(
        "Segment customers to identify high-value groups, groups with high discount levels, and groups with low profitability."
    )

    scaler = StandardScaler()

    cluster_features = [
        "Avg_Discount",
        "Total_Sales",
        "Total_Profit",
        "Order_Count"
    ]

    scaled_features = scaler.fit_transform(
        customer_df[cluster_features]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    customer_clustered = customer_df.copy()
    customer_clustered["Cluster"] = kmeans.fit_predict(scaled_features)

    cluster_summary = (
        customer_clustered.groupby("Cluster")
        .agg(
            Customers=("Customer ID", "count"),
            Avg_Discount=("Avg_Discount", "mean"),
            Total_Sales=("Total_Sales", "mean"),
            Total_Profit=("Total_Profit", "mean"),
            Avg_Order_Count=("Order_Count", "mean")
        )
        .reset_index()
    )

    cluster_summary["Estimated_Margin"] = (
        cluster_summary["Total_Profit"] /
        cluster_summary["Total_Sales"] * 100
    )

    st.subheader("Cluster Summary")
    st.dataframe(
        cluster_summary.round(2),
        use_container_width=True,
        hide_index=True
    )

    selected_cluster = st.selectbox(
        "Choose Cluster",
        sorted(customer_clustered["Cluster"].unique())
    )

    selected_customers = customer_clustered[
        customer_clustered["Cluster"] == selected_cluster
    ]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(selected_customers):,}")
    c2.metric(
        "Avg Discount",
        pct(selected_customers["Avg_Discount"].mean() * 100)
    )
    c3.metric(
        "Avg Sales",
        money(selected_customers["Total_Sales"].mean())
    )
    c4.metric(
        "Avg Profit",
        money(selected_customers["Total_Profit"].mean())
    )

    st.subheader("Customer clusters: Discount vs Profit")

    scatter_df = customer_clustered[
        ["Avg_Discount", "Total_Profit", "Cluster", "Total_Sales"]
    ].copy()

    scatter_df["Cluster"] = scatter_df["Cluster"].astype(str)

    st.scatter_chart(
        scatter_df,
        x="Avg_Discount",
        y="Total_Profit",
        color="Cluster",
        size="Total_Sales"
    )

    st.caption(
        "Each point represents a customer. The size of the point reflects Total Sales."
    )

    st.subheader("Pareto: Customer Profit Concentration")

    pareto = customer_df.sort_values(
        "Total_Profit", ascending=False
    ).reset_index(drop=True)

    pareto["Cumulative_Profit"] = pareto["Total_Profit"].cumsum()
    total_customer_profit = pareto["Total_Profit"].sum()
    pareto["Cumulative_Profit_Pct"] = (
        pareto["Cumulative_Profit"] /
        total_customer_profit * 100
    )

    top_20_n = max(1, int(len(pareto) * 0.20))
    top_20_pct = pareto.loc[
        top_20_n - 1, "Cumulative_Profit_Pct"
    ]

    st.metric(
        "Profit contributed by top 20% customers",
        f"{top_20_pct:.2f}%"
    )

    pareto_chart = pareto[["Cumulative_Profit_Pct"]].copy()
    pareto_chart.index = np.arange(1, len(pareto_chart) + 1)
    st.line_chart(pareto_chart)

    st.caption(
        "The Pareto result describes the concentration of profits within a data sample;"
        "it should not be interpreted as a universal 80/20 rule."
    )

# ============================================================
# PAGE 4 — PROFIT SIMULATOR
# ============================================================
elif page == "Profit Simulator":

    st.header("4. Profit Prediction Simulator")
    st.write(
        "Input a trading scenario for the Random Forest model to estimate profit. "
        "This tool facilitates what-if analysis and supports decision-making."
    )

    st.warning(
        "⚠️ This is a model forecast, not the causal impact of the discount. "
        "Specially, since Sales is an input variable "
        "the results align better with scenario analysis than with a simulation of pre-transaction profits"
    )

    col1, col2 = st.columns(2)

    with col1:
        sales_input = st.number_input(
            "Sales ($)",
            min_value=0.0,
            value=500.0,
            step=50.0
        )

        quantity_input = st.number_input(
            "Quantity",
            min_value=1,
            value=3,
            step=1
        )

        discount_input = st.slider(
            "Discount",
            min_value=0.0,
            max_value=0.80,
            value=0.10,
            step=0.01,
            format="%.2f"
        )

        category_input = st.selectbox(
            "Category",
            sorted(df["Category"].dropna().unique())
        )

        subcategory_options = sorted(
            df[df["Category"] == category_input]["Sub-Category"]
            .dropna()
            .unique()
        )

        subcategory_input = st.selectbox(
            "Sub-Category",
            subcategory_options
        )

    with col2:
        segment_input = st.selectbox(
            "Segment",
            sorted(df["Segment"].dropna().unique())
        )

        region_input = st.selectbox(
            "Region",
            sorted(df["Region"].dropna().unique())
        )

        ship_mode_input = st.selectbox(
            "Ship Mode",
            sorted(df["Ship Mode"].dropna().unique())
        )

    def prepare_input(
        sales,
        quantity,
        discount,
        category,
        subcategory,
        segment,
        region,
        ship_mode
    ):
        input_df = pd.DataFrame([{
            "Sales": sales,
            "Quantity": quantity,
            "Discount": discount,
            "Category": category,
            "Sub-Category": subcategory,
            "Segment": segment,
            "Region": region,
            "Ship Mode": ship_mode
        }])

        input_encoded = pd.get_dummies(
            input_df,
            columns=[
                "Category",
                "Sub-Category",
                "Segment",
                "Region",
                "Ship Mode"
            ],
            drop_first=True
        )

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        return input_encoded

    scenario_input = prepare_input(
        sales_input,
        quantity_input,
        discount_input,
        category_input,
        subcategory_input,
        segment_input,
        region_input,
        ship_mode_input
    )

    predicted_profit = model.predict(scenario_input)[0]
    predicted_margin = (
        predicted_profit / sales_input * 100
        if sales_input > 0 else 0
    )

    st.divider()

    c1, c2 = st.columns(2)

    c1.metric(
        "Predicted Profit",
        money(predicted_profit)
    )

    c2.metric(
        "Predicted Profit Margin",
        pct(predicted_margin)
    )

    # --------------------------------------------------------
    # WHAT-IF DISCOUNT SCENARIO
    # --------------------------------------------------------
    st.subheader("What-if: Discount Scenario")

    scenario_discounts = [0.00, 0.10, 0.20, 0.30, 0.40]

    scenario_results = []

    for d in scenario_discounts:
        scenario_X = prepare_input(
            sales_input,
            quantity_input,
            d,
            category_input,
            subcategory_input,
            segment_input,
            region_input,
            ship_mode_input
        )

        p = model.predict(scenario_X)[0]
        margin = p / sales_input * 100 if sales_input > 0 else 0

        scenario_results.append({
            "Discount": f"{int(d * 100)}%",
            "Discount_Value": d,
            "Predicted Profit": p,
            "Predicted Margin": margin
        })

    scenario_df = pd.DataFrame(scenario_results)

    chart_df = scenario_df.set_index("Discount")[["Predicted Profit"]]
    st.line_chart(chart_df)

    display_scenario = scenario_df[
        ["Discount", "Predicted Profit", "Predicted Margin"]
    ].copy()

    display_scenario["Predicted Profit"] = display_scenario[
        "Predicted Profit"
    ].map(lambda x: f"${x:,.2f}")

    display_scenario["Predicted Margin"] = display_scenario[
        "Predicted Margin"
    ].map(lambda x: f"{x:.2f}%")

    st.dataframe(
        display_scenario,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The discount levels in the table are incorporated into the same scenario "
        "with Sales, Quantity and other attributes held constant. The result is a "
        "model-based estimate, not evidence of a casual effect."
    )

# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.header("5. Model Performance")
    st.write(
        "This page evaluates whether the model predicts Profit better than the baseline and identifies which variables contribute most to the prediction. "
            )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("R²", f"{r2:.4f}")
    c4.metric("Baseline R²", f"{baseline_r2:.4f}")

    st.divider()

    st.subheader("Actual vs Predicted Profit")

    actual_pred = pd.DataFrame({
        "Actual Profit": y_test.values,
        "Predicted Profit": y_pred
    })

    st.scatter_chart(
        actual_pred,
        x="Actual Profit",
        y="Predicted Profit"
    )

    st.caption(
        "If the data points lie close to the 'Actual = Predicted' diagonal line"
        "the forecasts align more closely with the actual values. The chart above is used for visual assessment."
    )

    st.subheader("Feature Importance")

    top_features = feature_importance.head(15).sort_values(
        "Importance",
        ascending=True
    )

    importance_chart = top_features.set_index("Feature")[["Importance"]]
    st.bar_chart(importance_chart)

    st.dataframe(
        feature_importance.head(15).round(4),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Feature Importance reflects the contribution of a variable to the Random Forest's predictions "
        "not the percentage of causal impact on Profit."
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "GVHD: TS. Nguyễn Thôn Dã | HV: Huỳnh Trúc Ngân | MSHV: C25611251"
)
