import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"


def as_lines(text: str) -> list[str]:
    text = text.strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.split("\n")]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": as_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": as_lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_ROOT_HELPER = """
from pathlib import Path


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, current.parent):
        if (candidate / "orders.csv").exists() and (candidate / "products.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root that contains the Instacart CSV files.")


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"
QUALITY_DIR = OUTPUT_DIR / "quality"

for path in (OUTPUT_DIR, ANALYSIS_DIR, FIGURE_DIR, QUALITY_DIR):
    path.mkdir(parents=True, exist_ok=True)
"""


NOTEBOOKS = {
    "01_data_ingestion_and_quality_checks.ipynb": notebook([
        markdown_cell(
            """
# Step 1: Data Ingestion And Quality Checks

This notebook establishes the project baseline for the Instacart market basket analysis.

It covers:
- loading the three raw CSV files
- parsing timestamps and creating time features
- auditing missing values, duplicates, and row counts
- saving a lightweight data audit for downstream notebooks
"""
        ),
        code_cell(
            """
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.2f}")
"""
        ),
        code_cell(COMMON_ROOT_HELPER),
        code_cell(
            """
orders_path = PROJECT_ROOT / "orders.csv"
products_path = PROJECT_ROOT / "products.csv"
order_products_path = PROJECT_ROOT / "all_order_products.csv"

orders = pd.read_csv(orders_path, parse_dates=["timestamp"])
products = pd.read_csv(products_path)

orders["order_date"] = orders["timestamp"].dt.date
orders["order_hour"] = orders["timestamp"].dt.hour
orders["order_day_name"] = orders["timestamp"].dt.day_name()
orders["order_day_of_week"] = orders["timestamp"].dt.dayofweek
orders["order_month"] = orders["timestamp"].dt.to_period("M").astype(str)

orders.head()
"""
        ),
        code_cell(
            """
sample_order_products = pd.read_csv(order_products_path, nrows=100_000)

row_counts = {
    "orders": len(orders),
    "products": len(products),
    "order_products_sample": len(sample_order_products),
}

missing_summary = pd.DataFrame(
    {
        "orders_missing": orders.isna().sum(),
        "products_missing": products.isna().sum(),
    }
).fillna(0).astype(int)

duplicate_summary = pd.DataFrame(
    [
        {"table": "orders", "duplicate_rows": int(orders.duplicated().sum())},
        {"table": "products", "duplicate_rows": int(products.duplicated().sum())},
        {"table": "order_products_sample", "duplicate_rows": int(sample_order_products.duplicated().sum())},
    ]
)

audit_summary = pd.DataFrame(
    [
        {"metric": "orders_row_count", "value": len(orders)},
        {"metric": "products_row_count", "value": len(products)},
        {"metric": "unique_users", "value": orders["user_id"].nunique()},
        {"metric": "unique_orders", "value": orders["order_id"].nunique()},
        {"metric": "unique_products", "value": products["product_id"].nunique()},
        {"metric": "min_timestamp", "value": str(orders["timestamp"].min())},
        {"metric": "max_timestamp", "value": str(orders["timestamp"].max())},
    ]
)

print(row_counts)
missing_summary
"""
        ),
        code_cell(
            """
duplicate_summary
"""
        ),
        code_cell(
            """
audit_summary.to_csv(QUALITY_DIR / "data_audit_summary.csv", index=False)
missing_summary.to_csv(QUALITY_DIR / "missing_value_summary.csv")
duplicate_summary.to_csv(QUALITY_DIR / "duplicate_summary.csv", index=False)

audit_summary
"""
        ),
        markdown_cell(
            """
## Step 1 Outcome

After running this notebook you will have:
- validated the basic schema of each file
- generated reusable time-based features from `orders.csv`
- exported quality summaries into `outputs/quality/`
"""
        ),
    ]),
    "02_demand_patterns_and_eda.ipynb": notebook([
        markdown_cell(
            """
# Step 2: Demand Patterns And Exploratory Analysis

This notebook focuses on when customers place orders.

It builds the descriptive layer for the project by analyzing:
- hourly demand
- weekday demand
- a weekday-by-hour demand heatmap
- the busiest and slowest operating windows
"""
        ),
        code_cell(
            """
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
"""
        ),
        code_cell(COMMON_ROOT_HELPER),
        code_cell(
            """
orders = pd.read_csv(PROJECT_ROOT / "orders.csv", parse_dates=["timestamp"])

orders["order_hour"] = orders["timestamp"].dt.hour
orders["order_day_name"] = orders["timestamp"].dt.day_name()
orders["order_day_of_week"] = orders["timestamp"].dt.dayofweek
orders["order_date"] = orders["timestamp"].dt.date

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
"""
        ),
        code_cell(
            """
hourly_profile = (
    orders.groupby("order_hour")
    .size()
    .rename("order_count")
    .reset_index()
)

weekday_profile = (
    orders.groupby(["order_day_of_week", "order_day_name"])
    .size()
    .rename("order_count")
    .reset_index()
    .sort_values("order_day_of_week")
)

heatmap_profile = (
    orders.groupby(["order_day_name", "order_hour"])
    .size()
    .rename("order_count")
    .reset_index()
)

heatmap_table = (
    heatmap_profile.pivot(index="order_day_name", columns="order_hour", values="order_count")
    .reindex(weekday_order)
)

hourly_profile.head()
"""
        ),
        code_cell(
            """
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

sns.lineplot(data=hourly_profile, x="order_hour", y="order_count", marker="o", ax=axes[0])
axes[0].set_title("Order Volume By Hour")
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Orders")

sns.barplot(data=weekday_profile, x="order_day_name", y="order_count", order=weekday_order, ax=axes[1])
axes[1].set_title("Order Volume By Weekday")
axes[1].set_xlabel("Weekday")
axes[1].set_ylabel("Orders")
axes[1].tick_params(axis="x", rotation=20)

sns.heatmap(heatmap_table, cmap="Blues", ax=axes[2])
axes[2].set_title("Demand Heatmap: Weekday By Hour")
axes[2].set_xlabel("Hour of day")
axes[2].set_ylabel("Weekday")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "demand_patterns.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            """
busiest_hour = hourly_profile.loc[hourly_profile["order_count"].idxmax()]
slowest_hour = hourly_profile.loc[hourly_profile["order_count"].idxmin()]
busiest_day = weekday_profile.loc[weekday_profile["order_count"].idxmax()]
slowest_day = weekday_profile.loc[weekday_profile["order_count"].idxmin()]

summary = pd.DataFrame(
    [
        {"metric": "busiest_hour", "value": int(busiest_hour["order_hour"]), "order_count": int(busiest_hour["order_count"])},
        {"metric": "slowest_hour", "value": int(slowest_hour["order_hour"]), "order_count": int(slowest_hour["order_count"])},
        {"metric": "busiest_day", "value": busiest_day["order_day_name"], "order_count": int(busiest_day["order_count"])},
        {"metric": "slowest_day", "value": slowest_day["order_day_name"], "order_count": int(slowest_day["order_count"])},
    ]
)

hourly_profile.to_csv(ANALYSIS_DIR / "hourly_profile.csv", index=False)
weekday_profile.to_csv(ANALYSIS_DIR / "weekday_profile.csv", index=False)
heatmap_profile.to_csv(ANALYSIS_DIR / "weekday_hour_profile.csv", index=False)
summary.to_csv(ANALYSIS_DIR / "demand_summary.csv", index=False)

summary
"""
        ),
        markdown_cell(
            """
## Step 2 Outcome

After running this notebook you will have:
- visual evidence of peak and low-demand periods
- CSV summaries saved into `outputs/analysis/`
- a figure exported to `outputs/figures/demand_patterns.png`
"""
        ),
    ]),
    "03_basket_analysis_and_cross_sell.ipynb": notebook([
        markdown_cell(
            """
# Step 3: Basket Analysis And Cross-Sell Rules

This notebook analyzes what customers buy together.

It includes:
- top product frequency analysis using chunked reads
- a manageable sampled transaction set for basket mining
- association rule generation for cross-sell ideas

Install `mlxtend` in your notebook environment if it is not already available.
"""
        ),
        code_cell(
            """
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as exc:
    raise ImportError("Install mlxtend first with `%pip install mlxtend`.") from exc
"""
        ),
        code_cell(COMMON_ROOT_HELPER),
        code_cell(
            """
orders = pd.read_csv(PROJECT_ROOT / "orders.csv")
products = pd.read_csv(PROJECT_ROOT / "products.csv")

sample_order_ids = set(
    orders["order_id"].sample(n=8_000, random_state=42)
)

top_product_counts = None
sampled_order_products = []

for chunk in pd.read_csv(PROJECT_ROOT / "all_order_products.csv", chunksize=1_000_000):
    counts = chunk["product_id"].value_counts()
    top_product_counts = counts if top_product_counts is None else top_product_counts.add(counts, fill_value=0)

    sampled_chunk = chunk[chunk["order_id"].isin(sample_order_ids)]
    if not sampled_chunk.empty:
        sampled_order_products.append(sampled_chunk)

sampled_order_products = pd.concat(sampled_order_products, ignore_index=True)
top_products = (
    top_product_counts.sort_values(ascending=False)
    .head(25)
    .rename_axis("product_id")
    .reset_index(name="order_count")
    .merge(products, on="product_id", how="left")
)

top_products.head()
"""
        ),
        code_cell(
            """
sns.barplot(
    data=top_products.head(15),
    y="product_name",
    x="order_count",
    orient="h",
)
plt.title("Top 15 Products By Purchase Frequency")
plt.xlabel("Order count")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "top_products.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            """
analysis_products = set(top_products.head(20)["product_id"])

basket_sample = (
    sampled_order_products[sampled_order_products["product_id"].isin(analysis_products)]
    .merge(products[["product_id", "product_name"]], on="product_id", how="left")
)

basket_matrix = pd.crosstab(basket_sample["order_id"], basket_sample["product_name"]).astype(bool)

frequent_itemsets = apriori(basket_matrix, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.05)

rules = rules.sort_values(["lift", "confidence"], ascending=False)
rules["antecedents"] = rules["antecedents"].apply(lambda items: ", ".join(sorted(items)))
rules["consequents"] = rules["consequents"].apply(lambda items: ", ".join(sorted(items)))

rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
rules.head(15)
"""
        ),
        code_cell(
            """
top_products.to_csv(ANALYSIS_DIR / "top_products.csv", index=False)
rules.to_csv(ANALYSIS_DIR / "association_rules.csv", index=False)

rules.head(20)
"""
        ),
        markdown_cell(
            """
## Step 3 Outcome

After running this notebook you will have:
- a ranked list of the most frequently purchased products
- a sampled set of association rules for cross-sell opportunities
- outputs saved into `outputs/analysis/association_rules.csv`
"""
        ),
    ]),
    "04_forecasting_and_staffing_model.ipynb": notebook([
        markdown_cell(
            """
# Step 4: Forecasting And Staffing Model

This notebook moves from predictive analytics to prescriptive analytics.

It:
- builds an hourly demand table
- fits a multiplicative-style regression using log demand
- estimates mean and variability by weekday and hour
- converts demand forecasts into staffing recommendations with a newsvendor-style rule
"""
        ),
        code_cell(
            """
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

try:
    import scipy.stats as stats
    import statsmodels.formula.api as smf
except ImportError as exc:
    raise ImportError("Install scipy and statsmodels before running this notebook.") from exc
"""
        ),
        code_cell(COMMON_ROOT_HELPER),
        code_cell(
            """
orders = pd.read_csv(PROJECT_ROOT / "orders.csv", parse_dates=["timestamp"])

orders["order_date"] = orders["timestamp"].dt.date
orders["order_hour"] = orders["timestamp"].dt.hour
orders["day_name"] = orders["timestamp"].dt.day_name()
orders["day_of_week"] = orders["timestamp"].dt.dayofweek

hourly_demand = (
    orders.groupby(["order_date", "day_name", "day_of_week", "order_hour"])
    .size()
    .rename("order_count")
    .reset_index()
)

hourly_demand.head()
"""
        ),
        code_cell(
            """
model_data = hourly_demand.copy()
model_data["log_order_count"] = np.log1p(model_data["order_count"])

model = smf.ols("log_order_count ~ C(order_hour) + C(day_name)", data=model_data).fit()
print(model.summary())
"""
        ),
        code_cell(
            """
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

forecast_grid = pd.MultiIndex.from_product(
    [weekday_order, range(24)], names=["day_name", "order_hour"]
).to_frame(index=False)

forecast_grid["predicted_log_orders"] = model.predict(forecast_grid)
forecast_grid["expected_orders"] = np.expm1(forecast_grid["predicted_log_orders"]).clip(lower=0)

variability = (
    hourly_demand.groupby(["day_name", "order_hour"])["order_count"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "historical_mean", "std": "historical_std"})
)

staffing_plan = forecast_grid.merge(variability, on=["day_name", "order_hour"], how="left")
staffing_plan["historical_std"] = staffing_plan["historical_std"].fillna(0)
"""
        ),
        code_cell(
            """
orders_per_shopper_per_hour = 15
underage_cost = 45
overage_cost = 20

critical_fractile = underage_cost / (underage_cost + overage_cost)
z_value = stats.norm.ppf(critical_fractile)

staffing_plan["recommended_capacity_orders"] = (
    staffing_plan["expected_orders"] + z_value * staffing_plan["historical_std"]
).clip(lower=0)
staffing_plan["recommended_shoppers"] = np.ceil(
    staffing_plan["recommended_capacity_orders"] / orders_per_shopper_per_hour
).astype(int)

staffing_plan = staffing_plan.sort_values(["day_name", "order_hour"])
staffing_plan.head()
"""
        ),
        code_cell(
            """
heatmap_table = staffing_plan.pivot(index="day_name", columns="order_hour", values="recommended_shoppers")
heatmap_table = heatmap_table.reindex(weekday_order)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_table, cmap="YlOrRd")
plt.title("Recommended Shopper Staffing By Weekday And Hour")
plt.xlabel("Hour of day")
plt.ylabel("Weekday")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "staffing_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            """
staffing_plan.to_csv(ANALYSIS_DIR / "staffing_plan.csv", index=False)

peak_staffing = staffing_plan.sort_values("recommended_shoppers", ascending=False).head(10)
peak_staffing
"""
        ),
        markdown_cell(
            """
## Step 4 Outcome

After running this notebook you will have:
- a fitted demand model for weekday and hourly order volume
- a staffing plan based on configurable overage and underage costs
- a staffing heatmap saved into `outputs/figures/staffing_heatmap.png`
"""
        ),
    ]),
    "05_operational_recommendations.ipynb": notebook([
        markdown_cell(
            """
# Step 5: Operational Recommendations

This notebook consolidates the main findings from the prior notebooks into an executive-ready summary.

Run notebooks 2 through 4 first so the analysis files exist in `outputs/analysis/`.
"""
        ),
        code_cell(
            """
import pandas as pd
"""
        ),
        code_cell(COMMON_ROOT_HELPER),
        code_cell(
            """
demand_summary = pd.read_csv(ANALYSIS_DIR / "demand_summary.csv")
top_products = pd.read_csv(ANALYSIS_DIR / "top_products.csv")
association_rules = pd.read_csv(ANALYSIS_DIR / "association_rules.csv")
staffing_plan = pd.read_csv(ANALYSIS_DIR / "staffing_plan.csv")

demand_summary
"""
        ),
        code_cell(
            """
best_rule = association_rules.sort_values(["lift", "confidence"], ascending=False).iloc[0]
top_product = top_products.sort_values("order_count", ascending=False).iloc[0]
peak_staff = staffing_plan.sort_values("recommended_shoppers", ascending=False).iloc[0]

executive_summary = pd.DataFrame(
    [
        {
            "focus_area": "Demand timing",
            "finding": f"The busiest operating window is {demand_summary.loc[demand_summary['metric'] == 'busiest_day', 'value'].iloc[0]} and the peak hour is {demand_summary.loc[demand_summary['metric'] == 'busiest_hour', 'value'].iloc[0]}.",
            "recommended_action": "Prioritize staffing, picking capacity, and slot availability during the strongest demand windows.",
        },
        {
            "focus_area": "Top products",
            "finding": f"{top_product['product_name']} appears as the most frequently purchased product in the dataset slice used for basket analysis.",
            "recommended_action": "Protect availability for top-volume products and use them as anchor items in inventory planning.",
        },
        {
            "focus_area": "Cross-sell",
            "finding": f"A strong rule is '{best_rule['antecedents']}' -> '{best_rule['consequents']}' with lift {best_rule['lift']:.2f}.",
            "recommended_action": "Use the top rules for recommendation widgets, bundle prompts, and merchandising tests.",
        },
        {
            "focus_area": "Staffing",
            "finding": f"The highest suggested staffing level in the current model is {int(peak_staff['recommended_shoppers'])} shoppers at hour {int(peak_staff['order_hour'])} on {peak_staff['day_name']}.",
            "recommended_action": "Use the staffing table as the operational baseline, then recalibrate costs and shopper productivity assumptions with business input.",
        },
    ]
)

executive_summary
"""
        ),
        code_cell(
            """
executive_summary.to_csv(ANALYSIS_DIR / "executive_summary.csv", index=False)
executive_summary
"""
        ),
        markdown_cell(
            """
## Step 5 Outcome

After running this notebook you will have:
- a concise summary of demand, basket, and staffing insights
- a CSV export at `outputs/analysis/executive_summary.csv`
- a final notebook suitable for presentation prep and report writing
"""
        ),
    ]),
}


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for file_name, content in NOTEBOOKS.items():
        output_path = NOTEBOOK_DIR / file_name
        output_path.write_text(json.dumps(content, indent=2) + "\n")
        print(f"Wrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
