#!/usr/bin/env python
# coding: utf-8

"""
This script calculates a "Decreasing Returns to Scale" (DRS) metric for mutual funds
using TWO different benchmark definitions to analyze the relationship between fund
size, investment style, and active management. It also calculates Active Share for
both benchmark methodologies to compare them directly.

Benchmark 1 (Style-Based): Compares a fund's holdings to the market-cap
weighted portfolio of ALL stocks within its Morningstar Category.

Benchmark 2 (Holdings-Based): Compares a fund's holdings to the market-cap
weighted portfolio of ONLY the stocks it actually owns.
"""

# === 1. Import Libraries and Define Constants ===
import pyreadr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths for easy access
HOLDINGS_PATH = "../../data/funds/holdings_quarterly_us_de_active_mf_csv/2023.csv"
STYLES_PATH = "../../data/funds/indicative/morningstar_categories_monthly.RDS"


# === 2. Load, Merge, and Prepare Data ===
print("Loading and preparing data...")
# Load quarterly holdings data for US active mutual funds
holdings_raw = pd.read_csv(HOLDINGS_PATH)
holdings_raw["date"] = pd.to_datetime(holdings_raw["date"])
holdings_raw["yyyymm"] = holdings_raw["date"].dt.strftime("%Y%m").astype(int)
holdings_raw = holdings_raw[holdings_raw["yyyymm"] == holdings_raw["yyyymm"].max()]

# Load fund style data from an RDS file
styles_raw = pyreadr.read_r(STYLES_PATH)[None]

# NOTE: only use the latest yyyymm
styles_2023 = styles_raw[styles_raw["yyyymm"] == styles_raw["yyyymm"].max()]

# Merge holdings with styles to link each stock to its fund's category
holdings_df = pd.merge(styles_2023, holdings_raw, on="wficn", how="inner")
holdings_df.drop(columns=["yyyymm_x", "yyyymm_y"], inplace=True)

# Engineer key features: market cap of the stock and dollar value of the fund's investment
holdings_df["market_cap"] = holdings_df["shrout_tfn"] * holdings_df["prc_tfn"]
holdings_df["fund_investment"] = holdings_df["shares"] * holdings_df["prc_tfn"]

# Drop rows where essential identifiers are missing
holdings_df = holdings_df.dropna(subset=["permno", "wficn", "morningstar_category"])

print(f"Initial data loaded for {holdings_df['wficn'].nunique()} unique funds.")


# === 3. Create a Unified DataFrame with Both Benchmark Weights ===
print("\n--- Creating a unified DataFrame for all calculations ---")

# === 3a. Create the Style-Based Universe ===
print("Creating the style-based investment universe...")
funds_in_style = holdings_df[["morningstar_category", "wficn"]].drop_duplicates()
stocks_in_style = holdings_df[["morningstar_category", "permno"]].drop_duplicates()
universe = pd.merge(
    funds_in_style, stocks_in_style, on="morningstar_category", how="inner"
)

# === 3b. Aggregate actual holdings data ===

holdings_agg = holdings_df.groupby(
    ["wficn", "permno", "morningstar_category"], as_index=False
).agg(fund_investment=("fund_investment", "sum"), market_cap=("market_cap", "median"))

# === 3c. Create the main analytical DataFrame from the universe ===
# This starts with all possible fund-stock pairs
analytical_df = pd.merge(
    universe, holdings_agg, on=["wficn", "permno", "morningstar_category"], how="left"
)
analytical_df["fund_investment"] = analytical_df["fund_investment"].fillna(0)
analytical_df["market_cap"] = analytical_df.groupby("permno")["market_cap"].transform(
    lambda x: x.ffill().bfill()
)
analytical_df = analytical_df.dropna(subset=["market_cap"])

# === 3d. Calculate Fund Weights (same for both benchmarks) ===
print("Calculating fund weights...")
fund_total_investment = analytical_df.groupby("wficn")["fund_investment"].transform(
    "sum"
)
# Avoid division by zero for funds with no investments
fund_total_investment = fund_total_investment.replace(0, pd.NA)
analytical_df["fund_weight"] = analytical_df["fund_investment"] / fund_total_investment
del fund_total_investment

# === 3e. Calculate Benchmark 1 (Style-Based) Weights ===
print("Calculating Benchmark 1 (Style-Based) weights...")
benchmark_total_cap_style = analytical_df.groupby("wficn")["market_cap"].transform(
    "sum"
)

analytical_df["mcap_weight_style"] = (
    analytical_df["market_cap"] / benchmark_total_cap_style
)

# === 3f. Calculate Benchmark 2 (Holdings-Based) Weights and merge them in ===
print("Calculating Benchmark 2 (Holdings-Based) weights...")
# Calculate these weights *only* on the actual holdings
benchmark_total_cap_holdings = holdings_df.groupby("wficn")["market_cap"].transform(
    "sum"
)
holdings_df["mcap_weight_holdings"] = (
    holdings_df["market_cap"] / benchmark_total_cap_holdings
)

# Merge these new weights into the main analytical DataFrame
analytical_df = pd.merge(
    analytical_df,
    holdings_df[
        ["wficn", "permno", "mcap_weight_holdings"]
    ].drop_duplicates(),  # Use drop_duplicates to handle multiple yyyymm
    on=["wficn", "permno"],
    how="left",
)

# === 4. Calculate Active Weights and DRS Components ===
print("\n--- Calculating Active Weights and DRS Components for Both Benchmarks ---")
# Calculate active weights for both benchmarks
analytical_df["active_weight_style"] = (
    analytical_df["fund_weight"] - analytical_df["mcap_weight_style"]
).fillna(0)

# For the holdings benchmark, active weight is only meaningful for held stocks.
# fund_weight is 0 for non-held stocks, and mcap_weight_holdings is NaN.
# So, we can subtract, and the result for non-held stocks will be NaN, which we fill with 0.
analytical_df["active_weight_holdings"] = (
    analytical_df["fund_weight"] - analytical_df["mcap_weight_holdings"]
).fillna(0)

# Calculate DRS components for both
analytical_df["drs_comp_style"] = (analytical_df["active_weight_style"] ** 2) * (
    2 / analytical_df["market_cap"]
)
analytical_df["drs_comp_holdings"] = (analytical_df["active_weight_holdings"] ** 2) * (
    2 / analytical_df["market_cap"]
)


# === 5. Aggregate to Fund Level for DRS and Active Share ===
print("\n--- Aggregating to Fund Level ---")
# Group by fund and calculate all final metrics in one pass
final_drs = analytical_df.groupby(
    ["wficn", "morningstar_category"], as_index=False
).agg(
    # Sum DRS components
    drs_per_dollar_style=("drs_comp_style", "sum"),
    drs_per_dollar_holdings=("drs_comp_holdings", "sum"),
    # Calculate Active Share for both
    active_share_style=("active_weight_style", lambda x: x.abs().sum() / 2),
    active_share_holdings=("active_weight_holdings", lambda x: x.abs().sum() / 2),
    # Get total fund size
    fund_investment_total=("fund_investment", "sum"),
)

# Calculate the final scaled DRS scores. NOTE: changed here
final_drs["drs_double_style"] = (
    2 * final_drs["drs_per_dollar_style"] * final_drs["fund_investment_total"]
)
final_drs["drs_double_holdings"] = (
    2 * final_drs["drs_per_dollar_holdings"] * final_drs["fund_investment_total"]
)


# === 6. Analyze and Visualize Results ===
print("\n--- Combining and Visualizing Results ---")
# Bin funds into quintiles based on their total size
final_drs["fund_bin"] = pd.qcut(
    final_drs["fund_investment_total"], q=5, labels=[f"Q{i + 1}" for i in range(5)]
)

# --- 6a. Side-by-Side Comparison of Active Share ---

# === first of all, style-based active share is much higher

plt.figure(figsize=(12, 7))
# Reshape the data for easy plotting with seaborn
plot_data = final_drs.melt(
    id_vars=["fund_bin"],
    value_vars=["active_share_style", "active_share_holdings"],
    var_name="Benchmark Type",
    value_name="Active Share",
)
plot_data["Benchmark Type"] = plot_data["Benchmark Type"].map(
    {"active_share_style": "Style-Based", "active_share_holdings": "Holdings-Based"}
)

sns.boxplot(data=plot_data, x="fund_bin", y="Active Share", hue="Benchmark Type")
plt.title("Active Share by Fund Size and Benchmark Definition")
plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
plt.ylabel("Active Share")
plt.legend(title="Benchmark Type")
plt.tight_layout()
plt.show()

# --- 6b. Visualization for DRS Scores ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_drs, x="fund_bin", y="drs_double_style")
plt.title("DRS Score (Style-Based Benchmark) by Fund Size Quintile")
plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
plt.ylabel("DRS Score (Style-Based)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=final_drs, x="fund_bin", y="drs_double_holdings")
plt.title("DRS Score (Holdings-Based Benchmark) by Fund Size Quintile")
plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
plt.ylabel("DRS Score (Holdings-Based)")
plt.tight_layout()
plt.show()

# === 6c. Produce Final Summary Table ===
summary_table = final_drs.groupby("fund_bin").agg(
    # Active Share Summaries
    active_share_style_mean=("active_share_style", "mean"),
    active_share_holdings_mean=("active_share_holdings", "mean"),
    # DRS Summaries
    drs_style_mean=("drs_double_style", "mean"),
    drs_holdings_mean=("drs_double_holdings", "mean"),
    # Fund Info
    fund_size_avg=("fund_investment_total", "mean"),
    fund_count=("wficn", "count"),
)
print("\n--- Final Summary by Fund Size Quintile ---")
print(summary_table)
