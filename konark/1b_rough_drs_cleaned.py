#!/usr/bin/env python
# coding: utf-8

"""
This script calculates a "Decreasing Returns to Scale" (DRS) metric for mutual funds
to analyze how this metric relates to fund size and investment style.

The methodology involves:
1.  Loading 2023 fund holdings and style data.
2.  Creating an expanded "universe" of all possible fund-stock pairs for each
    investment style (a Cartesian product).
3.  Calculating the "active weight" of each fund's position relative to a
    market-cap-weighted benchmark for its style.
4.  Computing a final DRS score for each fund based on its active weights.
5.  Visualizing the relationship between the DRS score, fund size, and style.
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

# Load fund style data from an RDS file
styles_raw = pyreadr.read_r(STYLES_PATH)[None]

# Filter styles data to include only the year 2023
styles_2023 = styles_raw[styles_raw["yyyymm"].between(202301, 202312)]

# Merge holdings with styles to link each stock to its fund's category
holdings_df = pd.merge(styles_2023, holdings_raw, on="wficn", how="inner")

# Engineer key features: market cap of the stock and dollar value of the fund's investment
holdings_df["market_cap"] = holdings_df["shrout_tfn"] * holdings_df["prc_tfn"]
holdings_df["fund_investment"] = holdings_df["shares"] * holdings_df["prc_tfn"]

# Drop rows where essential identifiers are missing
holdings_df = holdings_df.dropna(subset=["permno", "wficn", "morningstar_category"])
print(f"Initial data loaded for {holdings_df['wficn'].nunique()} unique funds.")


# === 3. Create the Style-Based Investment Universe ===
print("Creating the style-based investment universe...")
# Identify all unique funds within each style
funds_in_style = holdings_df[["morningstar_category", "wficn"]].drop_duplicates()

# Identify all unique stocks available within each style
stocks_in_style = holdings_df[["morningstar_category", "permno"]].drop_duplicates()

# Create the complete universe of all possible fund-stock pairs for each style
# This is a critical step for calculating active share against a common benchmark
universe = pd.merge(
    funds_in_style, stocks_in_style, on="morningstar_category", how="inner"
)
print(f"Universe created with {len(universe)} potential fund-stock pairs.")


# === 4. Merge Universe with Holdings and Calculate Weights ===
print("Calculating active weights...")
# To ensure a clean merge, create an aggregated holdings table with one row per fund-stock pair
# We use the median market cap over the year to get a stable value for each stock
holdings_agg = holdings_df.groupby(
    ["wficn", "permno", "morningstar_category"], as_index=False
).agg(fund_investment=("fund_investment", "sum"), market_cap=("market_cap", "median"))

# Merge the universe with the actual aggregated holdings
# A left merge ensures all possible pairs from the universe are kept
merged_df = pd.merge(
    universe, holdings_agg, on=["wficn", "permno", "morningstar_category"], how="left"
)

# For fund-stock pairs that don't exist in reality, fund_investment will be NaN. Fill with 0.
merged_df["fund_investment"] = merged_df["fund_investment"].fillna(0)

# Propagate market_cap values to all rows for a given stock, even if a fund doesn't own it
merged_df["market_cap"] = merged_df.groupby("permno")["market_cap"].transform(
    lambda x: x.ffill().bfill()
)
merged_df = merged_df.dropna(
    subset=["market_cap"]
)  # Drop any stocks that still have no cap data

# Calculate the weight of each stock within its fund's portfolio
fund_total_investment = merged_df.groupby("wficn")["fund_investment"].transform("sum")
merged_df["fund_weight_j"] = merged_df["fund_investment"] / fund_total_investment

# Calculate the benchmark weight (market cap weight) for each stock within the fund's universe
benchmark_total_cap = merged_df.groupby("wficn")["market_cap"].transform("sum")
merged_df["market_cap_weight_j"] = merged_df["market_cap"] / benchmark_total_cap

# Active weight is the fund's deviation from the benchmark
merged_df["active_weight_j"] = (
    merged_df["fund_weight_j"] - merged_df["market_cap_weight_j"]
)


# === 5. Calculate Fund-Level DRS Scores ===
print("Calculating fund-level DRS scores...")
# Calculate the components of the DRS formula
merged_df["active_weight_j_sq"] = merged_df["active_weight_j"] ** 2
merged_df["c_j"] = 2 / merged_df["market_cap"]  # Illiquidity penalty term
merged_df["drs_component"] = merged_df["active_weight_j_sq"] * merged_df["c_j"]

# Aggregate the components to calculate the final DRS score per fund
# This vectorized .agg() approach is significantly faster than .apply()
drs = merged_df.groupby(["wficn", "morningstar_category"], as_index=False).agg(
    drs_per_dollar=("drs_component", "sum"),
    fund_investment_total=("fund_investment", "sum"),
)

# Calculate the final scaled DRS metric
drs["drs_double"] = 2 * drs["drs_per_dollar"] * 2 * drs["fund_investment_total"]
print(drs["drs_double"].describe())


# === 6. Analyze and Visualize Results ===
print("Analyzing and visualizing results...")

# --- Analysis by Fund Size Quintile ---
# Bin funds into five groups (quintiles) based on their total size
drs["fund_bin"] = pd.qcut(
    drs["fund_investment_total"], q=5, labels=[f"Q{i + 1}" for i in range(5)]
)

# Generate and print summary statistics for DRS by fund size
size_summary = drs.groupby("fund_bin").agg(
    drs_double_mean=("drs_double", "mean"),
    drs_double_median=("drs_double", "median"),
    fund_size_avg=("fund_investment_total", "mean"),
    fund_count=("wficn", "count"),
)
print("\n--- DRS Summary by Fund Size Quintile ---")
print(size_summary)

# Plot DRS score distribution across fund size quintiles
plt.figure(figsize=(10, 6))
sns.boxplot(data=drs, x="fund_bin", y="drs_double")
plt.title("DRS Score by Fund Size Quintile")
plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
plt.ylabel("DRS Score")
plt.tight_layout()
plt.show()

# --- Analysis by Morningstar Category ---
# For a cleaner plot, select the top 10 largest styles by number of funds
top_styles = drs["morningstar_category"].value_counts().nlargest(10).index
drs_top_styles = drs[drs["morningstar_category"].isin(top_styles)]

# Plot DRS score distribution across top investment styles
plt.figure(figsize=(14, 7))
sns.boxplot(data=drs_top_styles, x="morningstar_category", y="drs_double")
plt.title("DRS Score by Morningstar Category (Top 10 Styles by Fund Count)")
plt.xlabel("Morningstar Category")
plt.ylabel("DRS Score")
plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
plt.tight_layout()
plt.show()
