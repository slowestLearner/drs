#!/usr/bin/env python
# coding: utf-8

"""
This script calculates a "Decreasing Returns to Scale" (DRS) metric for mutual funds
using TWO different benchmark definitions to analyze the relationship between fund
size, investment style, and active management.

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


# === Part A: Benchmark 1 (Style-Based) ===
print("\n--- Starting Calculation for Benchmark 1: Style-Based ---")

# === A-1. Create the Style-Based Investment Universe ===
print("A-1. Creating the style-based investment universe...")
funds_in_style = holdings_df[["morningstar_category", "wficn"]].drop_duplicates()
stocks_in_style = holdings_df[["morningstar_category", "permno"]].drop_duplicates()
universe = pd.merge(
    funds_in_style, stocks_in_style, on="morningstar_category", how="inner"
)
print(f"Universe created with {len(universe)} potential fund-stock pairs.")

# === A-2. Merge Universe with Holdings and Calculate Weights ===
print("A-2. Calculating active weights against style benchmark...")
holdings_agg = holdings_df.groupby(
    ["wficn", "permno", "morningstar_category"], as_index=False
).agg(fund_investment=("fund_investment", "sum"), market_cap=("market_cap", "median"))
merged_df_style = pd.merge(
    universe, holdings_agg, on=["wficn", "permno", "morningstar_category"], how="left"
)
merged_df_style["fund_investment"] = merged_df_style["fund_investment"].fillna(0)
merged_df_style["market_cap"] = merged_df_style.groupby("permno")[
    "market_cap"
].transform(lambda x: x.ffill().bfill())
merged_df_style = merged_df_style.dropna(subset=["market_cap"])

fund_total_investment_style = merged_df_style.groupby("wficn")[
    "fund_investment"
].transform("sum")
merged_df_style["fund_weight_j"] = (
    merged_df_style["fund_investment"] / fund_total_investment_style
)

benchmark_total_cap_style = merged_df_style.groupby("wficn")["market_cap"].transform(
    "sum"
)
merged_df_style["market_cap_weight_j"] = (
    merged_df_style["market_cap"] / benchmark_total_cap_style
)

merged_df_style["active_weight_j"] = (
    merged_df_style["fund_weight_j"] - merged_df_style["market_cap_weight_j"]
)

# === A-3. Calculate Fund-Level DRS Scores (Style-Based) ===
print("A-3. Calculating fund-level DRS scores (Style-Based)...")
merged_df_style["drs_component"] = (merged_df_style["active_weight_j"] ** 2) * (
    2 / merged_df_style["market_cap"]
)

drs_style = merged_df_style.groupby(
    ["wficn", "morningstar_category"], as_index=False
).agg(
    drs_per_dollar=("drs_component", "sum"),
    fund_investment_total=("fund_investment", "sum"),
)
drs_style["drs_double_style"] = (
    2 * drs_style["drs_per_dollar"] * 2 * drs_style["fund_investment_total"]
)


# === Part B: Benchmark 2 (Holdings-Based) ===
print("\n--- Starting Calculation for Benchmark 2: Holdings-Based ---")

# === B-1. Prepare Data (No Universe Creation Needed) ===
print("B-1. Using actual holdings as the universe...")
df_holdings = holdings_df.copy()

# === B-2. Calculate Weights Against Holdings Benchmark ===
print("B-2. Calculating active weights against holdings benchmark...")
fund_total_investment_holdings = df_holdings.groupby("wficn")[
    "fund_investment"
].transform("sum")
df_holdings["fund_weight_j"] = (
    df_holdings["fund_investment"] / fund_total_investment_holdings
)

# CRITICAL DIFFERENCE: Sum of market cap is ONLY over stocks the fund actually holds
benchmark_total_cap_holdings = df_holdings.groupby("wficn")["market_cap"].transform(
    "sum"
)
df_holdings["market_cap_weight_j"] = (
    df_holdings["market_cap"] / benchmark_total_cap_holdings
)

df_holdings["active_weight_j"] = (
    df_holdings["fund_weight_j"] - df_holdings["market_cap_weight_j"]
)

# === B-3. Calculate Fund-Level DRS Scores (Holdings-Based) ===
print("B-3. Calculating fund-level DRS scores (Holdings-Based)...")
df_holdings["drs_component"] = (df_holdings["active_weight_j"] ** 2) * (
    2 / df_holdings["market_cap"]
)

drs_holdings = df_holdings.groupby(
    ["wficn", "morningstar_category"], as_index=False
).agg(
    drs_per_dollar=("drs_component", "sum"),
    fund_investment_total=("fund_investment", "sum"),
)
drs_holdings["drs_double_holdings"] = (
    2 * drs_holdings["drs_per_dollar"] * 2 * drs_holdings["fund_investment_total"]
)


# === Part C: Combine, Analyze, and Visualize Results ===
print("\n--- Combining and Visualizing Results ---")
# Merge the two sets of DRS scores into a single DataFrame for comparison
final_drs = pd.merge(
    drs_style[
        ["wficn", "morningstar_category", "fund_investment_total", "drs_double_style"]
    ],
    drs_holdings[["wficn", "drs_double_holdings"]],
    on="wficn",
)

# Bin funds into quintiles based on their total size
final_drs["fund_bin"] = pd.qcut(
    final_drs["fund_investment_total"], q=5, labels=[f"Q{i + 1}" for i in range(5)]
)

# --- Visualization for Benchmark 1: Style-Based ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_drs, x="fund_bin", y="drs_double_style")
plt.title("DRS Score (Style-Based Benchmark) by Fund Size Quintile")
plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
plt.ylabel("DRS Score (Style-Based)")
plt.tight_layout()
plt.show()

# --- Visualization for Benchmark 2: Holdings-Based ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=final_drs, x="fund_bin", y="drs_double_holdings")
plt.title("DRS Score (Holdings-Based Benchmark) by Fund Size Quintile")
plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
plt.ylabel("DRS Score (Holdings-Based)")
plt.tight_layout()
plt.show()


# === produce summaries
final_drs.groupby("fund_bin").agg(
    drs_double_style_mean=("drs_double_style", "mean"),
    drs_double_style_median=("drs_double_style", "median"),
    drs_double_holdings_mean=("drs_double_holdings", "mean"),
    drs_double_holdings_median=("drs_double_holdings", "median"),
    fund_size_avg=("fund_investment_total", "mean"),
    fund_count=("wficn", "count"),
)
