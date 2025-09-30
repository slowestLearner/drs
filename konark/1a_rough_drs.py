#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pyreadr
import pandas as pd

# get holdings data
holdings_raw = pd.read_csv(
    "../../data/funds/holdings_quarterly_us_de_active_mf_csv/2023.csv"
)
print(holdings_raw.head())

# get styles data
styles_raw = pyreadr.read_r(
    "../../data/funds/indicative/morningstar_categories_monthly.RDS"
)
styles_raw = styles_raw[None]
styles_2023 = styles_raw[styles_raw["yyyymm"].between(202301, 202312)]
print(styles_2023.head())

# merge holdings and styles
holdings_style = pd.merge(styles_2023, holdings_raw, on="wficn", how="inner")
holdings_style.head()

# Create new columns
holdings_style["market_cap"] = holdings_style["shrout_tfn"] * holdings_style["prc_tfn"]
holdings_style["fund_investment"] = holdings_style["shares"] * holdings_style["prc_tfn"]
holdings_style = holdings_style.dropna(subset=["permno"])

# Preview
holdings_style.head()

# === use average holdings by style as a benchmark

# total holdings by market cap (NOTE. this is over multiple yyyymm?)
style_stock_caps = (
    holdings_style.groupby(["morningstar_category", "permno"])["market_cap"]
    .sum()
    .reset_index(name="style_stock_market_cap")
)
print(style_stock_caps.head())

# Compute total market cap per style
style_totals = (
    style_stock_caps.groupby("morningstar_category")["style_stock_market_cap"]
    .sum()
    .reset_index(name="style_total_market_cap")
)

# Merge back
style_stock_caps = style_stock_caps.merge(
    style_totals, on="morningstar_category", how="left"
)

# Add percentage weight of each stock within the style
style_stock_caps["style_stock_weight"] = (
    style_stock_caps["style_stock_market_cap"]
    / style_stock_caps["style_total_market_cap"]
)
style_stock_caps.head()


n_funds = holdings_style["wficn"].nunique()
n_stocks = holdings_style["permno"].nunique()
print(f"Raw data has {n_funds} unique funds and {n_stocks} unique stocks.")


# unique funds per style
funds_in_style = holdings_style[["morningstar_category", "wficn"]].drop_duplicates()
print("Funds in style:", funds_in_style.shape)
print(funds_in_style.head())

# unique stocks per style
stocks_in_style = holdings_style[["morningstar_category", "permno"]].drop_duplicates()
print("\nStocks in style:", stocks_in_style.shape)
print(stocks_in_style.head())

# Cartesian product: all fund–stock pairs within each style
universe = funds_in_style.merge(stocks_in_style, on="morningstar_category", how="inner")
print("\nUniverse (fund x stock per style):", universe.shape)
print(universe.head())

# 2. Build stock-level market caps (per style × stock)
stock_caps = (
    holdings_style.groupby(["morningstar_category", "permno"], as_index=False)[
        "market_cap"
    ].median()
    # or .max(), .mean(), depending on how market_cap is stored
)
stock_caps.head()

# --- 0) Clean keys and drop missing permno (cannot join on NaN) ---
print(holdings_style.shape)
holdings_style = holdings_style.copy()
holdings_style["morningstar_category"] = (
    holdings_style["morningstar_category"].astype(str).str.strip()
)
print(holdings_style.shape)

for k in ["wficn", "permno"]:
    holdings_style[k] = pd.to_numeric(holdings_style[k], errors="coerce")

holdings_style = holdings_style.dropna(
    subset=["morningstar_category", "wficn", "permno"]
)
print(holdings_style.shape)

agg_spec = {
    "fund_investment": "sum",
    "shares": "sum",
    # market_cap should be merged from a stock-level table; if you must take from holdings:
    # "market_cap": "max",  # or "first"
}
holdings_agg = holdings_style.groupby(
    ["morningstar_category", "wficn", "permno"], as_index=False
).agg(agg_spec)


KEYS = [
    "morningstar_category",
    "wficn",
    "permno",
]  # add "yyyymm" if you want per-month rows

# 1) Are there duplicates on the RHS for the chosen keys?
dup_mask = holdings_agg.duplicated(KEYS, keep=False)
print("Duplicate RHS rows on keys:", dup_mask.sum())
print(holdings_agg.loc[dup_mask, KEYS + ["yyyymm"]].head(10))

# 2) Check how the merge behaves (will show many_to_many if keys aren’t unique)
probe = universe.merge(
    holdings_agg, on=KEYS, how="left", indicator=True, validate="many_to_one"
)
print(probe["_merge"].value_counts())

merged = universe.merge(
    holdings_agg, on=["morningstar_category", "wficn", "permno"], how="left"
)
merged.head()

# flag whether the fund actually holds the stock
merged["held_flag"] = merged["fund_investment"].notna()
merged.head()

cols_to_merge = ["fund_investment"]  # adjust to what you have

# fill missing numeric fields with 0 for “not held”
for c in cols_to_merge:
    merged[c] = merged[c].fillna(0)

merged.head()


# 5. Merge in the true market_cap (keeps values even when fund_investment=0)
merged = merged.merge(
    stock_caps,
    on=["morningstar_category", "permno"],
    how="left",
    validate="many_to_one",
)
merged.head()

# ==== finally, computing DRS?

df = merged.copy()
# Grouped calculation by fund (wficn)
df["market_cap_weight_j"] = df.groupby("wficn")["market_cap"].transform(
    lambda x: x / x.sum()
)
df["fund_weight_j"] = df.groupby("wficn")["fund_investment"].transform(
    lambda x: x / x.sum()
)
df["active_weight_j"] = df["fund_weight_j"] - df["market_cap_weight_j"]
df["active_weight_j^2"] = df["active_weight_j"] * df["active_weight_j"]
df["c_j"] = 2 / df["market_cap"]

# Preview for first few funds
df[
    [
        "wficn",
        "permno",
        "market_cap",
        "fund_investment",
        "market_cap_weight_j",
        "fund_weight_j",
        "active_weight_j",
        "active_weight_j^2",
        "c_j",
    ]
].head(100)


drs = (
    df.groupby("wficn")
    .apply(
        lambda g: pd.Series(
            {
                "drs_per_dollar": (g["active_weight_j^2"] * g["c_j"]).sum(),
                "fund_investment_total": g["fund_investment"].sum(),
            }
        )
    )
    .reset_index()
)
drs.head()

drs["drs_double"] = 2 * drs["drs_per_dollar"] * 2 * drs["fund_investment_total"]
drs["drs_double"].describe(include="all").T

# Suppose you have a column "fund_investment"
drs["fund_bin"] = pd.qcut(
    drs["fund_investment_total"], q=5, labels=[f"Q{i + 1}" for i in range(5)]
)

means = (
    drs.groupby("fund_bin")
    .agg(
        drs_double_mean=("drs_double", "mean"),
        drs_double_median=("drs_double", "median"),
        drs_double_q25=("drs_double", lambda x: x.quantile(0.25)),
        drs_double_q75=("drs_double", lambda x: x.quantile(0.75)),
        fund_size_total=("fund_investment_total", "sum"),
        fund_size_avg=("fund_investment_total", "mean"),
        fund_count=("fund_investment_total", "count"),
    )
    .reset_index()
)
means

import matplotlib.pyplot as plt

plt.hist(drs["drs_double"])

# In[66]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=drs, x="fund_bin", y="drs_double")
plt.title("drs_double by fund_investment_total quintile")
plt.show()


# === this is the DRS per dollar
drs = (
    df.groupby("wficn")
    .apply(
        lambda g: pd.Series(
            {
                "drs_per_dollar": (g["active_weight_j^2"] * g["c_j"]).sum(),
                "fund_investment_total": g["fund_investment"].sum(),
                "morningstar_category": g["morningstar_category"].iloc[0],
            }
        )
    )
    .reset_index()
)

drs.head()


drs["drs_double"] = 2 * drs["drs_per_dollar"] * 2 * drs["fund_investment_total"]
drs["drs_double"].describe(include="all").T

# === DRS by style
means = (
    drs.groupby("morningstar_category")
    .agg(
        drs_double_mean=("drs_double", "mean"),
        fund_size_total=("fund_investment_total", "sum"),
        fund_size_avg=("fund_investment_total", "mean"),
        fund_count=("fund_investment_total", "count"),
    )
    .reset_index()
)
means

sns.boxplot(data=drs, x="morningstar_category", y="drs_double")
plt.title("drs_double by fund_investment_total quintile")
plt.show()
