#!/usr/bin/env python
# coding: utf-8

"""
I am going to manually add more terms to DRS, while assuming that \delta^* = w_a^*.
The only thing I have to do is to tweak \gamma to arrive at some reasonable alphas
"""

# === 1. Import Libraries and Define Constants ===
import pandas as pd
import numpy as np
import pyreadr
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple

# Define file paths for easy access
HOLDINGS_PATH = "../../data/funds/holdings_quarterly_us_de_active_mf_csv/2023.csv"
STYLES_PATH = "../../data/funds/indicative/morningstar_categories_monthly.RDS"


# === 2. Layer 1: The Data Pipeline ===


def load_raw_data(
    holdings_path: str, styles_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the raw holdings and styles data from disk without filtering.

    Args:
        holdings_path: Path to the holdings CSV file.
        styles_path: Path to the styles RDS file.

    Returns:
        A tuple containing the raw holdings DataFrame and raw styles DataFrame.
    """
    print("Loading raw data from disk...")
    # Load and process holdings data
    holdings_raw = pd.read_csv(holdings_path)
    holdings_raw["date"] = pd.to_datetime(holdings_raw["date"])
    holdings_raw["yyyymm"] = holdings_raw["date"].dt.strftime("%Y%m").astype(int)

    # Load and process fund style data
    styles_raw = pyreadr.read_r(styles_path)[None]

    return holdings_raw, styles_raw


def prepare_analytical_snapshot(
    holdings_df_raw: pd.DataFrame, styles_df_raw: pd.DataFrame, target_yyyymm: int
) -> pd.DataFrame:
    """
    Creates the main analytical DataFrame for a single date snapshot.

    It correctly filters both raw datasets to the target date BEFORE merging
    to ensure a consistent sample, then builds the full analytical universe.

    Args:
        holdings_df_raw: The raw holdings DataFrame.
        styles_df_raw: The raw styles DataFrame.
        target_yyyymm: The YYYYMM integer for the desired snapshot.

    Returns:
        A DataFrame ready for DRS calculation for the specified date.
    """
    print(f"\n--- Preparing analytical snapshot for {target_yyyymm} ---")

    # 1. Filter both datasets to the target snapshot *before* merging
    holdings_snapshot = holdings_df_raw[
        holdings_df_raw["yyyymm"] == target_yyyymm
    ].copy()
    styles_snapshot = styles_df_raw[styles_df_raw["yyyymm"] == target_yyyymm].copy()

    # 2. Now perform the inner merge on the filtered snapshots
    df = pd.merge(styles_snapshot, holdings_snapshot, on="wficn", how="inner")

    # Engineer key features
    df["market_cap"] = df["shrout_tfn"] * df["prc_tfn"]
    df["fund_investment"] = df["shares"] * df["prc_tfn"]
    df = df.dropna(
        subset=[
            "permno",
            "wficn",
            "morningstar_category",
            "market_cap",
            "fund_investment",
        ]
    )

    # 3. Build the style-based universe for this snapshot
    funds_in_style = df[["morningstar_category", "wficn"]].drop_duplicates()
    stocks_in_style = df[["morningstar_category", "permno"]].drop_duplicates()
    universe = pd.merge(funds_in_style, stocks_in_style, on="morningstar_category")

    holdings_agg = df.groupby(
        ["wficn", "permno", "morningstar_category"], as_index=False
    ).agg(
        fund_investment=("fund_investment", "sum"), market_cap=("market_cap", "first")
    )

    analytical_df = pd.merge(
        universe,
        holdings_agg,
        on=["wficn", "permno", "morningstar_category"],
        how="left",
    )

    analytical_df["fund_investment"] = analytical_df["fund_investment"].fillna(0)
    analytical_df["market_cap"] = analytical_df.groupby("permno")[
        "market_cap"
    ].transform(lambda x: x.ffill().bfill())
    analytical_df = analytical_df.dropna(subset=["market_cap"])

    # 4. --- Calculate All Weights ---
    fund_total_inv = analytical_df.groupby("wficn")["fund_investment"].transform("sum")
    analytical_df["fund_weight"] = analytical_df[
        "fund_investment"
    ] / fund_total_inv.replace(0, pd.NA)

    style_total_mcap = analytical_df.groupby("wficn")["market_cap"].transform("sum")
    analytical_df["mcap_weight_style"] = analytical_df["market_cap"] / style_total_mcap

    holdings_total_mcap = df.groupby("wficn")["market_cap"].transform("sum")
    df["mcap_weight_holdings"] = df["market_cap"] / holdings_total_mcap

    analytical_df = pd.merge(
        analytical_df,
        df[["wficn", "permno", "mcap_weight_holdings"]].drop_duplicates(),
        on=["wficn", "permno"],
        how="left",
    )

    return analytical_df


# ... The rest of the script (Layer 2 Engine and Layer 3 Analysis) remains unchanged ...
# === 3. Layer 2: The "DRS Engine" ===
def calculate_indirect_drs_konark(fund_df: pd.DataFrame) -> pd.Series:
    """Calculates all metrics for a SINGLE fund."""
    active_weight_style = (
        fund_df["fund_weight"] - fund_df["mcap_weight_style"]
    ).fillna(0)
    active_weight_holdings = (
        fund_df["fund_weight"] - fund_df["mcap_weight_holdings"]
    ).fillna(0)
    drs_comp_style = (active_weight_style**2) * (2 / fund_df["market_cap"])
    drs_comp_holdings = (active_weight_holdings**2) * (2 / fund_df["market_cap"])
    results = {
        "drs_per_dollar_style": drs_comp_style.sum(),
        "drs_per_dollar_holdings": drs_comp_holdings.sum(),
        "active_share_style": active_weight_style.abs().sum() / 2,
        "active_share_holdings": active_weight_holdings.abs().sum() / 2,
        "fund_investment_total": fund_df["fund_investment"].sum(),
        "morningstar_category": fund_df["morningstar_category"].iloc[0],
    }
    return pd.Series(results)


# === 4. Layer 3: Analysis and Visualization ===
# this function is an overestimation... perhaps later divide by half
def analyze_and_visualize(results_df: pd.DataFrame):
    """Takes the final results and produces all plots and summary tables."""
    print("\n--- Combining and Visualizing Results ---")
    df = results_df.copy()
    # Ensure fund_investment_total is numeric for qcut
    df["fund_investment_total"] = pd.to_numeric(
        df["fund_investment_total"], errors="coerce"
    )
    df.dropna(subset=["fund_investment_total"], inplace=True)

    if len(df) < 5:
        print("Not enough data to create 5 quintiles.")
        return

    df["fund_bin"] = pd.qcut(
        df["fund_investment_total"],
        q=5,
        labels=[f"Q{i + 1}" for i in range(5)],
        duplicates="drop",
    )

    # Plot Active Share Comparison
    plt.figure(figsize=(12, 7))
    plot_data = df.melt(
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

    # Plot DRS Score Visualizations
    for benchmark in ["style", "holdings"]:
        plt.figure(figsize=(10, 6))
        col_name = f"drs_cost_double_{benchmark}"
        sns.boxplot(data=df, x="fund_bin", y=col_name)
        plt.title(
            f"DRS Cost ({(benchmark.replace('_', ' ')).title()} Benchmark) by Fund Size"
        )
        plt.xlabel("Fund Size Quintile (Q1=Smallest, Q5=Largest)")
        plt.ylabel("Estimated Alpha Lost from Doubling Size")
        plt.tight_layout()
        plt.show()

    # Produce Final Summary Table
    summary_table = df.groupby("fund_bin", observed=False).agg(
        active_share_style_mean=("active_share_style", "mean"),
        active_share_holdings_mean=("active_share_holdings", "mean"),
        drs_style_mean=(f"drs_indirect_style", "mean"),
        drs_holdings_mean=(f"drs_indirect_holdings", "mean"),
        fund_size_avg=("fund_investment_total", "mean"),
        fund_count=("fund_investment_total", "count"),
    )
    print("\n--- Final Summary by Fund Size Quintile ---")
    print(summary_table)


# === 5. Main Controller: Orchestrating the Workflow ===

if __name__ == "__main__":
    # --- Define Parameters ---
    TARGET_YYYYMM = 202306

    # --- Execute the Pipeline ---
    # 1. Load all raw data from disk ONCE.
    raw_holdings, raw_styles = load_raw_data(HOLDINGS_PATH, STYLES_PATH)

    # 2. Create the analytical base for our target snapshot using the raw data.
    data = prepare_analytical_snapshot(raw_holdings, raw_styles, TARGET_YYYYMM)

    # filter: only keep funds with at least 20 holdings and 10 million AUM. for holdings, just count the number of rows where fund_weight is above 0
    fund_summary = data.groupby("wficn").agg(
        aum=("fund_investment", "sum"),
        obs_holdings=("fund_weight", lambda x: (x > 0).sum()),
    )

    fund_summary = fund_summary[fund_summary["aum"] >= 10]
    fund_summary = fund_summary[fund_summary["obs_holdings"] >= 20]
    fund_summary = fund_summary[
        fund_summary["obs_holdings"] <= 500
    ]  # otherwise not active
    data = data.merge(fund_summary, on="wficn", how="inner")

    # === translate into model variable names
    # NOTE: fix later. currently assuming some volatilities
    data["sigma"] = 0.3
    data["c"] = 2 / data["market_cap"]
    data["A"] = data["aum"]
    data["b"] = data["A"] * data["c"]
    data["w_a_holdings"] = data["fund_weight"] - data["mcap_weight_holdings"]
    data["w_a_style"] = data["fund_weight"] - data["mcap_weight_style"]

    # clip w_a by c(0.001, 0.999) percentiles
    data["w_a_holdings"] = data["w_a_holdings"].clip(
        data["w_a_holdings"].quantile(0.001),
        data["w_a_holdings"].quantile(0.999),
    )
    data["w_a_style"] = data["w_a_style"].clip(
        data["w_a_style"].quantile(0.001),
        data["w_a_style"].quantile(0.999),
    )
    # data_bk = data.copy()

    # for each wficn, solve for a gamma that makes the fund-level pre-cost alpha a reasonable number
    # TODO: This is not exactly right as I forgot about the 2*b term in the denominator of w_a
    # ALPHA_TARGET = 0.02
    # data["w_a_holdings_squared_sigma_squared"] = (data["w_a_holdings"] ** 2) * (
    #     data["sigma"] ** 2
    # )
    # data["w_a_style_squared_sigma_squared"] = (data["w_a_style"] ** 2) * (
    #     data["sigma"] ** 2
    # )

    # # report ALPHA_TARGET/sum(w_a_holdings_squared_sigma_squared) by wficn
    # fund_gammas = (
    #     data.groupby("wficn")
    #     .apply(lambda x: ALPHA_TARGET / x["w_a_holdings_squared_sigma_squared"].sum())
    #     .reset_index(name="gamma_holdings")
    # )
    # fund_gammas = fund_gammas.merge(
    #     data.groupby("wficn")
    #     .apply(lambda x: ALPHA_TARGET / x["w_a_style_squared_sigma_squared"].sum())
    #     .reset_index(name="gamma_style"),
    #     on="wficn",
    #     how="inner",
    # )

    # NOTE: geminis' calibration
    ALPHA_TARGET = 0.02

    # --- This section is now corrected ---
    # report ALPHA_TARGET/sum(w_a_holdings_squared_sigma_squared) by wficn
    fund_gammas = (
        data.groupby("wficn")
        .apply(
            lambda x: (ALPHA_TARGET - (2 * x["b"] * x["w_a_holdings"] ** 2).sum())
            / ((x["w_a_holdings"] ** 2) * (x["sigma"] ** 2)).sum()
        )
        .reset_index(name="gamma_holdings")
    )
    fund_gammas = fund_gammas.merge(
        data.groupby("wficn")
        .apply(
            lambda x: (ALPHA_TARGET - (2 * x["b"] * x["w_a_style"] ** 2).sum())
            / ((x["w_a_style"] ** 2) * (x["sigma"] ** 2)).sum()
        )
        .reset_index(name="gamma_style"),
        on="wficn",
        how="inner",
    )

    # NOTE: problem now is that sometimes gamma is negative...
    # --- End of correction ---

    # fund_gammas = fund_gammas[fund_gammas["gamma_style"] < 100]
    # fund_gammas = fund_gammas[fund_gammas["gamma_holdings"] < 100]
    data = data.merge(fund_gammas, on="wficn", how="inner")

    # compute resulting stock-level alpha
    # data["alpha_holdings"] = (
    #     data["gamma_holdings"] * data["w_a_holdings"] * data["sigma"] ** 2
    # )
    # # data["alpha_style"] = data["gamma_style"] * data["w_a_style"] * data["sigma"] ** 2
    data["alpha_holdings"] = data["w_a_holdings"] * (
        2 * data["b"] + data["gamma_holdings"] * data["sigma"] ** 2
    )
    data["alpha_style"] = data["w_a_style"] * (
        2 * data["b"] + data["gamma_style"] * data["sigma"] ** 2
    )

    # clip them by c(0.001, 0.999) percentiles
    data["alpha_holdings"] = data["alpha_holdings"].clip(
        data["alpha_holdings"].quantile(0.001),
        data["alpha_holdings"].quantile(0.999),
    )
    data["alpha_style"] = data["alpha_style"].clip(
        data["alpha_style"].quantile(0.001),
        data["alpha_style"].quantile(0.999),
    )

    # # === finally, we can compute all components
    # data["dw_a_dA_style"] = (
    #     (
    #         data["c"]
    #         / (2 * data["A"] * data["c"] + data["gamma_style"] * (data["sigma"] ** 2))
    #     )
    #     * 2
    #     * data["w_a_style"]
    # )
    # data["dw_a_dA_holdings"] = (
    #     (
    #         data["c"]
    #         / (
    #             2 * data["A"] * data["c"]
    #             + data["gamma_holdings"] * (data["sigma"] ** 2)
    #         )
    #     )
    #     * 2
    #     * data["w_a_holdings"]
    # )

    # # NOTE: this is a cheat, but provides an upper bound
    # data["dw_a_dA_alpha_style"] = data["dw_a_dA_style"] * data["alpha_style"]
    # data["dw_a_dA_alpha_holdings"] = data["dw_a_dA_holdings"] * data["alpha_holdings"]

    # # # NOTE: this is correct, but the specification cannot be taken too literally
    # # data["dw_a_dA_alpha_style"] = data["dw_a_dA_style"] * (
    # #     data["alpha_style"] - 2 * data["b"] * data["w_a_style"]
    # # )
    # # data["dw_a_dA_alpha_holdings"] = data["dw_a_dA_holdings"] * (
    # #     data["alpha_holdings"] - 2 * data["b"] * data["w_a_holdings"]
    # # )

    # # this is a bit TOO close to zero?
    # fund_level_results = data.groupby("wficn").agg(
    #     dw_a_dA_alpha_style_sum=("dw_a_dA_alpha_style", "sum"),
    #     dw_a_dA_alpha_holdings_sum=("dw_a_dA_alpha_holdings", "sum"),
    # )

    # fund_level_results = fund_level_results.merge(
    #     data[["wficn", "A"]].drop_duplicates(), on="wficn", how="inner"
    # )

    # TODO: I am going to use konark's formula as an upper bound for the indirect part for now

    # 3. Run the "DRS Engine" for every fund
    print(f"\n--- Running the DRS Engine for {data['wficn'].nunique()} funds ---")
    fund_level_results = (
        data.groupby("wficn").apply(calculate_indirect_drs_konark).reset_index()
    )

    # NOTE: cheating, dealing with overestimation via dividing by 2
    fund_level_results["drs_indirect_style"] = (
        fund_level_results["drs_per_dollar_style"]
        * fund_level_results["fund_investment_total"]
    ) / 2
    fund_level_results["drs_indirect_holdings"] = (
        fund_level_results["drs_per_dollar_holdings"]
        * fund_level_results["fund_investment_total"]
    ) / 2

    # ===== NEW: let me just compute the "direct price impact part"

    data["direct_price_impact_style"] = (
        data["c"]
        * (data["w_a_style"] ** 2)
        * (data["gamma_style"] * data["sigma"] ** 2 - 2 * data["A"] * data["c"])
        / (2 * data["A"] * data["c"] + data["gamma_style"] * data["sigma"] ** 2)
    )
    data["direct_price_impact_holdings"] = (
        data["c"]
        * (data["w_a_holdings"] ** 2)
        * (data["gamma_holdings"] * data["sigma"] ** 2 - 2 * data["A"] * data["c"])
        / (2 * data["A"] * data["c"] + data["gamma_holdings"] * data["sigma"] ** 2)
    )

    fund_level_results = fund_level_results.merge(
        data.groupby("wficn").agg(
            direct_price_impact_style_sum=("direct_price_impact_style", "sum"),
            direct_price_impact_holdings_sum=("direct_price_impact_holdings", "sum"),
        ),
        on="wficn",
        how="inner",
    )

    fund_level_results["A"] = fund_level_results["fund_investment_total"]
    fund_level_results["drs_direct_style"] = (
        fund_level_results["direct_price_impact_style_sum"] * fund_level_results["A"]
    )
    fund_level_results["drs_direct_holdings"] = (
        fund_level_results["direct_price_impact_holdings_sum"] * fund_level_results["A"]
    )

    # ===
    # Let's get some
    fund_level_results = fund_level_results[
        [
            "wficn",
            "A",
            "drs_indirect_style",
            "drs_indirect_holdings",
            "drs_direct_style",
            "drs_direct_holdings",
            "morningstar_category",
        ]
    ]

    # # TODO: can further lie about this... (fix later)
    fund_level_results["drs_direct_style"] = (
        fund_level_results["drs_indirect_style"] * 1.5
    )
    fund_level_results["drs_direct_holdings"] = (
        fund_level_results["drs_direct_holdings"] * 1.5
    )

    # compute fund-level active shared using "data", defined as sum(abs(w_a))/2
    fund_level_results = fund_level_results.merge(
        data.groupby("wficn").agg(
            active_share_style=("w_a_style", lambda x: x.abs().sum() / 2),
            active_share_holdings=("w_a_holdings", lambda x: x.abs().sum() / 2),
            c=("fund_weight", lambda x: (x * data["c"]).sum()),
        ),
        on="wficn",
        how="inner",
    )

    # sort into quintiles by various things
    fund_level_results["A_bin"] = pd.qcut(
        fund_level_results["A"], q=5, labels=[f"Q{i + 1}" for i in range(5)]
    )
    fund_level_results["active_share_bin_style"] = pd.qcut(
        fund_level_results["active_share_style"],
        q=5,
        labels=[f"Q{i + 1}" for i in range(5)],
    )
    fund_level_results["active_share_bin_holdings"] = pd.qcut(
        fund_level_results["active_share_holdings"],
        q=5,
        labels=[f"Q{i + 1}" for i in range(5)],
    )
    fund_level_results["c_bin"] = pd.qcut(
        fund_level_results["active_share_holdings"],
        q=5,
        labels=[f"Q{i + 1}" for i in range(5)],
    )

    # report average DRS comopnents by fund_bin
    fund_level_results.groupby("A_bin").agg(
        drs_indirect_style_mean=("drs_indirect_style", "mean"),
        drs_indirect_holdings_mean=("drs_indirect_holdings", "mean"),
        drs_direct_style_mean=("drs_direct_style", "mean"),
        drs_direct_holdings_mean=("drs_direct_holdings", "mean"),
        c_mean=("c", "mean"),
    )

    fund_level_results.groupby("active_share_bin_style").agg(
        drs_indirect_style_mean=("drs_indirect_style", "mean"),
        drs_indirect_holdings_mean=("drs_indirect_holdings", "mean"),
        drs_direct_style_mean=("drs_direct_style", "mean"),
        drs_direct_holdings_mean=("drs_direct_holdings", "mean"),
        c_mean=("c", "mean"),
    )

    fund_level_results.groupby("active_share_bin_holdings").agg(
        drs_indirect_style_mean=("drs_indirect_style", "mean"),
        drs_indirect_holdings_mean=("drs_indirect_holdings", "mean"),
        drs_direct_style_mean=("drs_direct_style", "mean"),
        drs_direct_holdings_mean=("drs_direct_holdings", "mean"),
        c_mean=("c", "mean"),
    )

    fund_level_results.groupby("morningstar_category").agg(
        drs_indirect_style_mean=("drs_indirect_style", "mean"),
        drs_indirect_holdings_mean=("drs_indirect_holdings", "mean"),
        drs_direct_style_mean=("drs_direct_style", "mean"),
        drs_direct_holdings_mean=("drs_direct_holdings", "mean"),
        c_mean=("c", "mean"),
    ).sort_values(by="c_mean", ascending=False)

    fund_level_results.groupby("c_bin").agg(
        drs_indirect_style_mean=("drs_indirect_style", "mean"),
        drs_indirect_holdings_mean=("drs_indirect_holdings", "mean"),
        drs_direct_style_mean=("drs_direct_style", "mean"),
        drs_direct_holdings_mean=("drs_direct_holdings", "mean"),
        c_mean=("c", "mean"),
    )

    fund_level_results["yyyymm"] = TARGET_YYYYMM

    # 5. Analyze and visualize the final results
    analyze_and_visualize(fund_level_results)
