import os
import glob
import pandas as pd
import numpy as np

# ---------------- CONFIG ---------------- #
from pathlib import Path
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = "output"
OUTPUT_FILE = "breadth_daily.csv"

DMAS_STOCKS = [20, 50, 100, 200]
NH_NL_LOOKBACK = 252  # 52-week

# ---------------------------------------- #

def zscore(series, window=252):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def percentile(series, window=252):
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )


def load_all_stocks():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    stocks = {}

    for f in files:
        symbol = os.path.basename(f).replace(".csv", "")
        df = pd.read_csv(f, parse_dates=["Date"])
        df = df.sort_values("Date")
        df = df[["Date", "Open", "High", "Low", "Close"]]
        stocks[symbol] = df

    return stocks


def compute_breadth(stocks):
    all_dates = sorted(
        set().union(*[df["Date"] for df in stocks.values()])
    )

    results = []

    for date in all_dates:
        advances = 0
        declines = 0

        above_dma = {d: 0 for d in DMAS_STOCKS}
        total_valid = 0

        new_highs = 0
        new_lows = 0

        for df in stocks.values():
            if date not in df["Date"].values:
                continue

            row = df[df["Date"] == date].iloc[0]
            prev = df[df["Date"] < date]

            if prev.empty:
                continue

            prev_close = prev.iloc[-1]["Close"]
            close = row["Close"]

            # -------- A / D -------- #
            if close > prev_close:
                advances += 1
            elif close < prev_close:
                declines += 1

            # -------- DMAs -------- #
            total_valid += 1
            for d in DMAS_STOCKS:
                if len(prev) >= d:
                    dma = prev.tail(d)["Close"].mean()
                    if close > dma:
                        above_dma[d] += 1

            # -------- NH / NL -------- #
            if len(prev) >= NH_NL_LOOKBACK:
                high_52w = prev.tail(NH_NL_LOOKBACK)["High"].max()
                low_52w = prev.tail(NH_NL_LOOKBACK)["Low"].min()

                if close >= high_52w:
                    new_highs += 1
                elif close <= low_52w:
                    new_lows += 1

        if total_valid == 0:
            continue

        row_out = {
            "Date": date,
            "Advances": advances,
            "Declines": declines,
            "AD_Net": advances - declines,
            "NH": new_highs,
            "NL": new_lows,
            "NH_NL_Net": new_highs - new_lows,
        }

        for d in DMAS_STOCKS:
            row_out[f"Pct_Above_{d}DMA"] = above_dma[d] / total_valid * 100

        results.append(row_out)

    return pd.DataFrame(results).sort_values("Date")


def enrich_metrics(df):
    # ----- A/D ----- #
    df["AD_50DMA"] = df["AD_Net"].rolling(50).mean()
    df["AD_200DMA"] = df["AD_Net"].rolling(200).mean()
    df["AD_Z"] = zscore(df["AD_Net"])
    df["AD_Pctl"] = percentile(df["AD_Net"])

    # ----- NH / NL ----- #
    df["NHNL_Z"] = zscore(df["NH_NL_Net"])
    df["NHNL_Pctl"] = percentile(df["NH_NL_Net"])

    # ----- DMA Breadth ----- #
    for d in DMAS_STOCKS:
        col = f"Pct_Above_{d}DMA"
        df[f"{col}_Z"] = zscore(df[col])
        df[f"{col}_Pctl"] = percentile(df[col])

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading stock data...")
    stocks = load_all_stocks()

    print("Computing breadth metrics...")
    breadth = compute_breadth(stocks)

    print("Adding normalization & historical context...")
    breadth = enrich_metrics(breadth)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    breadth.to_csv(output_path, index=False)

    print(f"Breadth data saved to {output_path}")


if __name__ == "__main__":
    main()