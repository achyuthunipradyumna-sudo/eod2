import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================
# CONFIG
# =============================
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
AD_MA_WINDOWS = [50, 200]
NHNL_LOOKBACK = 252

# =============================
# HELPERS
# =============================
def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def percentile(series):
    return series.rank(pct=True)

def load_symbols():
    env = os.getenv("BREADTH_SYMBOLS")
    if env:
        symbols = [s.strip().upper() for s in env.split(",")]
        print(f"[INFO] Using BREADTH_SYMBOLS: {symbols}")
        return symbols
    print("[INFO] No BREADTH_SYMBOLS set – scanning all CSVs")
    return [f.stem for f in DATA_DIR.glob("*.csv")]

# =============================
# LOAD ALL DATA
# =============================
def load_all_prices(symbols):
    print("[INFO] Loading price data")
    frames = []

    for i, sym in enumerate(symbols, 1):
        path = DATA_DIR / f"{sym}.csv"
        if not path.exists():
            print(f"[WARN] Missing {sym}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df = df[["Date", "Close"]].copy()
        df["symbol"] = sym
        frames.append(df)

        if i % 10 == 0:
            print(f"[INFO] Loaded {i}/{len(symbols)} stocks")

    return pd.concat(frames, ignore_index=True)

# =============================
# A/D BREADTH
# =============================
def compute_ad(prices):
    print("[INFO] Computing Advances / Declines")

    prices.sort_values(["symbol", "Date"], inplace=True)
    prices["prev_close"] = prices.groupby("symbol")["Close"].shift(1)

    prices["advance"] = (prices["Close"] > prices["prev_close"]).astype(int)
    prices["decline"] = (prices["Close"] < prices["prev_close"]).astype(int)

    daily = prices.groupby("Date").agg(
        advances=("advance", "sum"),
        declines=("decline", "sum")
    )

    daily["net_ad"] = daily["advances"] - daily["declines"]
    daily["ad_ratio"] = daily["advances"] / daily["declines"].replace(0, np.nan)

    # Moving averages
    for w in AD_MA_WINDOWS:
        daily[f"net_ad_{w}dma"] = daily["net_ad"].rolling(w).mean()

    # Normalization
    daily["net_ad_z"] = zscore(daily["net_ad"], 200)
    daily["net_ad_pct"] = percentile(daily["net_ad"])

    return daily.reset_index()

# =============================
# DMA & NH/NL BREADTH
# =============================
def compute_stock_breadth(prices):
    print("[INFO] Computing stock-level breadth")

    results = []

    for sym, df in prices.groupby("symbol"):
        df = df.sort_values("Date")
        close = df["Close"]

        row = {"symbol": sym, "date": df["Date"].iloc[-1]}

        for w in DMA_WINDOWS:
            dma = close.rolling(w).mean()
            row[f"above_{w}dma"] = int(close.iloc[-1] > dma.iloc[-1])

        row["new_52w_high"] = int(close.iloc[-1] >= close.rolling(NHNL_LOOKBACK).max().iloc[-1])
        row["new_52w_low"] = int(close.iloc[-1] <= close.rolling(NHNL_LOOKBACK).min().iloc[-1])

        results.append(row)

    return pd.DataFrame(results)

# =============================
# MAIN
# =============================
def main():
    symbols = load_symbols()
    print(f"[INFO] Symbols count: {len(symbols)}")

    prices = load_all_prices(symbols)

    # --- A/D ---
    ad_df = compute_ad(prices)
    ad_out = OUTPUT_DIR / "ad_breadth.csv"
    ad_df.to_csv(ad_out, index=False)
    print(f"[SUCCESS] Saved {ad_out}")

    # --- Stock Breadth ---
    sb_df = compute_stock_breadth(prices)
    sb_out = OUTPUT_DIR / "stock_breadth_snapshot.csv"
    sb_df.to_csv(sb_out, index=False)
    print(f"[SUCCESS] Saved {sb_out}")

if __name__ == "__main__":
    main()