import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ================= CONFIG =================
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
NHNL_LOOKBACK = 252
AD_MA_WINDOWS = [50, 200]
LOOKBACK_DAILY = 260

FULL_HISTORY_FILE = OUTPUT_DIR / "breadth_full_history.csv"

INDEX_FILES = {
    "nifty50": "nifty 50.csv",
    "nifty_total": "nifty total market.csv"
}

# ================= HELPERS =================
def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def percentile(series):
    return series.rank(pct=True)

def load_stock(symbol):
    path = DATA_DIR / f"{symbol.lower()}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

def load_index(name):
    path = DATA_DIR / INDEX_FILES[name]
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    return df

# ================= TREND STRUCTURE =================
def compute_trend_structure(idx_df):
    close = idx_df["Close"]

    dma50 = close.rolling(50).mean()
    dma200 = close.rolling(200).mean()

    # ---------- Direction ----------
    price_above_200 = close > dma200
    price_above_50 = close > dma50

    dma200_slope = dma200.diff(20) / dma200.shift(20) * 100

    def trend_bias(row):
        if row["price_above_200"] and row["dma200_slope_pct"] > 0:
            return "bullish"
        if not row["price_above_200"] and row["dma200_slope_pct"] < 0:
            return "bearish"
        return "sideways"

    # ---------- Strength ----------
    dist_200 = (close / dma200 - 1) * 100
    dist_200_z = zscore(dist_200, 250)

    persist_200 = (
        (close > dma200)
        .rolling(250)
        .mean() * 100
    )

    hh = close > close.rolling(20).max().shift(1)
    ll = close < close.rolling(20).min().shift(1)
    hh_ll_ratio = hh.rolling(100).sum() / (ll.rolling(100).sum() + 1)

    # ---------- Integrity ----------
    idx_highs = close.rolling(252).max()
    idx_hh = close >= idx_highs

    trend_df = pd.DataFrame({
        "index_close": close,
        "price_above_200": price_above_200.astype(int),
        "price_above_50": price_above_50.astype(int),
        "dma200_slope_pct": dma200_slope,
        "trend_distance_200": dist_200,
        "trend_distance_200_z": dist_200_z,
        "trend_persistence_200": persist_200,
        "hh_ll_ratio": hh_ll_ratio,
        "index_hh": idx_hh.astype(int),
    })

    trend_df["trend_bias"] = trend_df.apply(trend_bias, axis=1)

    return trend_df.dropna()

# ================= MAIN =================
def main():
    # ---------- LOAD STOCK UNIVERSE ----------
    symbols = [
        f.stem.upper()
        for f in DATA_DIR.glob("*.csv")
        if f.stem.lower() not in ["nifty 50", "nifty total market"]
    ]

    prices = {}
    for sym in symbols:
        df = load_stock(sym)
        if df is not None:
            prices[sym] = df["Close"]

    price_df = pd.DataFrame(prices).dropna(how="all")
    universe = price_df.count(axis=1)

    # ---------- BREADTH ----------
    returns = price_df.diff()
    adv = (returns > 0).sum(axis=1)
    dec = (returns < 0).sum(axis=1)
    ad = adv - dec

    ad_df = pd.DataFrame({"advances": adv, "declines": dec, "ad": ad})
    ad_df["ad_z_200"] = zscore(ad_df["ad"], 200)

    dma_df = pd.DataFrame(index=price_df.index)
    for w in DMA_WINDOWS:
        dma = price_df.rolling(w).mean()
        above = (price_df > dma).sum(axis=1)
        dma_df[f"pct_above_{w}dma"] = above / universe * 100

    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    nh = (price_df >= highs).sum(axis=1)
    nl = (price_df <= lows).sum(axis=1)

    nh_pct = nh / universe * 100
    nl_pct = nl / universe * 100
    nhnl_z = zscore(nh_pct - nl_pct, 252)

    breadth = pd.concat(
        [ad_df, dma_df,
         nh.rename("new_highs"),
         nl.rename("new_lows"),
         nh_pct.rename("nh_pct"),
         nl_pct.rename("nl_pct"),
         nhnl_z.rename("nhnl_z")],
        axis=1
    ).dropna()

    # ---------- TREND (NIFTY TOTAL MARKET) ----------
    idx_df = load_index("nifty_total")
    trend_df = compute_trend_structure(idx_df)

    # ---------- MERGE ----------
    breadth = breadth.join(
        trend_df[[
            "trend_bias",
            "price_above_200",
            "price_above_50",
            "dma200_slope_pct",
            "trend_distance_200",
            "trend_distance_200_z",
            "trend_persistence_200",
            "hh_ll_ratio"
        ]],
        how="left"
    )

    # ---------- OUTPUT ----------
    today = breadth.index[-1]
    breadth.loc[[today]].to_csv(OUTPUT_DIR / f"breadth_{today.date()}.csv")
    breadth.tail(LOOKBACK_DAILY).to_csv(
        OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"
    )

    if not FULL_HISTORY_FILE.exists() or pd.read_csv(
        FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date"
    ).index.max().month != today.month:
        combined = (
            pd.concat([
                pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
                if FULL_HISTORY_FILE.exists() else pd.DataFrame(),
                breadth
            ])
            .loc[~lambda x: x.index.duplicated(keep="last")]
            .sort_index()
        )
        combined.to_csv(FULL_HISTORY_FILE)

if __name__ == "__main__":
    main()