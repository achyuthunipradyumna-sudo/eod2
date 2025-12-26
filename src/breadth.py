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
    path = DATA_DIR / symbol
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================= TREND FUNCTIONS =================
def compute_trend_structure(index_df, breadth):
    price = index_df["Close"]
    dma50 = price.rolling(50).mean()
    dma200 = price.rolling(200).mean()

    # ----- A. Trend Bias -----
    trend = pd.DataFrame(index=price.index)
    trend["price_above_50"] = (price > dma50).astype(int)
    trend["price_above_200"] = (price > dma200).astype(int)
    trend["dma200_slope_pct"] = dma200.pct_change(20) * 100

    trend["trend_bias"] = np.where(
        (trend["price_above_200"] == 1) & (trend["dma200_slope_pct"] > 0),
        "uptrend",
        np.where(
            (trend["price_above_200"] == 0) & (trend["dma200_slope_pct"] < 0),
            "downtrend",
            "sideways"
        )
    )

    # ----- B. Trend Strength -----
    dist_200 = (price / dma200 - 1)
    trend["dist_200_z"] = zscore(dist_200, 200)

    above_200 = price > dma200
    trend["persistence_200"] = (
        above_200.rolling(250).sum() / 250 * 100
    )

    hh = price > price.rolling(50).max().shift(1)
    ll = price < price.rolling(50).min().shift(1)
    trend["hh_ll_ratio"] = (
        hh.rolling(100).sum() - ll.rolling(100).sum()
    )

    # ----- C. Trend Integrity -----
    trend["ad_divergence"] = (
        (price.diff(50) > 0) &
        (breadth["ad"].diff(50) < 0)
    ).astype(int)

    trend["nhnl_divergence"] = (
        (price.diff(50) > 0) &
        (breadth["nhnl_net"].diff(50) < 0)
    ).astype(int)

    trend["dma_rollover"] = (
        (trend["price_above_200"] == 1) &
        (breadth["pct_above_200dma"].diff(20) < 0)
    ).astype(int)

    return trend

# ================= MAIN =================
def main():
    print("[INFO] Loading stock universe", flush=True)

    prices = {}
    for f in DATA_DIR.glob("*.csv"):
        if "nifty" in f.name.lower():
            continue
        df = load_stock(f.name)
        prices[f.stem.upper()] = df["Close"]

    price_df = pd.DataFrame(prices).dropna(how="all")
    universe = price_df.count(axis=1)

    # ---------- BREADTH ----------
    ret = price_df.diff()
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    ad = adv - dec

    ad_df = pd.DataFrame({
        "advances": adv,
        "declines": dec,
        "ad": ad
    })

    for w in AD_MA_WINDOWS:
        ad_df[f"ad_ma_{w}"] = ad_df["ad"].rolling(w).mean()
        ad_df[f"ad_z_{w}"] = zscore(ad_df["ad"], w)

    ad_df["ad_pct"] = percentile(ad_df["ad"])

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
    nhnl_net = nh_pct - nl_pct
    nhnl_z = zscore(nhnl_net, NHNL_LOOKBACK)

    nhnl_df = pd.DataFrame({
        "new_highs": nh,
        "new_lows": nl,
        "nh_pct": nh_pct,
        "nl_pct": nl_pct,
        "nhnl_net": nhnl_net,
        "nhnl_z": nhnl_z
    })

    breadth = pd.concat([ad_df, dma_df, nhnl_df], axis=1).dropna()

    # ---------- TREND (INDEX LEVEL) ----------
    for name, file in INDEX_FILES.items():
        idx_df = load_stock(file)
        trend = compute_trend_structure(idx_df, breadth)
        trend.columns = [f"{name}_{c}" for c in trend.columns]
        breadth = breadth.join(trend, how="left")

    # ---------- OUTPUTS ----------
    today = breadth.index[-1]
    daily = breadth.loc[[today]]
    daily.to_csv(OUTPUT_DIR / f"breadth_{today.date()}.csv")

    if FULL_HISTORY_FILE.exists():
        full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
        combined = pd.concat([full, breadth])
    else:
        combined = breadth.copy()

    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_csv(FULL_HISTORY_FILE)

    breadth.tail(LOOKBACK_DAILY).to_csv(
        OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"
    )

    print("[SUCCESS] Breadth + Trend computation complete", flush=True)

if __name__ == "__main__":
    main()