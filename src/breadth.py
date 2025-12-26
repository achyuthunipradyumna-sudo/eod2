import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

np.seterr(divide="ignore", invalid="ignore")

# ================= CONFIG =================
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
VOL_WINDOWS = [20, 50, 200]
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

def load_stock(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================= TREND =================
def compute_trend_structure(index_df, breadth):
    price = index_df["Close"]
    dma50 = price.rolling(50).mean()
    dma200 = price.rolling(200).mean()

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

    dist_200 = (price / dma200 - 1)
    trend["dist_200_z"] = zscore(dist_200, 200)

    trend["persistence_200"] = (
        (price > dma200).rolling(250).sum() / 250 * 100
    )

    hh = price > price.rolling(50).max().shift(1)
    ll = price < price.rolling(50).min().shift(1)
    trend["hh_ll_ratio"] = hh.rolling(100).sum() - ll.rolling(100).sum()

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

# ================= VOLATILITY =================
def compute_index_volatility(index_df):
    ret = np.log(index_df["Close"] / index_df["Close"].shift(1))
    vol = pd.DataFrame(index=index_df.index)

    for w in VOL_WINDOWS:
        rv = ret.rolling(w).std() * np.sqrt(252)
        vol[f"vol_{w}"] = rv
        vol[f"vol_{w}_z"] = zscore(rv, 200)
        vol[f"vol_{w}_pct"] = percentile(rv)

    return vol

# ================= MOMENTUM =================
def compute_index_momentum(index_df):
    price = index_df["Close"]

    mom = pd.DataFrame(index=price.index)
    mom["mom_63"] = price.pct_change(63)
    mom["mom_126"] = price.pct_change(126)
    mom["mom_12_1"] = price.pct_change(252) - price.pct_change(21)

    mom["mom_12_1_z"] = zscore(mom["mom_12_1"], 200)
    mom["mom_12_1_pct"] = percentile(mom["mom_12_1"])

    return mom

# ================= MAIN =================
def main():
    prices = {}
    stock_mom_126 = {}

    for f in DATA_DIR.glob("*.csv"):
        if "nifty" in f.name.lower():
            continue

        df = load_stock(f.name)
        prices[f.stem.upper()] = df["Close"]
        stock_mom_126[f.stem.upper()] = df["Close"].pct_change(126)

    price_df = pd.DataFrame(prices).dropna(how="all")
    universe = price_df.count(axis=1)

    # ---------- BREADTH ----------
    ret = price_df.diff()
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    ad = adv - dec

    ad_df = pd.DataFrame({"advances": adv, "declines": dec, "ad": ad})

    for w in AD_MA_WINDOWS:
        ad_df[f"ad_ma_{w}"] = ad_df["ad"].rolling(w).mean()
        ad_df[f"ad_z_{w}"] = zscore(ad_df["ad"], w)

    ad_df["ad_pct"] = percentile(ad_df["ad"])

    dma_df = pd.DataFrame(index=price_df.index)
    for w in DMA_WINDOWS:
        dma_df[f"pct_above_{w}dma"] = (price_df > price_df.rolling(w).mean()).sum(axis=1) / universe * 100

    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    nh = (price_df >= highs).sum(axis=1)
    nl = (price_df <= lows).sum(axis=1)

    nh_pct = nh / universe * 100
    nl_pct = nl / universe * 100

    breadth = pd.concat([
        ad_df,
        dma_df,
        pd.DataFrame({
            "new_highs": nh,
            "new_lows": nl,
            "nh_pct": nh_pct,
            "nl_pct": nl_pct,
            "nhnl_net": nh_pct - nl_pct,
            "nhnl_z": zscore(nh_pct - nl_pct, NHNL_LOOKBACK)
        })
    ], axis=1).dropna()

    # ---------- STOCK MOMENTUM BREADTH ----------
    mom_df = pd.DataFrame(stock_mom_126)
    breadth["pct_stocks_pos_mom_126"] = (mom_df > 0).sum(axis=1) / universe * 100
    breadth["median_stock_mom_126"] = mom_df.median(axis=1)
    breadth["median_stock_mom_126_z"] = zscore(breadth["median_stock_mom_126"], 200)

    # ---------- INDEX LAYERS ----------
    for name, file in INDEX_FILES.items():
        idx = load_stock(file)

        trend = compute_trend_structure(idx, breadth).add_prefix(f"{name}_")
        vol = compute_index_volatility(idx).add_prefix(f"{name}_")
        mom = compute_index_momentum(idx).add_prefix(f"{name}_")

        breadth = breadth.join(trend).join(vol).join(mom)

    # ---------- OUTPUT ----------
    today = breadth.index[-1]
    breadth.loc[[today]].to_csv(OUTPUT_DIR / f"breadth_{today.date()}.csv")

    if FULL_HISTORY_FILE.exists():
        full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
        breadth = pd.concat([full, breadth])

    breadth = breadth[~breadth.index.duplicated(keep="last")]
    breadth.sort_index(inplace=True)
    breadth.to_csv(FULL_HISTORY_FILE)

    breadth.tail(LOOKBACK_DAILY).to_csv(
        OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"
    )

    print("[SUCCESS] Breadth + Trend + Volatility + Momentum computed", flush=True)

if __name__ == "__main__":
    main()