import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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

# 🔑 Use patterns, not filenames
INDEX_PATTERNS = {
    "nifty50": ["nifty 50.csv"],
    "nifty_total": ["nifty total market.csv"]
}

# ================= HELPERS =================
def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def percentile(series):
    return series.rank(pct=True)

def resolve_index_file(patterns):
    matches = []
    for p in patterns:
        matches.extend(DATA_DIR.glob(p))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"[ERROR] No index file found for patterns: {patterns}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"[ERROR] Multiple index files matched {patterns}: "
            f"{[m.name for m in matches]}"
        )

    return matches[0]

def load_stock_by_path(path: Path):
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

    dist_200 = price / dma200 - 1
    trend["dist_200_z"] = zscore(dist_200, 200)
    trend["persistence_200"] = (price > dma200).rolling(250).mean() * 100

    return trend

# ================= VOLATILITY =================
def compute_index_volatility(index_df):
    ret = np.log(index_df["Close"] / index_df["Close"].shift(1))
    vol = pd.DataFrame(index=index_df.index)

    for w in VOL_WINDOWS:
        rv = ret.rolling(w).std() * np.sqrt(252)
        vol[f"vol_{w}"] = rv
        vol[f"vol_{w}_z"] = zscore(rv, 200)

    return vol

# ================= MOMENTUM =================
def compute_index_momentum(index_df):
    price = index_df["Close"]
    mom = pd.DataFrame(index=price.index)

    mom["mom_63"] = price.pct_change(63)
    mom["mom_126"] = price.pct_change(126)
    mom["mom_12_1"] = price.pct_change(252) - price.pct_change(21)
    mom["mom_12_1_z"] = zscore(mom["mom_12_1"], 200)

    return mom

# ================= DASHBOARD =================
def plot_dashboard(df, name):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["ad_z_50"])
    axes[0].set_title("Short-term Breadth")

    axes[1].plot(df.index, df["pct_above_200dma"])
    axes[1].plot(df.index, df[f"{name}_mom_126"])
    axes[1].set_title("Medium-term Trend & Momentum")

    axes[2].plot(df.index, df["nhnl_z"])
    axes[2].plot(df.index, df[f"{name}_mom_12_1_z"])
    axes[2].set_title("Long-term Structure")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_dashboard.png")
    plt.close()

# ================= MAIN =================
def main():
    prices = {}

    for f in DATA_DIR.glob("*.csv"):
        if "nifty" in f.name.lower():
            continue
        prices[f.stem] = load_stock_by_path(f)["Close"]

    price_df = pd.DataFrame(prices)
    universe = price_df.count(axis=1)

    ret = price_df.diff()
    ad = (ret > 0).sum(axis=1) - (ret < 0).sum(axis=1)

    breadth = pd.DataFrame({"ad": ad})
    breadth["ad_z_50"] = zscore(ad, 50)

    for w in DMA_WINDOWS:
        breadth[f"pct_above_{w}dma"] = (
            (price_df > price_df.rolling(w).mean()).mean(axis=1) * 100
        )

    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    breadth["nhnl_z"] = zscore(
        (price_df >= highs).sum(axis=1) - (price_df <= lows).sum(axis=1),
        NHNL_LOOKBACK
    )

    # ----- INDEX LAYERS -----
    for name, patterns in INDEX_PATTERNS.items():
        idx_path = resolve_index_file(patterns)
        idx = load_stock_by_path(idx_path)

        breadth = breadth.join(compute_trend_structure(idx, breadth).add_prefix(f"{name}_"))
        breadth = breadth.join(compute_index_volatility(idx).add_prefix(f"{name}_"))
        breadth = breadth.join(compute_index_momentum(idx).add_prefix(f"{name}_"))

        plot_dashboard(breadth.tail(LOOKBACK_DAILY), name)

    # ----- OUTPUT -----
    today = breadth.index[-1]
    breadth.loc[[today]].to_csv(OUTPUT_DIR / f"breadth_{today.date()}.csv")

    if FULL_HISTORY_FILE.exists():
        full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
        breadth = pd.concat([full, breadth])

    breadth = breadth[~breadth.index.duplicated(keep="last")]
    breadth.sort_index(inplace=True)
    breadth.to_csv(FULL_HISTORY_FILE)

    print("[SUCCESS] Fixed index resolution + dashboards generated")

if __name__ == "__main__":
    main()