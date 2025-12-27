import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

np.seterr(divide="ignore", invalid="ignore")

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "eod2_data" / "daily"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
VOL_WINDOWS = [20, 50]
NHNL_LOOKBACK = 252
LOOKBACK_DAILY = 260

FULL_HISTORY_FILE = OUTPUT_DIR / "breadth_full_history.csv"
LOOKBACK_FILE = OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"

INDEX_KEYS = {
    "nifty50": "nifty 50",
    "nifty_total": "nifty total market",
}

# ================= HELPERS =================
def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def resolve_index_file(index_name: str) -> Path:
    index_name = index_name.lower().strip()
    matches = []

    for f in DATA_DIR.iterdir():
        if not f.name.lower().endswith(".csv"):
            continue

        name = f.name.lower().replace(".csv", "").strip()
        if any(x in name for x in ["futures", "arbitrage", "tr"]):
            continue

        if name == index_name:
            matches.append(f)

    if len(matches) != 1:
        raise RuntimeError(f"[ERROR] Index resolution failed for {index_name}")

    return matches[0]

def load_stock_by_path(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================= TREND =================
def compute_trend_structure(index_df):
    price = index_df["Close"]
    dma50 = price.rolling(50).mean()
    dma200 = price.rolling(200).mean()

    trend = pd.DataFrame(index=price.index)
    trend["price_above_50"] = (price > dma50).astype(int)
    trend["price_above_200"] = (price > dma200).astype(int)
    trend["dma200_slope_pct"] = dma200.pct_change(20) * 100
    trend["dist_200_z"] = zscore(price / dma200 - 1, 200)
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

# ================= MAIN =================
def main():
    prices = {}
    stock_vols = {20: {}, 50: {}}

    # ---------- STOCK UNIVERSE ----------
    for f in DATA_DIR.iterdir():
        if not f.name.lower().endswith(".csv"):
            continue
        if "nifty" in f.name.lower():
            continue

        df = load_stock_by_path(f)
        prices[f.stem] = df["Close"]

        ret = np.log(df["Close"] / df["Close"].shift(1))
        for w in VOL_WINDOWS:
            stock_vols[w][f.stem] = ret.rolling(w).std() * np.sqrt(252)

    price_df = pd.DataFrame(prices)
    universe = price_df.count(axis=1)

    # ---------- BREADTH ----------
    ret = price_df.diff()
    ad = (ret > 0).sum(axis=1) - (ret < 0).sum(axis=1)

    breadth = pd.DataFrame({"ad": ad})
    breadth["ad_z_50"] = zscore(ad, 50)

    # % above DMAs
    for w in DMA_WINDOWS:
        pct = (price_df > price_df.rolling(w).mean()).mean(axis=1) * 100
        breadth[f"pct_above_{w}dma"] = pct
        breadth[f"pct_above_{w}dma_z"] = zscore(pct, 200)
        breadth[f"pct_above_{w}dma_chg"] = pct.diff(20)

    # NH–NL
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    nhnl = (price_df >= highs).sum(axis=1) - (price_df <= lows).sum(axis=1)
    breadth["nhnl_z"] = zscore(nhnl, NHNL_LOOKBACK)

    # ---------- VOLATILITY BREADTH ----------
    for w in VOL_WINDOWS:
        vol_df = pd.DataFrame(stock_vols[w])
        median_vol = vol_df.median(axis=1)
        breadth[f"median_stock_vol_{w}"] = median_vol
        breadth[f"median_stock_vol_{w}_z"] = zscore(median_vol, 200)

    # ---------- INDEX LAYERS ----------
    for name, key in INDEX_KEYS.items():
        idx = load_stock_by_path(resolve_index_file(key))

        breadth = breadth.join(compute_trend_structure(idx).add_prefix(f"{name}_"))
        breadth = breadth.join(compute_index_volatility(idx).add_prefix(f"{name}_"))
        breadth = breadth.join(compute_index_momentum(idx).add_prefix(f"{name}_"))

    # ---------- OUTPUT FILES ----------
    today = breadth.index[-1]
    today_str = today.date()

    # 1️⃣ Daily snapshot
    daily_file = OUTPUT_DIR / f"breadth_{today_str}.csv"
    breadth.loc[[today]].to_csv(daily_file)

    # 2️⃣ Rolling lookback
    breadth.tail(LOOKBACK_DAILY).to_csv(LOOKBACK_FILE)

    # 3️⃣ Monthly full history (only on 1st)
    if today.day == 1:
        if FULL_HISTORY_FILE.exists():
            full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
            breadth = pd.concat([full, breadth])

        breadth = breadth[~breadth.index.duplicated(keep="last")]
        breadth.sort_index(inplace=True)
        breadth.to_csv(FULL_HISTORY_FILE)

        print("[INFO] Monthly full history updated")

    # ---------- LOG ----------
    print("\n========== GENERATED STATE FILES ==========")
    print(f" - {daily_file.name}")
    print(f" - {LOOKBACK_FILE.name}")
    if today.day == 1:
        print(f" - {FULL_HISTORY_FILE.name}")
    print("==========================================")
    print("[SUCCESS] Daily market state pipeline completed")

if __name__ == "__main__":
    main()