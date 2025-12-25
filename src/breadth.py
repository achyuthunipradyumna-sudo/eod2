import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ================ CONFIG =================
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
NHNL_LOOKBACK = 252
AD_MA_WINDOWS = [50, 200]
LOOKBACK_DAILY = 260

FULL_HISTORY = OUTPUT_DIR / "breadth_full_history.csv"

# ================ HELPERS =================
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
        print(f"[INFO] Using BREADTH_SYMBOLS = {symbols}", flush=True)
        return symbols

    symbols = [f.stem.upper() for f in DATA_DIR.glob("*.csv")]
    print(f"[INFO] Loaded {len(symbols)} symbols from disk", flush=True)
    return symbols

def load_stock(symbol):
    path = DATA_DIR / f"{symbol.lower()}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================ LABELING =================
def label_ad(z):
    if z > 1:
        return "Strong Expansion"
    if z > 0:
        return "Moderate Expansion"
    if z >= -0.5:
        return "Neutral"
    if z >= -1:
        return "Moderate Contraction"
    return "Strong Contraction"

def label_dma(pct):
    if pct >= 70:
        return "Bullish Participation"
    if pct >= 40:
        return "Neutral Participation"
    return "Bearish Participation"

def label_nhnl(h, l):
    if h > l:
        return "Healthy"
    if l > h:
        return "Deteriorating"
    return "Balanced"

# ================ MAIN =================
def main():
    symbols = load_symbols()
    prices = {}

    for sym in symbols:
        df = load_stock(sym)
        if df is not None:
            prices[sym] = df["Close"]

    if not prices:
        raise RuntimeError("No price data loaded")

    price_df = pd.DataFrame(prices).dropna(how="all")

    # -------- PILLAR 1: ADV / DECL --------
    returns = price_df.diff()
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)
    ad = advances - declines

    ad_df = pd.DataFrame({
        "advances": advances,
        "declines": declines,
        "ad": ad
    })

    ad_df["ad_ma_200"] = ad_df["ad"].rolling(200).mean()
    ad_df["ad_z_200"] = zscore(ad_df["ad"], 200)
    ad_df["ad_pct"] = percentile(ad_df["ad"])

    # -------- PILLAR 2: DMA BREADTH --------
    dma = price_df.rolling(200).mean()
    above_200 = (price_df > dma).sum(axis=1)
    pct_above_200dma = above_200 / price_df.count(axis=1) * 100

    # -------- PILLAR 3: NH / NL --------
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    new_highs = (price_df >= highs).sum(axis=1)
    new_lows = (price_df <= lows).sum(axis=1)

    # -------- MERGE --------
    breadth = pd.DataFrame({
        "advances": advances,
        "declines": declines,
        "ad": ad,
        "ad_ma_200": ad_df["ad_ma_200"],
        "ad_z_200": ad_df["ad_z_200"],
        "ad_pct": ad_df["ad_pct"],
        "pct_above_200dma": pct_above_200dma,
        "new_highs": new_highs,
        "new_lows": new_lows
    }).dropna()

    # -------- LABELS --------
    breadth["ad_regime"] = breadth["ad_z_200"].apply(label_ad)
    breadth["dma_regime"] = breadth["pct_above_200dma"].apply(label_dma)
    breadth["internal_health"] = breadth.apply(
        lambda r: label_nhnl(r["new_highs"], r["new_lows"]), axis=1
    )

    # -------- DAILY SNAPSHOT (ONE ROW) --------
    today_row = breadth.iloc[[-1]]
    today = today_row.index[0].strftime("%Y-%m-%d")

    daily_file = OUTPUT_DIR / f"breadth_{today}.csv"
    today_row.to_csv(daily_file)
    print(f"[SUCCESS] Daily snapshot saved → {daily_file}", flush=True)

    # -------- FULL HISTORY (MONTHLY UPDATE) --------
    update_full = False
    if FULL_HISTORY.exists():
        full = pd.read_csv(FULL_HISTORY, parse_dates=["Date"], index_col="Date")
        last_month = full.index.max().month
        if datetime.utcnow().month != last_month:
            update_full = True
    else:
        full = None
        update_full = True

    if update_full:
        combined = pd.concat([full, breadth]) if full is not None else breadth
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_csv(FULL_HISTORY)
        print(f"[SUCCESS] Full history updated → {FULL_HISTORY}", flush=True)

    # -------- LOOKBACK --------
    breadth.tail(LOOKBACK_DAILY).to_csv(
        OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"
    )

if __name__ == "__main__":
    main()