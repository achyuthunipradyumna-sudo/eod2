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
        print(f"[WARN] Missing {symbol}", flush=True)
        return None

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)

    print(f"[OK] Loaded {symbol} ({len(df)})", flush=True)
    return df

# ================ MAIN =================
def main():
    symbols = load_symbols()
    print(f"[INFO] Symbols to process: {len(symbols)}", flush=True)

    prices = {}
    for sym in symbols:
        df = load_stock(sym)
        if df is not None:
            prices[sym] = df["Close"]

    if not prices:
        raise RuntimeError("No price data loaded")

    price_df = pd.DataFrame(prices).dropna(how="all")
    print(f"[INFO] Price matrix shape: {price_df.shape}", flush=True)

    # ===== PILLAR 1: ADV / DECL =====
    daily_returns = price_df.diff()
    advances = (daily_returns > 0).sum(axis=1)
    declines = (daily_returns < 0).sum(axis=1)
    ad = advances - declines

    ad_df = pd.DataFrame({
        "advances": advances,
        "declines": declines,
        "ad": ad
    })

    for w in AD_MA_WINDOWS:
        ad_df[f"ad_ma_{w}"] = ad_df["ad"].rolling(w).mean()
        ad_df[f"ad_z_{w}"] = zscore(ad_df["ad"], w)

    ad_df["ad_pct"] = percentile(ad_df["ad"])

    # ===== PILLAR 2: DMA BREADTH =====
    dma_df = pd.DataFrame(index=price_df.index)
    for w in DMA_WINDOWS:
        dma = price_df.rolling(w).mean()
        above = (price_df > dma).sum(axis=1)
        dma_df[f"pct_above_{w}dma"] = above / price_df.count(axis=1) * 100

    # ===== PILLAR 3: NEW HIGHS / LOWS =====
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()

    nhnl_df = pd.DataFrame({
        "new_highs": (price_df >= highs).sum(axis=1),
        "new_lows": (price_df <= lows).sum(axis=1)
    })

    # ===== MERGE ALL =====
    breadth = pd.concat([ad_df, dma_df, nhnl_df], axis=1).dropna()
    breadth.sort_index(inplace=True)

    # ===== DAILY SNAPSHOT (ONE ROW ONLY) =====
    latest_row = breadth.iloc[[-1]]
    today = latest_row.index[0].strftime("%Y-%m-%d")
    daily_file = OUTPUT_DIR / f"breadth_{today}.csv"
    latest_row.to_csv(daily_file)
    print(f"[SUCCESS] Daily breadth saved → {daily_file}", flush=True)

    # ===== FULL HISTORY (MONTHLY UPDATE) =====
    update_full = False
    if not FULL_HISTORY.exists():
        update_full = True
        print("[INFO] Creating full history", flush=True)
        full = None
    else:
        full = pd.read_csv(FULL_HISTORY, parse_dates=["Date"], index_col="Date")
        last_month = full.index.max().month
        if last_month != breadth.index[-1].month:
            update_full = True
            print("[INFO] New month detected → updating full history", flush=True)

    if update_full:
        combined = breadth if full is None else pd.concat([full, breadth])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_csv(FULL_HISTORY)
        print(f"[SUCCESS] Full history updated → {FULL_HISTORY}", flush=True)

    # ===== LOOKBACK FILE =====
    lookback = breadth.tail(LOOKBACK_DAILY)
    look_file = OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"
    lookback.to_csv(look_file)
    print(f"[SUCCESS] Lookback saved → {look_file}", flush=True)

if __name__ == "__main__":
    main()