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
NHNL_LOOKBACK = 252  # ~52 weeks
AD_MA_WINDOWS = [50, 200]

# ================= HELPERS =================
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

# ================= MAIN =================
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

    # -------- ALIGN DATES --------
    price_df = pd.DataFrame(prices).dropna(how="all")
    print(f"[INFO] Price matrix shape: {price_df.shape}", flush=True)

    # ================= PILLAR 1: ADV / DECL =================
    daily_returns = price_df.diff()

    advances = (daily_returns > 0).sum(axis=1)
    declines = (daily_returns < 0).sum(axis=1)

    ad_line = advances - declines
    ad_df = pd.DataFrame({
        "advances": advances,
        "declines": declines,
        "ad": ad_line
    })

    for w in AD_MA_WINDOWS:
        ad_df[f"ad_ma_{w}"] = ad_df["ad"].rolling(w).mean()
        ad_df[f"ad_z_{w}"] = zscore(ad_df["ad"], w)

    ad_df["ad_percentile"] = percentile(ad_df["ad"])

    # ================= PILLAR 2: DMA BREADTH =================
    dma_breadth = {}

    for w in DMA_WINDOWS:
        dma = price_df.rolling(w).mean()
        above = (price_df > dma).sum(axis=1)
        dma_breadth[f"pct_above_{w}dma"] = above / price_df.count(axis=1) * 100

    dma_df = pd.DataFrame(dma_breadth)

    # ================= PILLAR 3: NEW HIGHS / LOWS =================
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()

    nh = (price_df >= highs).sum(axis=1)
    nl = (price_df <= lows).sum(axis=1)

    nhnl_df = pd.DataFrame({
        "new_highs": nh,
        "new_lows": nl
    })

    # ================= MERGE ALL =================
    breadth = pd.concat([ad_df, dma_df, nhnl_df], axis=1).dropna()

    # ================= SAVE =================
    today = datetime.utcnow().strftime("%Y-%m-%d")
    out_file = OUTPUT_DIR / f"breadth_{today}.csv"
    breadth.to_csv(out_file)

    print(f"[SUCCESS] Breadth saved → {out_file}", flush=True)
    print(breadth.tail(3), flush=True)

if __name__ == "__main__":
    main()