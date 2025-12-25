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
NHNL_LOOKBACK = 252          # ~52 weeks
AD_MA_WINDOWS = [50, 200]
DAILY_LOOKBACK = 260         # ~1 year
FULL_HISTORY_FILE = OUTPUT_DIR / "breadth_full_history.csv"

MODE = os.getenv("BREADTH_MODE", "daily").lower()

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

    return df

# ================= MAIN =================
def main():
    print(f"[INFO] Breadth mode = {MODE}", flush=True)

    symbols = load_symbols()
    prices = {}

    for sym in symbols:
        df = load_stock(sym)
        if df is not None:
            prices[sym] = df["Close"]

    if not prices:
        raise RuntimeError("No price data loaded")

    # -------- ALIGN DATES --------
    price_df = pd.DataFrame(prices).dropna(how="all")

    if MODE == "daily":
        price_df = price_df.tail(DAILY_LOOKBACK)

    print(f"[INFO] Price matrix shape: {price_df.shape}", flush=True)

    # ================= PILLAR 1: ADV / DECL =================
    daily_returns = price_df.diff()

    advances = (daily_returns > 0).sum(axis=1)
    declines = (daily_returns < 0).sum(axis=1)

    ad = advances - declines
    total = price_df.count(axis=1)

    ad_df = pd.DataFrame({
        "advances": advances,
        "declines": declines,
        "ad": ad,
        "ad_cum": ad.cumsum(),
        "ad_participation_pct": (advances + declines) / total * 100
    })

    for w in AD_MA_WINDOWS:
        ad_df[f"ad_ma_{w}"] = ad_df["ad"].rolling(w).mean()
        ad_df[f"ad_z_{w}"] = zscore(ad_df["ad"], w)

    ad_df["ad_percentile"] = percentile(ad_df["ad"])

    # ================= PILLAR 2: DMA BREADTH =================
    dma_frames = []

    for w in DMA_WINDOWS:
        dma = price_df.rolling(w).mean()
        above = (price_df > dma).sum(axis=1)
        pct = above / price_df.count(axis=1) * 100

        df = pd.DataFrame({
            f"pct_above_{w}dma": pct,
            f"pct_above_{w}dma_z": zscore(pct, 200),
            f"pct_above_{w}dma_pctile": percentile(pct),
            f"pct_above_{w}dma_mom": pct.diff(20)
        })
        dma_frames.append(df)

    dma_df = pd.concat(dma_frames, axis=1)

    # ================= PILLAR 3: NEW HIGHS / LOWS =================
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()

    nh = (price_df >= highs).sum(axis=1)
    nl = (price_df <= lows).sum(axis=1)

    nhnl_df = pd.DataFrame({
        "new_highs": nh,
        "new_lows": nl,
        "nhnl_diff": nh - nl,
        "nhnl_ratio": nh / (nh + nl).replace(0, np.nan)
    })

    # ================= MERGE =================
    breadth = pd.concat([ad_df, dma_df, nhnl_df], axis=1).dropna()

    # ================= SAVE =================
    today = datetime.utcnow().strftime("%Y-%m-%d")

    if MODE == "full":
        breadth.to_csv(FULL_HISTORY_FILE)
        print(f"[SUCCESS] Full history updated → {FULL_HISTORY_FILE}", flush=True)
    else:
        out_file = OUTPUT_DIR / "breadth_latest.csv"
        breadth.tail(1).to_csv(out_file)
        print(f"[SUCCESS] Daily breadth saved → {out_file}", flush=True)

    print(breadth.tail(3), flush=True)

if __name__ == "__main__":
    main()