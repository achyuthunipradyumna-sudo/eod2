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

# ================= LABEL LOGIC =================
def participation_label(ad_z):
    if ad_z > 1:
        return "strong_participation"
    if ad_z < -1:
        return "weak_participation"
    return "neutral_participation"

def dma_label(pct_above_50):
    if pct_above_50 > 70:
        return "broad_uptrend"
    if pct_above_50 < 30:
        return "broad_downtrend"
    return "range_bound"

def nhnl_label(nh, nl, universe):
    if nh > 0.1 * universe:
        return "momentum_expansion"
    if nl > 0.1 * universe:
        return "panic_or_distribution"
    return "normal"

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

    price_df = pd.DataFrame(prices).dropna(how="all")
    universe = price_df.count(axis=1)
    print(f"[INFO] Price matrix shape: {price_df.shape}", flush=True)

    # -------- PILLAR 1: ADV / DECL --------
    ret = price_df.diff()
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    ad = adv - dec

    ad_df = pd.DataFrame({
        "advances": adv,
        "declines": dec,
        "ad": ad,
    })

    for w in AD_MA_WINDOWS:
        ad_df[f"ad_ma_{w}"] = ad_df["ad"].rolling(w).mean()
        ad_df[f"ad_z_{w}"] = zscore(ad_df["ad"], w)

    ad_df["ad_pct"] = percentile(ad_df["ad"])

    # -------- PILLAR 2: DMA BREADTH --------
    dma_df = pd.DataFrame(index=price_df.index)
    for w in DMA_WINDOWS:
        dma = price_df.rolling(w).mean()
        above = (price_df > dma).sum(axis=1)
        dma_df[f"pct_above_{w}dma"] = above / universe * 100

    # -------- PILLAR 3: NH / NL --------
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    nh = (price_df >= highs).sum(axis=1)
    nl = (price_df <= lows).sum(axis=1)

    nhnl_df = pd.DataFrame({"new_highs": nh, "new_lows": nl})

    # -------- MERGE --------
    breadth = pd.concat([ad_df, dma_df, nhnl_df], axis=1).dropna()

    # -------- LABELS --------
    breadth["participation_label"] = breadth["ad_z_50"].apply(participation_label)
    breadth["dma_label"] = breadth["pct_above_50dma"].apply(dma_label)
    breadth["nhnl_label"] = [
        nhnl_label(nh, nl, universe.loc[d])
        for d, nh, nl in zip(breadth.index,
                             breadth["new_highs"],
                             breadth["new_lows"])
    ]

    # ================= OUTPUTS =================
    today = breadth.index[-1]

    # 1️⃣ DAILY SNAPSHOT (ONE ROW ONLY)
    daily_df = breadth.loc[[today]]
    daily_file = OUTPUT_DIR / f"breadth_{today.date()}.csv"
    daily_df.to_csv(daily_file)
    print(f"[SUCCESS] Daily breadth saved → {daily_file}", flush=True)

    # 2️⃣ FULL HISTORY (MONTHLY UPDATE)
    update_full = False
    if not FULL_HISTORY_FILE.exists():
        update_full = True
        print("[INFO] Creating full history", flush=True)
        full = pd.DataFrame()
    else:
        full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
        if full.index.max().month != today.month:
            update_full = True
            print("[INFO] Month rollover detected – updating full history", flush=True)

    if update_full:
        combined = pd.concat([full, breadth])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_csv(FULL_HISTORY_FILE)
        print(f"[SUCCESS] Full history updated → {FULL_HISTORY_FILE}", flush=True)

    # 3️⃣ LOOKBACK FILE (260 days)
    lookback = breadth.tail(LOOKBACK_DAILY)
    lookback_file = OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"
    lookback.to_csv(lookback_file)
    print(f"[SUCCESS] Lookback saved → {lookback_file}", flush=True)

if __name__ == "__main__":
    main()