import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------- CONFIG ----------------
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
NHNL_LOOKBACK = 252
Z_WINDOW = 200

# ---------------- HELPERS ----------------
def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def percentile(series, window):
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def load_symbols():
    env = os.getenv("BREADTH_SYMBOLS")
    if env:
        syms = [s.strip().upper() for s in env.split(",")]
        print(f"[INFO] Using BREADTH_SYMBOLS = {syms}")
        return syms
    return [f.stem.upper() for f in DATA_DIR.glob("*.csv")]

def load_price(symbol):
    file = DATA_DIR / f"{symbol}.csv"
    if not file.exists():
        print(f"[WARN] Missing {symbol}")
        return None
    df = pd.read_csv(file, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    print(f"[OK] Loaded {symbol} ({len(df)})")
    return df[["Close"]]

# ---------------- MAIN ----------------
def main():
    symbols = load_symbols()
    print(f"[INFO] Symbols to process: {len(symbols)}")

    # -------- LOAD & ALIGN PRICES --------
    price_map = {}
    for s in symbols:
        df = load_price(s)
        if df is not None:
            price_map[s] = df["Close"]

    if not price_map:
        print("[ERROR] No price data loaded")
        return

    price_df = pd.concat(price_map, axis=1)
    print(f"[INFO] Price matrix shape: {price_df.shape}")

    # -------- PILLAR A: ADVANCE / DECLINE --------
    adv = (price_df > price_df.shift(1)).sum(axis=1)
    dec = (price_df < price_df.shift(1)).sum(axis=1)
    ad = adv - dec
    ad_line = ad.cumsum()

    ad_50 = ad.rolling(50).mean()
    ad_200 = ad.rolling(200).mean()
    ad_z = zscore(ad, Z_WINDOW)
    ad_pct = percentile(ad, Z_WINDOW)

    # -------- PILLAR B: DMA BREADTH --------
    dma_breadth = {}

    for w in DMA_WINDOWS:
        above = (price_df > price_df.rolling(w).mean()).sum(axis=1)
        pct_above = above / price_df.count(axis=1) * 100

        dma_breadth[f"pct_above_{w}dma"] = pct_above
        dma_breadth[f"pct_above_{w}dma_z"] = zscore(pct_above, Z_WINDOW)
        dma_breadth[f"pct_above_{w}dma_pct"] = percentile(pct_above, Z_WINDOW)

    dma_df = pd.DataFrame(dma_breadth)

    # -------- PILLAR C: NEW HIGHS / LOWS --------
    highs = (price_df >= price_df.rolling(NHNL_LOOKBACK).max()).sum(axis=1)
    lows = (price_df <= price_df.rolling(NHNL_LOOKBACK).min()).sum(axis=1)

    nh_nl = highs - lows
    nh_z = zscore(nh_nl, Z_WINDOW)
    nh_pct = percentile(nh_nl, Z_WINDOW)

    # -------- FINAL OUTPUT --------
    out = pd.DataFrame({
        "advances": adv,
        "declines": dec,
        "ad": ad,
        "ad_line": ad_line,
        "ad_50dma": ad_50,
        "ad_200dma": ad_200,
        "ad_z": ad_z,
        "ad_pct": ad_pct,
        "new_highs": highs,
        "new_lows": lows,
        "nh_nl": nh_nl,
        "nh_nl_z": nh_z,
        "nh_nl_pct": nh_pct,
    })

    out = pd.concat([out, dma_df], axis=1)

    today = datetime.now().strftime("%Y-%m-%d")
    out_file = OUTPUT_DIR / f"market_breadth_{today}.csv"
    out.to_csv(out_file)

    print(f"[SUCCESS] Breadth snapshot saved → {out_file}")
    print(out.tail(3))

if __name__ == "__main__":
    main()