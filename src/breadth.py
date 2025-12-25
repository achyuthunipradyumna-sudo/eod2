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
AD_MA_WINDOWS = [50, 200]
NHNL_LOOKBACK = 252

# ---------------- HELPERS ----------------
def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std

def percentile(series):
    return series.rank(pct=True)

def load_symbols():
    env = os.getenv("BREADTH_SYMBOLS")
    if env:
        syms = [s.strip().upper() for s in env.split(",")]
        print(f"[INFO] Using BREADTH_SYMBOLS = {syms}", flush=True)
        return syms

    syms = [f.stem for f in DATA_DIR.glob("*.csv")]
    print(f"[INFO] Using ALL symbols ({len(syms)})", flush=True)
    return syms

# ---------------- MAIN ----------------
def main():
    symbols = load_symbols()
    print(f"[INFO] Symbols to process: {len(symbols)}", flush=True)

    closes = {}
    dates = None

    # ---- Load data ----
    for sym in symbols:
        path = DATA_DIR / f"{sym}.csv"
        if not path.exists():
            print(f"[WARN] Missing {sym}", flush=True)
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)

        if dates is None:
            dates = df["Date"]

        closes[sym] = df["Close"].reset_index(drop=True)

        print(f"[LOAD] {sym} rows={len(df)}", flush=True)

    price_df = pd.DataFrame(closes)
    price_df.index = dates

    # ---- ADV / DECL ----
    adv = (price_df > price_df.shift(1)).sum(axis=1)
    dec = (price_df < price_df.shift(1)).sum(axis=1)

    net_ad = adv - dec
    ad_ratio = adv / (adv + dec)

    ad_df = pd.DataFrame({
        "date": price_df.index,
        "advances": adv,
        "declines": dec,
        "net_ad": net_ad,
        "ad_ratio": ad_ratio
    })

    # ---- A/D MAs ----
    for w in AD_MA_WINDOWS:
        ad_df[f"net_ad_ma_{w}"] = net_ad.rolling(w).mean()
        ad_df[f"net_ad_z_{w}"] = zscore(net_ad, w)
        ad_df[f"net_ad_pct_{w}"] = percentile(net_ad)

    print("[INFO] A/D metrics computed", flush=True)

    # ---- DMA Breadth ----
    dma_rows = []

    for sym, close in closes.items():
        row = {"symbol": sym}
        for w in DMA_WINDOWS:
            dma = close.rolling(w).mean()
            row[f"above_{w}dma"] = int(close.iloc[-1] > dma.iloc[-1])
            row[f"dist_{w}dma_pct"] = (close.iloc[-1] / dma.iloc[-1] - 1) * 100
        dma_rows.append(row)

    dma_df = pd.DataFrame(dma_rows)

    # ---- New High / Low ----
    nh = (price_df >= price_df.rolling(NHNL_LOOKBACK).max()).iloc[-1].sum()
    nl = (price_df <= price_df.rolling(NHNL_LOOKBACK).min()).iloc[-1].sum()

    # ---- Save ----
    today = datetime.now().strftime("%Y-%m-%d")

    ad_file = OUTPUT_DIR / f"ad_breadth_{today}.csv"
    dma_file = OUTPUT_DIR / f"dma_breadth_{today}.csv"

    ad_df.to_csv(ad_file, index=False)
    dma_df.to_csv(dma_file, index=False)

    print(f"[SUCCESS] Saved {ad_file}", flush=True)
    print(f"[SUCCESS] Saved {dma_file}", flush=True)
    print(f"[SUMMARY] New Highs={nh} New Lows={nl}", flush=True)

if __name__ == "__main__":
    main()