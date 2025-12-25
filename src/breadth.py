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
Z_WINDOW = 200

# ---------------- HELPERS ----------------
def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def percentile(series):
    return series.rank(pct=True)

def load_symbols():
    env = os.getenv("BREADTH_SYMBOLS")
    if env:
        syms = [s.strip().upper() for s in env.split(",")]
        print(f"[INFO] Using BREADTH_SYMBOLS = {syms}", flush=True)
        return syms
    syms = [f.stem.upper() for f in DATA_DIR.glob("*.csv")]
    print(f"[INFO] Loaded {len(syms)} symbols from disk", flush=True)
    return syms

# ---------------- MAIN ----------------
def main():
    symbols = load_symbols()
    print(f"[INFO] Symbols to process: {len(symbols)}", flush=True)

    price_matrix = {}
    dates = None

    # ---------- LOAD PRICES ----------
    for sym in symbols:
        path = DATA_DIR / f"{sym.lower()}.csv"  # 🔑 CASE FIX
        if not path.exists():
            print(f"[WARN] Missing {sym}", flush=True)
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        df.sort_values("Date", inplace=True)

        price_matrix[sym] = df["Close"].values
        dates = df["Date"].values

        print(f"[OK] Loaded {sym} ({len(df)})", flush=True)

    if not price_matrix:
        raise RuntimeError("No price data loaded")

    price_df = pd.DataFrame(price_matrix, index=dates)

    # ---------- ADVANCE / DECLINE ----------
    returns = price_df.diff()
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)

    ad_df = pd.DataFrame({
        "Date": price_df.index,
        "Advances": advances,
        "Declines": declines,
        "AD_Net": advances - declines
    }).set_index("Date")

    ad_df["AD_50DMA"] = ad_df["AD_Net"].rolling(50).mean()
    ad_df["AD_200DMA"] = ad_df["AD_Net"].rolling(200).mean()
    ad_df["AD_Z"] = zscore(ad_df["AD_Net"], Z_WINDOW)
    ad_df["AD_PCTL"] = percentile(ad_df["AD_Net"])

    # ---------- DMA BREADTH ----------
    dma_rows = []
    for w in DMA_WINDOWS:
        dma = price_df.rolling(w).mean()
        pct_above = (price_df > dma).sum(axis=1) / price_df.count(axis=1) * 100
        dma_rows.append(pd.DataFrame({
            "Date": price_df.index,
            f"Pct_Above_{w}DMA": pct_above
        }).set_index("Date"))

    dma_df = pd.concat(dma_rows, axis=1)
    for w in DMA_WINDOWS:
        col = f"Pct_Above_{w}DMA"
        dma_df[f"{col}_Z"] = zscore(dma_df[col], Z_WINDOW)
        dma_df[f"{col}_PCTL"] = percentile(dma_df[col])

    # ---------- SAVE ----------
    today = datetime.now().strftime("%Y-%m-%d")
    ad_df.to_csv(OUTPUT_DIR / f"ad_breadth_{today}.csv")
    dma_df.to_csv(OUTPUT_DIR / f"dma_breadth_{today}.csv")

    print("[SUCCESS] Breadth files written", flush=True)

if __name__ == "__main__":
    main()