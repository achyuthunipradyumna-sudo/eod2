import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("src/eod2_data/daily")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
NHNL_LOOKBACK = 252  # 52-week highs/lows
Z_WINDOWS = [50, 200]

# -----------------------------
# HELPERS
# -----------------------------
def zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def percentile_rank(series, window):
    return series.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )


def load_symbols():
    """
    Priority:
    1. BREADTH_SYMBOLS env variable
    2. All CSVs in data directory
    """
    env_symbols = os.getenv("BREADTH_SYMBOLS")

    if env_symbols:
        symbols = [s.strip().upper() for s in env_symbols.split(",")]
        print(f"[INFO] Using BREADTH_SYMBOLS: {symbols}")
        return symbols

    print("[INFO] No BREADTH_SYMBOLS set – scanning all stocks")
    return [f.stem for f in DATA_DIR.glob("*.csv")]


# -----------------------------
# MAIN LOGIC
# -----------------------------
def process_stock(symbol):
    file_path = DATA_DIR / f"{symbol}.csv"
    if not file_path.exists():
        print(f"[WARN] Missing file for {symbol}")
        return None

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)

    close = df["Close"]

    out = {
        "symbol": symbol,
        "date": df["Date"].iloc[-1]
    }

    # ----- DMA breadth (individual stock) -----
    for w in DMA_WINDOWS:
        dma = close.rolling(w).mean()
        above = close.iloc[-1] > dma.iloc[-1]

        out[f"above_{w}dma"] = int(above)
        out[f"dist_{w}dma_pct"] = (close.iloc[-1] / dma.iloc[-1] - 1) * 100

        # Normalization
        out[f"z_{w}dma_dist"] = zscore(close / dma - 1, 200).iloc[-1]
        out[f"pct_{w}dma_dist"] = percentile_rank(close / dma - 1, 200).iloc[-1]

    # ----- New High / New Low -----
    rolling_high = close.rolling(NHNL_LOOKBACK).max()
    rolling_low = close.rolling(NHNL_LOOKBACK).min()

    out["new_52w_high"] = int(close.iloc[-1] >= rolling_high.iloc[-1])
    out["new_52w_low"] = int(close.iloc[-1] <= rolling_low.iloc[-1])

    return out


def main():
    symbols = load_symbols()
    results = []

    print(f"[INFO] Processing {len(symbols)} stocks")

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Processing {sym}")
        row = process_stock(sym)
        if row:
            results.append(row)

    if not results:
        print("[ERROR] No data processed")
        return

    df_out = pd.DataFrame(results)
    out_file = OUTPUT_DIR / f"breadth_snapshot_{datetime.now().date()}.csv"
    df_out.to_csv(out_file, index=False)

    print(f"[SUCCESS] Saved breadth snapshot → {out_file}")


if __name__ == "__main__":
    main()