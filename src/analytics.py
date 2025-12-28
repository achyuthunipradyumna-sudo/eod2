import pandas as pd
import numpy as np
from pathlib import Path

np.seterr(divide="ignore", invalid="ignore")

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "eod2_data" / "daily"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DMA_WINDOWS = [20, 50, 200]
BREAKOUT_WINDOWS = [20, 63, 126, 252]   # 252 = NH / NL
LOOKBACK_DAILY = 260

OUTPUT_FILE = OUTPUT_DIR / "breadth_pct_base_with_net_hl.csv"

# ================= HELPERS =================
def load_stock(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================= MAIN =================
def main():
    prices = {}

    # ---------- STOCK UNIVERSE ----------
    for f in DATA_DIR.iterdir():
        if not f.name.lower().endswith(".csv"):
            continue
        if "nifty" in f.name.lower():
            continue

        df = load_stock(f)
        prices[f.stem] = df["Close"]

    price_df = pd.DataFrame(prices)
    universe = price_df.count(axis=1)

    # ---------- DAILY RETURNS (%) ----------
    ret_pct = price_df.pct_change() * 100

    breadth = pd.DataFrame(index=price_df.index)

    # ---------- ADV / DECL / UNCH (PERCENT) ----------
    adv = (ret_pct > 0).sum(axis=1)
    dec = (ret_pct < 0).sum(axis=1)
    unch = (ret_pct == 0).sum(axis=1)

    breadth["adv_pct"] = (adv / universe * 100).round(2)
    breadth["dec_pct"] = (dec / universe * 100).round(2)
    breadth["unch_pct"] = (unch / universe * 100).round(2)

    # ---------- DMA BREADTH (PERCENT) ----------
    for w in DMA_WINDOWS:
        pct = (
            (price_df > price_df.rolling(w).mean()).sum(axis=1)
            / universe * 100
        )
        breadth[f"pct_above_{w}dma"] = pct.round(2)

    # ---------- RETURN DISTRIBUTION BUCKETS (PERCENT) ----------
    buckets = {
        "ret_gt_5_pct":      ret_pct > 5,
        "ret_4_5_pct":      (ret_pct > 4) & (ret_pct <= 5),
        "ret_3_4_pct":      (ret_pct > 3) & (ret_pct <= 4),
        "ret_2_3_pct":      (ret_pct > 2) & (ret_pct <= 3),
        "ret_1_2_pct":      (ret_pct > 1) & (ret_pct <= 2),
        "ret_0_1_pct":      (ret_pct > 0) & (ret_pct <= 1),
        "ret_m1_0_pct":     (ret_pct > -1) & (ret_pct <= 0),
        "ret_m2_m1_pct":    (ret_pct > -2) & (ret_pct <= -1),
        "ret_m3_m2_pct":    (ret_pct > -3) & (ret_pct <= -2),
        "ret_m4_m3_pct":    (ret_pct > -4) & (ret_pct <= -3),
        "ret_m5_m4_pct":    (ret_pct > -5) & (ret_pct <= -4),
        "ret_lt_m5_pct":     ret_pct <= -5,
    }

    for name, condition in buckets.items():
        breadth[name] = (condition.sum(axis=1) / universe * 100).round(2)

    # ---------- BREAKOUT / BREAKDOWN + NET HL (PERCENT) ----------
    for w in BREAKOUT_WINDOWS:
        # Avoid look-ahead bias
        rolling_high = price_df.shift(1).rolling(w).max()
        rolling_low  = price_df.shift(1).rolling(w).min()

        highs = (price_df > rolling_high).sum(axis=1)
        lows  = (price_df < rolling_low).sum(axis=1)

        high_pct = (highs / universe * 100)
        low_pct  = (lows / universe * 100)

        breadth[f"high_{w}d_pct"] = high_pct.round(2)
        breadth[f"low_{w}d_pct"] = low_pct.round(2)
        breadth[f"net_hl_{w}d_pct"] = ((highs - lows) / universe * 100).round(2)

    # ---------- OUTPUT ----------
    breadth.tail(LOOKBACK_DAILY).to_csv(OUTPUT_FILE)

    print("[SUCCESS] Base breadth dataset generated (percent-only + net HLs)")

if __name__ == "__main__":
    main()