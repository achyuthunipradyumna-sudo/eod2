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
VOL_WINDOWS = [20, 50]
NHNL_LOOKBACK = 252
LOOKBACK_DAILY = 260

FULL_HISTORY_FILE = OUTPUT_DIR / "breadth_full_history.csv"
LOOKBACK_FILE = OUTPUT_DIR / f"breadth_lookback_{LOOKBACK_DAILY}.csv"

INDEX_KEYS = {
    "nifty50": "nifty 50",
    "nifty_total": "nifty total market",
}

# ================= HELPERS =================
def zscore(series, window):
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def resolve_index_file(index_name: str) -> Path:
    index_name = index_name.lower().strip()
    matches = []

    for f in DATA_DIR.iterdir():
        if not f.name.lower().endswith(".csv"):
            continue
        name = f.name.lower().replace(".csv", "").strip()
        if any(x in name for x in ["futures", "arbitrage", "tr"]):
            continue
        if name == index_name:
            matches.append(f)

    if len(matches) != 1:
        raise RuntimeError(f"[ERROR] Index resolution failed for {index_name}")

    return matches[0]

def load_stock(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================= MAIN =================
def main():
    prices = {}
    stock_vols = {20: {}, 50: {}}

    # ---------- STOCK UNIVERSE ----------
    for f in DATA_DIR.iterdir():
        if not f.name.lower().endswith(".csv"):
            continue
        if "nifty" in f.name.lower():
            continue

        df = load_stock(f)
        prices[f.stem] = df["Close"]

        ret = np.log(df["Close"] / df["Close"].shift(1))
        for w in VOL_WINDOWS:
            stock_vols[w][f.stem] = ret.rolling(w).std() * np.sqrt(252)

    price_df = pd.DataFrame(prices)
    universe = price_df.count(axis=1)

    # ---------- ADV / DECL ----------
    ret = price_df.diff()
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    ad = adv - dec

    breadth = pd.DataFrame(index=price_df.index)
    breadth["adv"] = adv
    breadth["dec"] = dec
    breadth["ad"] = ad
    breadth["ad_z_50"] = zscore(ad, 50)

    # ---------- % ABOVE DMAs ----------
    for w in DMA_WINDOWS:
        pct = (price_df > price_df.rolling(w).mean()).sum(axis=1) / universe * 100
        breadth[f"pct_above_{w}dma"] = pct
        breadth[f"pct_above_{w}dma_z"] = zscore(pct, 200)
        breadth[f"pct_above_{w}dma_chg"] = pct.diff(20)

    # ---------- NEW HIGHS / LOWS ----------
    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()

    nh = (price_df >= highs).sum(axis=1)
    nl = (price_df <= lows).sum(axis=1)
    nhnl = nh - nl

    breadth["nh"] = nh
    breadth["nl"] = nl
    breadth["nhnl"] = nhnl
    breadth["nhnl_z"] = zscore(nhnl, NHNL_LOOKBACK)

    # ---------- VOLATILITY BREADTH ----------
    for w in VOL_WINDOWS:
        vol_df = pd.DataFrame(stock_vols[w])
        med = vol_df.median(axis=1)
        breadth[f"median_stock_vol_{w}"] = med
        breadth[f"median_stock_vol_{w}_z"] = zscore(med, 200)

    # ---------- INDEX LAYERS ----------
    for name, key in INDEX_KEYS.items():
        idx = load_stock(resolve_index_file(key))
        price = idx["Close"]
        dma200 = price.rolling(200).mean()
        ret = np.log(price / price.shift(1))

        breadth[f"{name}_dist_200_z"] = zscore(price / dma200 - 1, 200)
        breadth[f"{name}_dma200_slope_pct"] = dma200.pct_change(20) * 100
        breadth[f"{name}_persistence_200"] = (price > dma200).rolling(250).mean() * 100
        breadth[f"{name}_vol_20_z"] = zscore(ret.rolling(20).std() * np.sqrt(252), 200)
        breadth[f"{name}_vol_50_z"] = zscore(ret.rolling(50).std() * np.sqrt(252), 200)
        breadth[f"{name}_mom_126"] = price.pct_change(126)

    # ---------- OUTPUT ----------
    today = breadth.index[-1]

    breadth.loc[[today]].to_csv(OUTPUT_DIR / f"breadth_{today.date()}.csv")
    breadth.tail(LOOKBACK_DAILY).to_csv(LOOKBACK_FILE)

    if FULL_HISTORY_FILE.exists():
        full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
        breadth = pd.concat([full, breadth])

    breadth = breadth[~breadth.index.duplicated(keep="last")]
    breadth.sort_index(inplace=True)
    breadth.to_csv(FULL_HISTORY_FILE)

    print("[SUCCESS] Raw + Z-score breadth fully generated")

if __name__ == "__main__":
    main()
