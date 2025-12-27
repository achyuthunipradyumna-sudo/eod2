import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

def load_stock_by_path(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

# ================= TREND =================
def compute_trend_structure(index_df):
    price = index_df["Close"]
    dma50 = price.rolling(50).mean()
    dma200 = price.rolling(200).mean()

    trend = pd.DataFrame(index=price.index)
    trend["price_above_50"] = (price > dma50).astype(int)
    trend["price_above_200"] = (price > dma200).astype(int)
    trend["dma200_slope_pct"] = dma200.pct_change(20) * 100
    trend["dist_200_z"] = zscore(price / dma200 - 1, 200)
    trend["persistence_200"] = (price > dma200).rolling(250).mean() * 100

    return trend

# ================= VOLATILITY =================
def compute_index_volatility(index_df):
    ret = np.log(index_df["Close"] / index_df["Close"].shift(1))
    vol = pd.DataFrame(index=index_df.index)

    for w in VOL_WINDOWS:
        rv = ret.rolling(w).std() * np.sqrt(252)
        vol[f"vol_{w}"] = rv
        vol[f"vol_{w}_z"] = zscore(rv, 200)

    return vol

# ================= MOMENTUM =================
def compute_index_momentum(index_df):
    price = index_df["Close"]
    mom = pd.DataFrame(index=price.index)

    mom["mom_63"] = price.pct_change(63)
    mom["mom_126"] = price.pct_change(126)
    mom["mom_12_1"] = price.pct_change(252) - price.pct_change(21)
    mom["mom_12_1_z"] = zscore(mom["mom_12_1"], 200)

    return mom

# ================= DASHBOARD PLOTTING =================
def style_axis(ax, title):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)

def plot_index_dashboard(df, index_key):
    df = df.tail(LOOKBACK_DAILY)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"hspace": 0.25}
    )

    # ---------- PANEL 1: BREADTH ----------
    axes[0].plot(df.index, df["ad_z_50"], label="AD Z(50)", color="tab:blue", linewidth=2)
    axes[0].plot(df.index, df["pct_above_50dma_z"], label="%>50DMA Z", color="tab:green", alpha=0.8)
    style_axis(axes[0], "Panel 1 — Short-Term Breadth & Participation")
    axes[0].legend(loc="upper left")

    # ---------- PANEL 2: TREND + MOMENTUM ----------
    axes[1].plot(df.index, df[f"{index_key}_dist_200_z"], label="Distance from 200DMA (Z)", color="tab:orange", linewidth=2)
    axes[1].plot(df.index, df[f"{index_key}_mom_126"], label="6M Momentum", color="tab:red", alpha=0.7)
    style_axis(axes[1], "Panel 2 — Trend Strength & Medium-Term Momentum")
    axes[1].legend(loc="upper left")

    # ---------- PANEL 3: VOL + STRUCTURE ----------
    axes[2].plot(df.index, df["nhnl_z"], label="NH-NL Z", color="tab:purple", linewidth=2)
    axes[2].plot(df.index, df[f"{index_key}_vol_20_z"], label="Volatility Z(20)", color="tab:brown", alpha=0.7)
    style_axis(axes[2], "Panel 3 — Volatility & Market Structure")
    axes[2].legend(loc="upper left")

    # ---------- X-AXIS ----------
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        f"{index_key.upper()} — Market Regime Dashboard",
        fontsize=14,
        weight="bold"
    )

    out = OUTPUT_DIR / f"{index_key}_3panel_dashboard.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"[INFO] Dashboard saved → {out.name}")

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

        df = load_stock_by_path(f)
        prices[f.stem] = df["Close"]

        ret = np.log(df["Close"] / df["Close"].shift(1))
        for w in VOL_WINDOWS:
            stock_vols[w][f.stem] = ret.rolling(w).std() * np.sqrt(252)

    price_df = pd.DataFrame(prices)

    # ---------- BREADTH ----------
    ret = price_df.diff()
    ad = (ret > 0).sum(axis=1) - (ret < 0).sum(axis=1)

    breadth = pd.DataFrame({"ad": ad})
    breadth["ad_z_50"] = zscore(ad, 50)

    for w in DMA_WINDOWS:
        pct = (price_df > price_df.rolling(w).mean()).mean(axis=1) * 100
        breadth[f"pct_above_{w}dma"] = pct
        breadth[f"pct_above_{w}dma_z"] = zscore(pct, 200)

    highs = price_df.rolling(NHNL_LOOKBACK).max()
    lows = price_df.rolling(NHNL_LOOKBACK).min()
    nhnl = (price_df >= highs).sum(axis=1) - (price_df <= lows).sum(axis=1)
    breadth["nhnl_z"] = zscore(nhnl, NHNL_LOOKBACK)

    for w in VOL_WINDOWS:
        vol_df = pd.DataFrame(stock_vols[w])
        med = vol_df.median(axis=1)
        breadth[f"median_stock_vol_{w}"] = med
        breadth[f"median_stock_vol_{w}_z"] = zscore(med, 200)

    # ---------- INDEX LAYERS + DASHBOARDS ----------
    for name, key in INDEX_KEYS.items():
        idx = load_stock_by_path(resolve_index_file(key))

        breadth = breadth.join(compute_trend_structure(idx).add_prefix(f"{name}_"))
        breadth = breadth.join(compute_index_volatility(idx).add_prefix(f"{name}_"))
        breadth = breadth.join(compute_index_momentum(idx).add_prefix(f"{name}_"))

        plot_index_dashboard(breadth, name)

    # ---------- STATE FILES ----------
    today = breadth.index[-1]
    breadth.loc[[today]].to_csv(OUTPUT_DIR / f"breadth_{today.date()}.csv")
    breadth.tail(LOOKBACK_DAILY).to_csv(LOOKBACK_FILE)

    if today.day == 1:
        if FULL_HISTORY_FILE.exists():
            full = pd.read_csv(FULL_HISTORY_FILE, parse_dates=["Date"], index_col="Date")
            breadth = pd.concat([full, breadth])
        breadth = breadth[~breadth.index.duplicated(keep="last")]
        breadth.sort_index(inplace=True)
        breadth.to_csv(FULL_HISTORY_FILE)

    print("[SUCCESS] State files + readable dashboards generated")

if __name__ == "__main__":
    main()