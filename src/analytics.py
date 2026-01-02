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

AD_Z_WINDOW = 55
AD_MA_WINDOW = 5

RISING_LOOKBACK = {
    20: 3,
    50: 5,
    200: 20,
}

MOMENTUM_LOOKBACKS = {
    20: (3, 10),
    50: (5, 20),
    200: (20, 60),
}

OUTPUT_FILE = OUTPUT_DIR / "dma_breadth.csv"
AD_OUTPUT_FILE = OUTPUT_DIR / "ad_breadth.csv"
DIAG_FILE = OUTPUT_DIR / "universe_shrink_diagnostics.csv"

# ================= HELPERS =================
def load_stock(path: Path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date")[["Date", "Close"]]
    df.set_index("Date", inplace=True)
    return df

def diagnose_date(price_df, date):
    row = price_df.loc[date]
    total_files = price_df.shape[1]

    available = row.notna().sum()
    missing = total_files - available

    present = row[row.notna()].index.tolist()
    missing_stocks = row[row.isna()].index.tolist()

    return {
        "Date": date.strftime("%Y-%m-%d"),
        "Total Files": total_files,
        "Stocks Available": available,
        "Stocks Missing": missing,
        "Availability %": round(available / total_files * 100, 2),
        "Present Stocks (sample)": ", ".join(present[:10]),
        "Missing Stocks (sample)": ", ".join(missing_stocks[:10]),
    }

# ================= MAIN =================
def main():
    prices = {}

    # ---------- LOAD EQUITY FILES ----------
    for f in DATA_DIR.iterdir():
        if not f.name.lower().endswith(".csv"):
            continue
        if "nifty" in f.name.lower():
            continue
        prices[f.stem] = load_stock(f)["Close"]

    price_df = pd.DataFrame(prices)

    # ---------- UNIVERSE SHRINK DIAGNOSTICS ----------
    universe_raw = price_df.count(axis=1)
    pct_change = universe_raw.pct_change()

    diag_rows = []
    for date, chg in pct_change.items():
        if pd.notna(chg) and chg < -0.03:
            diag_rows.append(diagnose_date(price_df, date))

    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(DIAG_FILE, index=False)

    # ---------- ROLLING UNIVERSE INTEGRITY ----------
    rolling_peak = universe_raw.expanding().max()
    valid_universe = universe_raw >= 0.9 * rolling_peak
    price_df = price_df[valid_universe]

    universe = price_df.count(axis=1)

    breadth = pd.DataFrame(index=price_df.index)

    # ---------- METRICS ----------
    breadth["Total Stocks"] = universe

    dma = {w: price_df.rolling(w).mean() for w in DMA_WINDOWS}

    # Price vs DMA
    breadth["%>20DMA"]  = ((price_df > dma[20]).sum(axis=1) / universe * 100)
    breadth["%>50DMA"]  = ((price_df > dma[50]).sum(axis=1) / universe * 100)
    breadth["%>200DMA"] = ((price_df > dma[200]).sum(axis=1) / universe * 100)

    # Rising DMA
    breadth["%↑20DMA"]  = (((dma[20]  - dma[20].shift(3))   > 0).sum(axis=1) / universe * 100)
    breadth["%↑50DMA"]  = (((dma[50]  - dma[50].shift(5))   > 0).sum(axis=1) / universe * 100)
    breadth["%↑200DMA"] = (((dma[200] - dma[200].shift(20)) > 0).sum(axis=1) / universe * 100)

    # Momentum
    breadth["Δ%↑20DMA_3D"]   = breadth["%↑20DMA"]  - breadth["%↑20DMA"].shift(3)
    breadth["Δ%↑20DMA_10D"]  = breadth["%↑20DMA"]  - breadth["%↑20DMA"].shift(10)
    breadth["Δ%↑50DMA_5D"]   = breadth["%↑50DMA"]  - breadth["%↑50DMA"].shift(5)
    breadth["Δ%↑50DMA_20D"]  = breadth["%↑50DMA"]  - breadth["%↑50DMA"].shift(20)
    breadth["Δ%↑200DMA_20D"] = breadth["%↑200DMA"] - breadth["%↑200DMA"].shift(20)
    breadth["Δ%↑200DMA_60D"] = breadth["%↑200DMA"] - breadth["%↑200DMA"].shift(60)

    # =========================================================
    # ADVANCE / DECLINE (A/D) BREADTH
    # =========================================================

    # Daily price change
    daily_change = price_df.diff()

    advances = (daily_change > 0).sum(axis=1)
    declines = (daily_change < 0).sum(axis=1)

    ad = pd.DataFrame(index=price_df.index)

    ad["Universe"] = universe
    ad["Advances"] = advances
    ad["Declines"] = declines
    ad["Net_AD"] = advances - declines

    ad["Advance_%"] = advances / (advances + declines) * 100

    # A/D Line
    ad["AD_Line"] = ad["Net_AD"].cumsum()

    # 55-day z-score of Net A/D
    rolling_mean = ad["Net_AD"].rolling(AD_Z_WINDOW).mean()
    rolling_std = ad["Net_AD"].rolling(AD_Z_WINDOW).std()

    ad["Net_AD_z55"] = (ad["Net_AD"] - rolling_mean) / rolling_std

    # Net A/D Momentum
    ad_ma = ad["Net_AD"].rolling(AD_MA_WINDOW).mean()

    ad["ΔNet_AD_MA_3D"] = ad_ma - ad_ma.shift(3)
    ad["ΔNet_AD_MA_10D"] = ad_ma - ad_ma.shift(10)

    ad = ad.round(2)
    breadth = breadth.round(2)

    # ---------- MOVE DATE FROM INDEX TO COLUMN ----------
    breadth = breadth.reset_index()
    breadth.rename(columns={"index": "Date"}, inplace=True)
    breadth["Date"] = breadth["Date"].dt.strftime("%Y-%m-%d")

    # ---------- SORT (LATEST ON TOP) ----------
    breadth = breadth.sort_values("Date", ascending=False)

    # ---------- MOVE DATE FROM INDEX TO COLUMN ----------
    ad = ad.reset_index()
    ad.rename(columns={"index": "Date"}, inplace=True)
    ad["Date"] = ad["Date"].dt.strftime("%Y-%m-%d")

    # ---------- SORT (LATEST ON TOP) ----------
    ad = ad.sort_values("Date", ascending=False)


    # ---------- DESCRIPTION ROW (SECOND ROW) ----------
    DESCRIPTIONS = {
        "Date": "Trading date",
        "Total Stocks": "Number of equities with valid data on this date",

        "%>20DMA": "% of stocks with Price > 20DMA",
        "%>50DMA": "% of stocks with Price > 50DMA",
        "%>200DMA": "% of stocks with Price > 200DMA",

        "%↑20DMA": "% of stocks with rising 20DMA over last 3 days",
        "%↑50DMA": "% of stocks with rising 50DMA over last 5 days",
        "%↑200DMA": "% of stocks with rising 200DMA over last 20 days",

        "Δ%↑20DMA_3D": "Change in %↑20DMA over last 3 days",
        "Δ%↑20DMA_10D": "Change in %↑20DMA over last 10 days",
        "Δ%↑50DMA_5D": "Change in %↑50DMA over last 5 days",
        "Δ%↑50DMA_20D": "Change in %↑50DMA over last 20 days",
        "Δ%↑200DMA_20D": "Change in %↑200DMA over last 20 days",
        "Δ%↑200DMA_60D": "Change in %↑200DMA over last 60 days",
    }

    desc_row = pd.DataFrame(
        [[DESCRIPTIONS.get(col, "") for col in breadth.columns]],
        columns=breadth.columns
    )

    final_df = pd.concat([desc_row, breadth], ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    AD_DESCRIPTIONS = {
        "Date": "Trading date",
        "Universe": "Number of equities with valid data on this date",
        "Advances": "Number of stocks closing higher than previous close",
        "Declines": "Number of stocks closing lower than previous close",
        "Net_AD": "Advances minus Declines",
        "Advance_%": "Advances as % of (Advances + Declines)",
        "AD_Line": "Cumulative sum of Net Advances",
        "Net_AD_z55": "55-day z-score of Net Advances",
        "ΔNet_AD_MA_3D": "3-day momentum of 5-day MA of Net Advances",
        "ΔNet_AD_MA_10D": "10-day momentum of 5-day MA of Net Advances",
    }

    ad_desc = pd.DataFrame(
        [[AD_DESCRIPTIONS.get(col, "") for col in ad.columns]],
        columns=ad.columns
    )

    ad_final = pd.concat([ad_desc, ad], ignore_index=True)
    ad_final.to_csv(AD_OUTPUT_FILE, index=False)

    print("[SUCCESS] breadth_base.csv generated")
    print("[SUCCESS] ad_breadth.csv generated")

if __name__ == "__main__":
    main()
