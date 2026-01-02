import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent

AD_FILE  = BASE_DIR / "output" / "ad_breadth.csv"
DMA_FILE = BASE_DIR / "output" / "dma_breadth.csv"

PLOTS_DIR = BASE_DIR / "plots"
AD_PLOT_DIR  = PLOTS_DIR / "ad"
DMA_PLOT_DIR = PLOTS_DIR / "dma"

AD_PLOT_DIR.mkdir(parents=True, exist_ok=True)
DMA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 5)

TACTICAL_DAYS = 90
STRUCTURAL_DAYS = 250

# ================= NUMERIC COLUMNS =================
AD_NUMERIC_COLS = [
    "Universe","Advances","Declines","Net_AD","Advance_%",
    "AD_Line","Net_AD_z55","ΔNet_AD_MA_3D","ΔNet_AD_MA_10D"
]

DMA_NUMERIC_COLS = [
    "%>20DMA","%>50DMA","%>200DMA",
    "%↑20DMA","%↑50DMA","%↑200DMA",
    "Δ%↑20DMA_3D","Δ%↑20DMA_10D",
    "Δ%↑50DMA_5D","Δ%↑50DMA_20D",
    "Δ%↑200DMA_20D","Δ%↑200DMA_60D"
]

# ================= LOAD CSV =================
def load_csv(path, numeric_cols):
    df = pd.read_csv(path)
    df = df[df["Date"] != "Trading date"]
    df["Date"] = pd.to_datetime(df["Date"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("Date").dropna()
    return df

ad  = load_csv(AD_FILE, AD_NUMERIC_COLS)
dma = load_csv(DMA_FILE, DMA_NUMERIC_COLS)

ad_90  = ad.tail(TACTICAL_DAYS)
ad_250 = ad.tail(STRUCTURAL_DAYS)
dma_90 = dma.tail(TACTICAL_DAYS)

# ================= VISUAL HELPERS =================
def pct_bands():
    plt.axhspan(70, 100, color="green", alpha=0.18)
    plt.axhspan(60, 70,  color="green", alpha=0.08)
    plt.axhspan(40, 60,  color="grey",  alpha=0.05)
    plt.axhspan(30, 40,  color="red",   alpha=0.08)
    plt.axhspan(0, 30,   color="red",   alpha=0.18)

def zscore_bands():
    plt.axhspan(2, 5,    color="green", alpha=0.18)
    plt.axhspan(1, 2,    color="green", alpha=0.08)
    plt.axhspan(-1, 1,   color="grey",  alpha=0.05)
    plt.axhspan(-2, -1,  color="red",   alpha=0.08)
    plt.axhspan(-5, -2,  color="red",   alpha=0.18)

def momentum_fill(x, y):
    plt.fill_between(x, y, 0, where=y > 0, color="green", alpha=0.08)
    plt.fill_between(x, y, 0, where=y < 0, color="red",   alpha=0.08)

def annotate_latest(x, y, label):
    plt.annotate(
        f"{label}: {y.iloc[-1]:.2f}",
        xy=(x.iloc[-1], y.iloc[-1]),
        xytext=(6, 0),          # ← RIGHT of last point (restored)
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="center"
    )

def add_note(text):
    plt.text(
        0.98, 0.98,
        text,
        transform=plt.gca().transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="grey",
            alpha=0.85
        )
    )

# ================= A/D BREADTH =================

# --- Net Advances ---
plt.figure()
plt.plot(ad_90["Date"], ad_90["Net_AD"], color="black")
plt.axhline(0)
annotate_latest(ad_90["Date"], ad_90["Net_AD"], "Net AD")
add_note(
    "Measures daily participation\n"
    "Above 0 = broad buying\n"
    "Below 0 = broad selling\n"
    "Persistent negatives = distribution"
)
plt.title("Net Advances (Last 90 Trading Days)")
plt.savefig(AD_PLOT_DIR / "01_net_ad_90d.png", dpi=150)
plt.close()

# --- Advance % ---
plt.figure()
pct_bands()
plt.plot(ad_90["Date"], ad_90["Advance_%"], color="black")
annotate_latest(ad_90["Date"], ad_90["Advance_%"], "Advance %")
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
add_note(
    "% of stocks advancing\n"
    ">60% = accumulation\n"
    "40–60% = neutral\n"
    "<40% = weak participation"
)
plt.title("Advance Percentage (Last 90 Trading Days)")
plt.savefig(AD_PLOT_DIR / "02_advance_pct_90d.png", dpi=150)
plt.close()

# --- A/D Line ---
ad_norm = ad_250["AD_Line"] - ad_250["AD_Line"].iloc[0]
ad_ma   = ad_norm.rolling(20).mean()

plt.figure()
plt.plot(ad_250["Date"], ad_norm, label="Normalized A/D", color="black")
plt.plot(ad_250["Date"], ad_ma, label="20D MA", linewidth=2)
plt.axhline(0)
annotate_latest(ad_250["Date"], ad_norm, "A/D")
add_note(
    "Cumulative participation trend\n"
    "Rising slope = accumulation\n"
    "Falling slope = distribution\n"
    "Structural, slow signal"
)
plt.legend()
plt.title("Advance–Decline Line (Normalized, 250 Trading Days)")
plt.savefig(AD_PLOT_DIR / "03_ad_line_250d.png", dpi=150)
plt.close()

# --- z-score ---
plt.figure()
zscore_bands()
plt.plot(ad_90["Date"], ad_90["Net_AD_z55"], color="black")
annotate_latest(ad_90["Date"], ad_90["Net_AD_z55"], "z55")
add_note(
    "Participation vs recent history\n"
    "< −1 = weak breadth\n"
    "< −2 = capitulation\n"
    "Mean-reverting indicator"
)
plt.title("Net A/D z-Score (55-day)")
plt.savefig(AD_PLOT_DIR / "04_net_ad_z55_90d.png", dpi=150)
plt.close()

# --- A/D Momentum ---
plt.figure()
momentum_fill(ad_90["Date"], ad_90["ΔNet_AD_MA_3D"])
plt.plot(ad_90["Date"], ad_90["ΔNet_AD_MA_3D"], label="Δ3D")
plt.plot(ad_90["Date"], ad_90["ΔNet_AD_MA_10D"], label="Δ10D")
annotate_latest(ad_90["Date"], ad_90["ΔNet_AD_MA_3D"], "Δ3D")
annotate_latest(ad_90["Date"], ad_90["ΔNet_AD_MA_10D"], "Δ10D")
plt.axhline(0)
add_note(
    "Acceleration of participation\n"
    "Above 0 = selling easing\n"
    "Below 0 = selling accelerating\n"
    "Leading signal"
)
plt.legend()
plt.title("A/D Breadth Momentum")
plt.savefig(AD_PLOT_DIR / "05_breadth_momentum_90d.png", dpi=150)
plt.close()

# ================= DMA BREADTH =================

# --- DMA Level Breadth ---
plt.figure()
pct_bands()
plt.plot(dma_90["Date"], dma_90["%>20DMA"], label="%>20DMA")
plt.plot(dma_90["Date"], dma_90["%>50DMA"], label="%>50DMA")
plt.plot(dma_90["Date"], dma_90["%>200DMA"], label="%>200DMA")
annotate_latest(dma_90["Date"], dma_90["%>20DMA"], ">20")
annotate_latest(dma_90["Date"], dma_90["%>50DMA"], ">50")
annotate_latest(dma_90["Date"], dma_90["%>200DMA"], ">200")
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
add_note(
    "% of stocks above key MAs\n"
    "High = broad participation\n"
    "Low = narrow market\n"
    "Defines regime"
)
plt.legend()
plt.title("DMA Level Breadth (Last 90 Trading Days)")
plt.savefig(DMA_PLOT_DIR / "01_dma_level_90d.png", dpi=150)
plt.close()

# --- DMA Trend Participation (20/50) ---
plt.figure()
plt.plot(dma_90["Date"], dma_90["%↑20DMA"], label="%↑20DMA")
plt.plot(dma_90["Date"], dma_90["%↑50DMA"], label="%↑50DMA")
annotate_latest(dma_90["Date"], dma_90["%↑20DMA"], "↑20")
annotate_latest(dma_90["Date"], dma_90["%↑50DMA"], "↑50")
plt.axhline(50, linestyle="--")
add_note(
    "% of stocks with rising short MAs\n"
    "Rising = trend health improving\n"
    "Falling = internal deterioration"
)
plt.legend()
plt.title("DMA Trend Participation (20/50)")
plt.savefig(DMA_PLOT_DIR / "02_dma_trend_20_50.png", dpi=150)
plt.close()

# --- DMA Trend Participation (200) ---
plt.figure()
plt.plot(dma_90["Date"], dma_90["%↑200DMA"], color="black")
annotate_latest(dma_90["Date"], dma_90["%↑200DMA"], "↑200")
plt.axhline(50, linestyle="--")
add_note(
    "Long-term trend participation\n"
    "Very slow-moving indicator\n"
    "Confirms major regime shifts"
)
plt.title("DMA Trend Participation (200DMA)")
plt.savefig(DMA_PLOT_DIR / "03_dma_trend_200.png", dpi=150)
plt.close()

# --- DMA Momentum 20DMA ---
plt.figure()
momentum_fill(dma_90["Date"], dma_90["Δ%↑20DMA_3D"])
plt.plot(dma_90["Date"], dma_90["Δ%↑20DMA_3D"], label="Δ3D")
plt.plot(dma_90["Date"], dma_90["Δ%↑20DMA_10D"], label="Δ10D")
annotate_latest(dma_90["Date"], dma_90["Δ%↑20DMA_3D"], "Δ3D")
annotate_latest(dma_90["Date"], dma_90["Δ%↑20DMA_10D"], "Δ10D")
plt.axhline(0)
add_note(
    "Short-term breadth momentum\n"
    "Positive = improvement\n"
    "Negative = deterioration"
)
plt.legend()
plt.title("20DMA Breadth Momentum")
plt.savefig(DMA_PLOT_DIR / "04_dma_mom_20.png", dpi=150)
plt.close()

# --- DMA Momentum 50DMA ---
plt.figure()
momentum_fill(dma_90["Date"], dma_90["Δ%↑50DMA_5D"])
plt.plot(dma_90["Date"], dma_90["Δ%↑50DMA_5D"], label="Δ5D")
plt.plot(dma_90["Date"], dma_90["Δ%↑50DMA_20D"], label="Δ20D")
annotate_latest(dma_90["Date"], dma_90["Δ%↑50DMA_5D"], "Δ5D")
annotate_latest(dma_90["Date"], dma_90["Δ%↑50DMA_20D"], "Δ20D")
plt.axhline(0)
add_note(
    "Medium-term breadth momentum\n"
    "Confirms trend strength"
)
plt.legend()
plt.title("50DMA Breadth Momentum")
plt.savefig(DMA_PLOT_DIR / "05_dma_mom_50.png", dpi=150)
plt.close()

# --- DMA Momentum 200DMA ---
plt.figure()
momentum_fill(dma_90["Date"], dma_90["Δ%↑200DMA_20D"])
plt.plot(dma_90["Date"], dma_90["Δ%↑200DMA_20D"], label="Δ20D")
plt.plot(dma_90["Date"], dma_90["Δ%↑200DMA_60D"], label="Δ60D")
annotate_latest(dma_90["Date"], dma_90["Δ%↑200DMA_20D"], "Δ20D")
annotate_latest(dma_90["Date"], dma_90["Δ%↑200DMA_60D"], "Δ60D")
plt.axhline(0)
add_note(
    "Long-term breadth momentum\n"
    "Very slow signal\n"
    "Confirms structural shifts"
)
plt.legend()
plt.title("200DMA Breadth Momentum")
plt.savefig(DMA_PLOT_DIR / "06_dma_mom_200.png", dpi=150)
plt.close()

print("[SUCCESS] A/D and DMA breadth plots generated with top-right notes and clean labels.")
