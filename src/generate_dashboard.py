import pandas as pd
from pathlib import Path
import shutil

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "output" / "ad_breadth.csv"

PLOTS_REL_PATH = "plots/ad"   # relative to dashboard/
DASHBOARD_DIR = BASE_DIR / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)
OUTPUT_HTML = DASHBOARD_DIR / "index.html"

SRC_PLOTS = BASE_DIR / "plots" / "ad"
DST_PLOTS = DASHBOARD_DIR / "plots" / "ad"
DST_PLOTS.mkdir(parents=True, exist_ok=True)

# Copy plots into dashboard (robust for file:// and GitHub Pages)
for img in SRC_PLOTS.glob("*.png"):
    shutil.copy(img, DST_PLOTS / img.name)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_FILE)
df = df[df["Date"] != "Trading date"]
df["Date"] = pd.to_datetime(df["Date"])

numeric_cols = [
    "Universe",
    "Advances",
    "Declines",
    "Net_AD",
    "Advance_%",
    "AD_Line",
    "Net_AD_z55",
    "ΔNet_AD_MA_3D",
    "ΔNet_AD_MA_10D",
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
df = df.sort_values("Date").dropna()

latest = df.iloc[-1]

# ================= STATE LOGIC =================
def derive_regime(row):
    if row["Net_AD_z55"] < -1:
        return "Distribution"
    if row["Net_AD_z55"] > 1:
        return "Expansion"
    return "Neutral"

def derive_participation(row):
    if row["Advance_%"] < 40:
        return "Weak"
    if row["Advance_%"] > 60:
        return "Strong"
    return "Neutral"

def derive_momentum(row):
    if row["ΔNet_AD_MA_3D"] < 0 and row["ΔNet_AD_MA_10D"] < 0:
        return "Selling accelerating"
    if row["ΔNet_AD_MA_3D"] > 0 and row["ΔNet_AD_MA_10D"] > 0:
        return "Selling pressure easing"
    return "Mixed"

def derive_risk(regime, momentum):
    if regime == "Distribution" and momentum.startswith("Selling"):
        return "Defensive"
    if regime == "Expansion":
        return "Risk-On"
    return "Neutral"

state = {
    "date": latest["Date"].strftime("%Y-%m-%d"),
    "regime": derive_regime(latest),
    "participation": derive_participation(latest),
    "momentum": derive_momentum(latest),
}
state["risk"] = derive_risk(state["regime"], state["momentum"])

# ================= COLOR MAPS =================
REGIME_CLASS = {
    "Distribution": "red",
    "Expansion": "green",
    "Neutral": "grey",
}

PARTICIPATION_CLASS = {
    "Weak": "red",
    "Strong": "green",
    "Neutral": "grey",
}

RISK_CLASS = {
    "Defensive": "red",
    "Risk-On": "green",
    "Neutral": "grey",
}

# ================= HTML TEMPLATE =================
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>A/D Breadth Dashboard</title>

<style>
body {{
    font-family: Arial, sans-serif;
    background: #fafafa;
    margin: 40px;
    color: #222;
}}

h1 {{
    margin-bottom: 6px;
}}

.badge {{
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 13px;
}}

.green {{ background: #dff0d8; color: #2e7d32; }}
.red   {{ background: #f8d7da; color: #842029; }}
.grey  {{ background: #eeeeee; color: #444; }}

.summary {{
    background: #ffffff;
    padding: 16px;
    border-left: 6px solid #444;
    margin-bottom: 30px;
    width: fit-content;
}}

.summary-row {{
    margin-bottom: 6px;
}}

.section {{
    margin-bottom: 36px;
}}

.section img {{
    width: 100%;
    max-width: 1100px;
    border: 1px solid #ddd;
    background: #fff;
}}

.caption {{
    font-size: 14px;
    color: #555;
    margin: 6px 0 10px 0;
}}

.footer {{
    margin-top: 50px;
    font-size: 12px;
    color: #777;
}}
</style>
</head>

<body>

<h1>Advance–Decline Breadth Dashboard</h1>
<p><b>Date:</b> {state["date"]}</p>

<div class="summary">
  <div class="summary-row">
    <b>Structural regime</b> :
    <span class="badge {REGIME_CLASS[state["regime"]]}">{state["regime"]}</span>
  </div>

  <div class="summary-row">
    <b>Participation</b> :
    <span class="badge {PARTICIPATION_CLASS[state["participation"]]}">
      {state["participation"]}
    </span>
  </div>

  <div class="summary-row">
    <b>Momentum</b> :
    <span class="badge {REGIME_CLASS["Distribution" if state["momentum"].startswith("Selling") else "Neutral"]}">
      {state["momentum"]}
    </span>
  </div>

  <div class="summary-row">
    <b>Risk posture</b> :
    <span class="badge {RISK_CLASS[state["risk"]]}">
      {state["risk"]}
    </span>
  </div>
</div>

<div class="section">
  <div class="caption"><b>Structural participation</b> — Advance–Decline Line (normalized)</div>
  <img src="{PLOTS_REL_PATH}/03_ad_line_250d.png">
</div>

<div class="section">
  <div class="caption"><b>Daily participation</b> — Net Advances</div>
  <img src="{PLOTS_REL_PATH}/01_net_ad_90d.png">
</div>

<div class="section">
  <div class="caption"><b>Participation quality</b> — Advance Percentage</div>
  <img src="{PLOTS_REL_PATH}/02_advance_pct_90d.png">
</div>

<div class="section">
  <div class="caption"><b>Breadth regime</b> — Net A/D z-score (55-day)</div>
  <img src="{PLOTS_REL_PATH}/04_net_ad_z55_90d.png">
</div>

<div class="section">
  <div class="caption"><b>Momentum</b> — Acceleration / deceleration of participation</div>
  <img src="{PLOTS_REL_PATH}/05_breadth_momentum_90d.png">
</div>

<div class="footer">
Generated automatically from A/D breadth data.<br>
Colours encode regime and risk. Charts show participation, not price.
</div>

</body>
</html>
"""

# ================= WRITE FILE =================
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML_TEMPLATE)

print(f"[SUCCESS] Dashboard generated → {OUTPUT_HTML}")
