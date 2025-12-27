from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

template = BASE_DIR / "dashboard.html.template"
target = OUTPUT_DIR / "dashboard.html"

csv = OUTPUT_DIR / "breadth_full_history.csv"
if not csv.exists():
    raise FileNotFoundError("breadth_full_history.csv not found in output/")

shutil.copy(template, target)
print(f"[INFO] HTML dashboard generated → {target}")