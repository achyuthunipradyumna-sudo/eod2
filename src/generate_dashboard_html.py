from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

template = BASE_DIR / "dashboard.html.template"
target = OUTPUT_DIR / "dashboard.html"

shutil.copy(template, target)
print(f"[INFO] HTML dashboard generated → {target}")
