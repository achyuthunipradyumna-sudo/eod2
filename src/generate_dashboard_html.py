from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"

templates = {
    "breadth": "breadth.html.template",
    "trend": "trend.html.template",
    "volatility": "volatility.html.template",
}

for name, tpl in templates.items():
    src = BASE / tpl
    dst = OUT / f"{name}.html"
    shutil.copy(src, dst)
    print(f"[OK] Generated {dst.name}")