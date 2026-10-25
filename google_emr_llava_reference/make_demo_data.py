"""Create synthetic non-patient test data for the educational pipeline."""
from pathlib import Path

import pandas as pd
import pymupdf
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "demo_input"
OUT.mkdir(parents=True, exist_ok=True)

# Synthetic text-based PDF, with a small vector-drawn table.
doc = pymupdf.open()
page = doc.new_page(width=580, height=420)
page.insert_text((45, 55), "Synthetic Research Form", fontsize=17)
page.insert_text((45, 90), "This is fictional data for a programming demonstration.", fontsize=11)
for y in (120, 150, 180):
    page.draw_line((45, y), (420, y))
for x in (45, 200, 420):
    page.draw_line((x, 120), (x, 180))
page.insert_text((55, 141), "Variable", fontsize=11)
page.insert_text((215, 141), "Synthetic value", fontsize=11)
page.insert_text((55, 171), "Visits", fontsize=11)
page.insert_text((215, 171), "3", fontsize=11)
doc.save(OUT / "sample_document.pdf")
doc.close()

pd.DataFrame({"case_id": ["synthetic_a", "synthetic_b"],
              "visit_count": [3, 4], "service_category": ["A", "B"]}).to_csv(
    OUT / "sample_data.csv", index=False)
(OUT / "sample_note.txt").write_text(
    "Fictional note for document-processing tests. No real patient information.",
    encoding="utf-8"
)
image = Image.new("RGB", (700, 430), "white")
draw = ImageDraw.Draw(image)
draw.text((50, 50), "SYNTHETIC RESEARCH FORM", fill="black")
draw.rectangle((45, 110, 650, 330), outline="black", width=3)
draw.line((45, 190, 650, 190), fill="black", width=2)
draw.text((70, 145), "Field: demo category", fill="black")
draw.text((70, 240), "Value: example", fill="black")
image.save(OUT / "sample_scan.png")
print(f"Created synthetic files in {OUT}")
