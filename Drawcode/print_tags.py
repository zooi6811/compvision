import os
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from marker_lookup import id_to_grid  # your generated lookup file

# -------------------------
# Settings
# -------------------------
MARKER_SIZE_CM = 10       # marker physical size
PAGE_MARGIN_CM = 2        # page margin
GRID_SIZE = 4             # inner grid size
BORDER = 1                # black border thickness in cells
IDS_TO_PRINT = [6, 8]

# -------------------------
# Draw marker with border
# -------------------------
def draw_marker(c, x_cm, y_cm, grid):
    full_grid = GRID_SIZE + 2*BORDER
    cell_size_pt = MARKER_SIZE_CM * cm / full_grid

    for r in range(full_grid):
        for col in range(full_grid):
            # determine if this is a border cell
            if r < BORDER or r >= GRID_SIZE + BORDER or col < BORDER or col >= GRID_SIZE + BORDER:
                val = 1  # black border
            else:
                val = grid[r - BORDER, col - BORDER]  # inner grid
            c.setFillColorRGB(0,0,0) if val else c.setFillColorRGB(1,1,1)
            c.rect(
                x_cm*cm + col*cell_size_pt,
                y_cm*cm + (full_grid - 1 - r)*cell_size_pt,
                cell_size_pt, cell_size_pt, fill=1, stroke=0
            )

# -------------------------
# Create PDF
# -------------------------
def create_pdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "markers.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_width, page_height = A4

    positions_cm = [
        (PAGE_MARGIN_CM, page_height/cm - PAGE_MARGIN_CM - MARKER_SIZE_CM),
        (PAGE_MARGIN_CM, page_height/cm - PAGE_MARGIN_CM - 2*MARKER_SIZE_CM - PAGE_MARGIN_CM)
    ]

    for idx, marker_id in enumerate(IDS_TO_PRINT):
        grid = id_to_grid[marker_id]
        x_cm, y_cm = positions_cm[idx % 2]
        draw_marker(c, x_cm, y_cm, grid)
        if (idx + 1) % 2 == 0:
            c.showPage()  # new page after 2 markers

    # finish last page if odd
    if len(IDS_TO_PRINT) % 2 != 0:
        c.showPage()

    c.save()
    print(f"Saved PDF with markers to {pdf_path}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    create_pdf()
