import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image

"""
This script generates a printable ArUco marker for use in hand-eye calibration.
The marker is saved as a high-resolution PDF that can be printed and mounted
on a small piece of cardboard, then placed inside the robot gripper.

To ensure accurate scale, the PDF should be printed at the specified DPI
(e.g., 300 DPI) so that the marker appears at the correct physical size 
(e.g., 50 mm x 50 mm).
"""

# --- Config ---
marker_id = 0
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_size_mm = 50  # Desired printed size in mm
dpi = 300  # Print resolution

# --- Generate marker ---
size_inch = marker_size_mm / 25.4
marker_size_px = int(size_inch * dpi)

marker_img = aruco.drawMarker(marker_dict, marker_id, marker_size_px)

# --- Add border ---
border_px = int(0.2 * marker_size_px)
bordered_img = cv2.copyMakeBorder(
    marker_img,
    border_px, border_px, border_px, border_px,
    borderType=cv2.BORDER_CONSTANT,
    value=255
)

# --- Convert to PIL and save as PDF ---
pil_img = Image.fromarray(bordered_img)
total_size_px = bordered_img.shape[0]  # square image
physical_size_inch = total_size_px / dpi

pdf_filename = f"aruco_marker_id{marker_id}_{marker_size_mm}mm.pdf"
pil_img.save(pdf_filename, "PDF", resolution=dpi)

print(f"Saved: {pdf_filename} — print at {dpi} DPI for true {marker_size_mm}mm size.")
