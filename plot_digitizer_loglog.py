#!/usr/bin/env python3
"""
plot_digitizer_loglog.py

Digitize data from a log-log plot screenshot by clicking calibration points and tracing the curve.

Usage:
    python plot_digitizer_loglog.py path/to/your_image.png

Workflow:
1) The image opens.
2) Click TWO known x-axis tick marks (left->right). After each click, you'll be prompted
   to type their numeric x values (e.g., 0.1 then 10). These are data values, not logs.
3) Click TWO known y-axis tick marks (bottom->top). Type their numeric y values.
4) Click along the curve to trace it. Press ENTER when you're done.
5) The script will fit a smooth spline along your clicks (optional) and sample N points
   along the path to export as CSV (x,y). You can also choose to export raw clicked points.

Assumptions:
- Axes are straight and aligned with the image borders (no rotation). If the image is rotated,
  rotate it in an image editor first.
- The plot uses true log10 scales on both axes.
- You can clearly identify at least two tick marks with known numeric values on each axis.
- The data curve is visible enough to click along it.

Outputs:
- A CSV saved next to the image with suffix _digitized.csv
- An overlaid PNG showing the sampled points with suffix _digitized_overlay.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy.interpolate import splprep, splev

def prompt_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a number.")

def pixel_to_logaffine(p1_pix, p2_pix, v1, v2):
    """
    Build an affine transform mapping pixel coordinate -> log10(value).
    For x: p is horizontal pixel; for y: p is vertical pixel.
    v1, v2 are numeric data values (NOT logs) at pixel positions p1_pix, p2_pix.
    Returns (a, b) such that log10(value) = a * p + b.
    """
    lp1 = np.log10(v1)
    lp2 = np.log10(v2)
    if p1_pix == p2_pix:
        raise ValueError("Two calibration pixels must not be equal.")
    a = (lp2 - lp1) / (p2_pix - p1_pix)
    b = lp1 - a * p1_pix
    return a, b

def main(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(1)

    img = imread(img_path)
    h, w = img.shape[0], img.shape[1]
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Step 1/4: Click TWO known x-axis ticks (LEFT to RIGHT)")
    plt.axis('on')
    plt.tight_layout()
    plt.show(block=False)

    # --- X calibration clicks ---
    print("Click TWO known x-axis ticks (left to right).")
    x_clicks = plt.ginput(2, timeout=0)
    if len(x_clicks) < 2:
        print("Not enough clicks for X calibration. Exiting.")
        sys.exit(1)
    (x1, yx1), (x2, yx2) = x_clicks
    print(f"X ticks at pixels: {x1:.1f}, {x2:.1f}")
    xv1 = prompt_float("Enter numeric value for the FIRST x tick (not log): ")
    xv2 = prompt_float("Enter numeric value for the SECOND x tick (not log): ")
    ax.plot([x1, x2], [yx1, yx2], 'o-', lw=1)
    ax.set_title("Step 2/4: Click TWO known y-axis ticks (BOTTOM to TOP)")
    fig.canvas.draw()
    plt.pause(0.1)

    # --- Y calibration clicks ---
    print("Click TWO known y-axis ticks (bottom to top).")
    y_clicks = plt.ginput(2, timeout=0)
    if len(y_clicks) < 2:
        print("Not enough clicks for Y calibration. Exiting.")
        sys.exit(1)
    (xy1, y1), (xy2, y2) = y_clicks
    print(f"Y ticks at pixels: {y1:.1f}, {y2:.1f}")
    yv1 = prompt_float("Enter numeric value for the FIRST y tick (not log): ")
    yv2 = prompt_float("Enter numeric value for the SECOND y tick (not log): ")
    ax.plot([xy1, xy2], [y1, y2], 'o-', lw=1)
    ax.set_title("Step 3/4: Click ALONG the curve to trace it; press ENTER when done")
    fig.canvas.draw()
    plt.pause(0.1)

    # Build pixel->log transforms
    try:
        ax_a, ax_b = pixel_to_logaffine(x1, x2, xv1, xv2)  # for horizontal pixel -> log10(x)
        ay_a, ay_b = pixel_to_logaffine(y1, y2, yv1, yv2)  # for vertical pixel -> log10(y)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # --- Curve clicks ---
    print("Click along the curve; press ENTER when done.")
    curve = plt.ginput(n=-1, timeout=0)
    if len(curve) < 2:
        print("Not enough points clicked to trace a curve. Exiting.")
        sys.exit(1)
    curve = np.array(curve)
    xs_pix, ys_pix = curve[:,0], curve[:,1]

    # Convert pixels to data (log-log mapping)
    # Note: In image coords, y increases downward. We built ay mapping using clicked tick pixels directly,
    # so we apply it to the image y pixels as-is.
    logx = ax_a * xs_pix + ax_b
    logy = ay_a * ys_pix + ay_b
    x_raw = 10**logx
    y_raw = 10**logy

    # Ask about smoothing/sampling
    print("\nSmoothing options:")
    print("1) Export raw clicked points only")
    print("2) Fit a spline and sample N points along pixel-distance")
    choice = input("Choose 1 or 2 [default 2]: ").strip() or "2"

    out_base = os.path.splitext(img_path)[0] + "_digitized"
    csv_path = out_base + ".csv"
    overlay_path = out_base + "_overlay.png"

    if choice == "1":
        xs_out, ys_out = x_raw, y_raw
    else:
        # Fit a spline to the clicked pixels in pixel space to get a nicely parameterized path
        # then sample along, convert to data coordinates.
        # Prepare points in order of clicking
        pts = np.vstack([xs_pix, ys_pix])
        # Smoothness parameter s can be tuned; 0 for interpolating spline; higher to smooth.
        try:
            tck, u = splprep(pts, s=len(xs_pix)*0.5)  # mild smoothing
            u_fine = np.linspace(0, 1, 300)
            xs_s, ys_s = splev(u_fine, tck)
            xs_s = np.array(xs_s)
            ys_s = np.array(ys_s)
        except Exception as e:
            print(f"Warning: spline fitting failed ({e}). Falling back to raw points.")
            xs_s, ys_s = xs_pix, ys_pix

        logx_s = ax_a * xs_s + ax_b
        logy_s = ay_a * ys_s + ay_b
        xs_out = 10**logx_s
        ys_out = 10**logy_s

    # Save CSV
    data = np.column_stack([xs_out, ys_out])
    header = "x,y"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments='', fmt="%.10g")
    print(f"\nSaved CSV: {csv_path}")

    # Plot overlay
    fig2, ax2 = plt.subplots()
    ax2.imshow(img)
    ax2.plot(xs_pix, ys_pix, 'o', ms=3, label="Clicked")
    ax2.plot((10**((np.arange(2)) - 1)), (10**((np.arange(2)) - 1)), alpha=0)  # no-op to keep legend simple
    if choice != "1":
        ax2.plot(np.interp(np.linspace(0, len(xs_out)-1, len(xs_out)), np.arange(len(xs_out)), xs_s if 'xs_s' in locals() else xs_pix),
                 np.interp(np.linspace(0, len(ys_out)-1, len(ys_out)), np.arange(len(ys_out)), ys_s if 'ys_s' in locals() else ys_pix),
                 '-', lw=1, label="Spline path")
    ax2.set_title("Digitized overlay")
    ax2.legend(loc="best")
    plt.tight_layout()
    fig2.savefig(overlay_path, dpi=200)
    plt.show(block=True)

    print(f"Saved overlay: {overlay_path}")
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_digitizer_loglog.py path/to/your_image.png")
        sys.exit(1)
    main(sys.argv[1])
