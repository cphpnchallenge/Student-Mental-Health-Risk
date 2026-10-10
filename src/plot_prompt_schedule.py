#!/usr/bin/env python3
"""
Publication-quality version of the report's opening figure: lookback windows
relative to the daily survey prompt (24 h, 4 h, prior night; 1 h dropped).

Output
  figures/prompt_schedule.pdf   vector, fonts embedded (TrueType)
  figures/prompt_schedule.svg   vector, text kept as text
  figures/prompt_schedule.png   600 dpi raster
  figures/prompt_schedule.tiff  600 dpi, LZW (for journals that require TIFF)

Sized for a double-column figure (178 mm wide); font sizes are the sizes that
will appear in print, so do not rescale the figure when placing it.

Prompt timing and sleep-onset timing are computed from ssaqs_analysis_table.csv.
The prior-night bar is placed at the median sleep onset and spans the 10-hour
window the table uses.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = "/Users/ruizhang/work/student-mental-health/clean_data"
FIG = "/Users/ruizhang/work/student-mental-health/figures"
NIGHT_WINDOW_H = 10          # build_table.py: onset + 10 h, truncated at prompt

MM = 1 / 25.4
WIDTH, HEIGHT = 178 * MM, 56 * MM

INK = "#222222"; SUB = "#555555"; RULE = "#bdbdbd"
ACT = "#2c6fbb"              # activity / steps windows
PHYS = "#1a9e8f"             # prior-night physiology
PROMPT = "#c0392b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 3,
    "lines.linewidth": 0.9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType, editable text
    "svg.fonttype": "none",
    "savefig.dpi": 600,
})

# ------------------------------------------------------------------ data
df = pd.read_csv(f"{OUT}/ssaqs_analysis_table.csv")
h = df.local_hour
pct = 100 * h.between(13, 18).mean()
med = h.median()
med_txt = f"{int(med):02d}:{int(round(med % 1 * 60)):02d}"

on = df.hours_since_sleep_onset.dropna()
on_med, on_q1, on_q3 = on.median(), on.quantile(.25), on.quantile(.75)
night0 = -on_med
night1 = min(0.0, night0 + NIGHT_WINDOW_H)

print(f"prompts: {pct:.1f}% between 13:00-18:00, median {med_txt}, n={len(h)}")
print(f"sleep onset before prompt: median {on_med:.1f} h, "
      f"IQR {on_q1:.1f}-{on_q3:.1f} h, n={len(on)}")

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
fig.subplots_adjust(left=.13, right=.985, top=.98, bottom=.22)

H = .56   # bar height in row units
rows = [
    # y, t0, t1, colour, fill alpha, label, streams
    (3, -24, 0, ACT, .18, "24 h window", "Steps, activity levels"),
    (2, -4, 0, ACT, .32, "4 h window", "Steps, activity levels"),
    (1, night0, night1, PHYS, .30, "Prior night",
     r"HRV, SpO$_2$, sleep score"),
]
for y, t0, t1, col, a, name, streams in rows:
    ax.add_patch(Rectangle((t0, y - H / 2), t1 - t0, H, facecolor=col,
                           alpha=a, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((t0, y - H / 2), t1 - t0, H, facecolor="none",
                           edgecolor=col, lw=.8, zorder=3))
    ax.text(-25.2, y, name, ha="right", va="center", fontsize=8,
            fontweight="bold", color=INK)
    if t1 - t0 > 8:
        ax.text(t0 + .45, y, streams, ha="left", va="center", fontsize=7.5,
                color=SUB, zorder=4)
    else:
        ax.text(t0 - .45, y, streams, ha="right", va="center", fontsize=7.5,
                color=SUB, zorder=4)

# prompt marker -- label sits ABOVE the plotting area, clear of every bar
ax.axvline(0, ymin=0, ymax=1, color=PROMPT, lw=1.0, ls=(0, (3, 2)), zorder=5)
ax.annotate(f"Prompt sent\n{pct:.1f}% between 13:00 and 18:00 local, "
            f"median {med_txt}",
            xy=(0, 3.78), xycoords="data",
            xytext=(-.5, 3.78), textcoords="data",
            ha="right", va="center", fontsize=7.5, color=PROMPT,
            linespacing=1.25, annotation_clip=False)

ax.set_xlim(-25, .9)
ax.set_ylim(.45, 4.12)
ax.set_xticks([-24, -20, -16, -12, -8, -4, 0])
ax.set_xticklabels(["−24 h", "−20 h", "−16 h", "−12 h",
                    "−8 h", "−4 h", "0 h"])
ax.set_xlabel("Time relative to the survey prompt", labelpad=4)
ax.set_yticks([])
ax.grid(True, axis="x", color=RULE, lw=.4, ls=(0, (1, 2)))
ax.set_axisbelow(True)


for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{FIG}/prompt_schedule.{ext}")
fig.savefig(f"{FIG}/prompt_schedule.tiff", pil_kwargs={"compression": "tiff_lzw"})
print("wrote prompt_schedule.{pdf,svg,png,tiff}")
