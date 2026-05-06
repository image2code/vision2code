import matplotlib.pyplot as plt
import numpy as np

# Figure setup
fig = plt.figure(figsize=(4.2, 7.2), facecolor="white")
ax = plt.axes([0.06, 0.08, 0.88, 0.84])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Colors
gold = "#b08a1f"
gold2 = "#d9a61a"
blue = "#1f4e79"
gray = "#666666"
lightgray = "#b7b7b7"
black = "#111111"

# Title and subtitle
ax.text(
    0, 99,
    "Young people, minorities, less likely to say Muslims\n"
    "should receive greater scrutiny because of their faith",
    ha="left", va="top", fontsize=12, fontweight="bold", color=black, family="DejaVu Sans"
)
ax.text(
    0, 90.5,
    "As part of the federal government's efforts to prevent terrorism, should\n"
    "Muslims living in the U.S. ... (%)",
    ha="left", va="top", fontsize=9.5, color=blue, style="italic", family="DejaVu Serif"
)

# Column headers
ax.text(43, 82.5, "Not be subject to\nadditional scrutiny solely\nbecause of religion",
        ha="center", va="top", fontsize=8.5, color=gold2, fontweight="bold")
ax.text(78, 82.5, "Be subject to more\nscrutiny than people of\nother religious groups",
        ha="center", va="top", fontsize=8.5, color=gold2, fontweight="bold")

# Data
groups = [
    ("Total", 61, 32),
    ("White", 57, 36),
    ("Black", 74, 17),
    ("Hispanic", 66, 25),
    ("18-29", 80, 17),
    ("30-49", 63, 30),
    ("50-64", 50, 40),
    ("65+", 50, 41),
    ("Postgrad", 69, 28),
    ("College grad", 65, 28),
    ("Some coll", 59, 33),
    ("HS or less", 58, 34),
    ("Republican", 44, 49),
    ("Independent", 62, 31),
    ("Democrat", 76, 20),
    ("White evang Prot", 43, 50),
    ("White mainline Prot", 56, 36),
    ("Black Prot", 71, 20),
    ("Catholic", 55, 38),
    ("Unaffiliated", 72, 24),
]

# Layout parameters
y0 = 76
dy = 3.35
bar_h = 1.9
x_center = 58
left_max = 80
right_max = 50
left_scale = 28 / left_max
right_scale = 28 / right_max

# Center divider
ax.plot([x_center, x_center], [6, 79], color="#8a8a8a", lw=0.8)

# Bars and labels
for i, (label, left_val, right_val) in enumerate(groups):
    y = y0 - i * dy

    # Left label
    ax.text(12, y, label, ha="left", va="center", fontsize=8.5, color=black)

    # Left bar
    left_w = left_val * left_scale
    ax.add_patch(plt.Rectangle((x_center - left_w, y - bar_h/2), left_w, bar_h,
                               facecolor=gold, edgecolor="none"))
    # Right bar
    right_w = right_val * right_scale
    ax.add_patch(plt.Rectangle((x_center, y - bar_h/2), right_w, bar_h,
                               facecolor=gold2, edgecolor="none"))

    # Values
    ax.text(x_center - left_w + 1.5, y, f"{left_val}", ha="left", va="center",
            fontsize=9, color="white")
    ax.text(x_center + right_w - 1.5, y, f"{right_val}", ha="right", va="center",
            fontsize=9, color=black)

# Footnotes
ax.text(0, 4.8, "Source: Survey conducted Dec. 8-13, 2015.", ha="left", va="bottom",
        fontsize=7.5, color="#7a7a7a")
ax.text(0, 3.2, "Whites and blacks include only those who are not Hispanic; Hispanics are of any race.",
        ha="left", va="bottom", fontsize=7.5, color="#7a7a7a")
ax.text(0, 1.6, "Don't know responses not shown.", ha="left", va="bottom",
        fontsize=7.5, color="#7a7a7a")
ax.text(0, 0.2, "PEW RESEARCH CENTER", ha="left", va="bottom",
        fontsize=8, color=black, fontweight="bold")

# Save
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
