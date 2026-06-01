import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Function to draw blocks
def draw_block(x, y, text, color):
    box = FancyBboxPatch(
        (x, y), 2.5, 1.2,
        boxstyle="round,pad=0.3",
        linewidth=1.5,
        edgecolor='black',
        facecolor=color
    )
    ax.add_patch(box)
    ax.text(x + 1.25, y + 0.6, text,
            ha='center', va='center', fontsize=10, fontweight='bold')

# Function to draw arrows
def draw_arrow(x1, y1, x2, y2):
    arrow = Arrow(x1, y1, x2-x1, y2-y1, width=0.2)
    ax.add_patch(arrow)

# Blocks
draw_block(0.5, 2.5, "Raw GPR\nB-Scan", "#cce5ff")
draw_block(3.5, 2.5, "Missing Traces\n(Mask Applied)", "#ffcccc")
draw_block(6.5, 2.5, "U-Net++ Framework\n(Encoder-Decoder)", "#d5f5e3")
draw_block(6.5, 0.8, "SE Attention Block\n(Feature Recalibration)", "#f9e79f")
draw_block(10.0, 2.5, "Reconstructed\nGPR Data", "#d6eaf8")

# Arrows
draw_arrow(3.0, 3.1, 3.5, 3.1)
draw_arrow(6.0, 3.1, 6.5, 3.1)
draw_arrow(9.0, 3.1, 10.0, 3.1)
draw_arrow(7.75, 2.5, 7.75, 2.0)

# Labels
ax.text(6.5, 4.8, "Multi-scale Feature Learning", fontsize=11, ha='center')
ax.text(10.0, 4.8, "Improved Continuity & Noise Reduction", fontsize=11, ha='center')

# Title
plt.title("U-Net++ Based Framework for GPR Data Reconstruction", fontsize=14)

# Save high resolution
plt.savefig("gpr_unetpp_framework.png", dpi=300, bbox_inches='tight')
plt.show()