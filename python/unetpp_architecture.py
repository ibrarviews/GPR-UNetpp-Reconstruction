import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Create canvas
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Function to draw blocks
def draw_block(x, y, text):
    box = FancyBboxPatch(
        (x, y), 1.6, 0.9,
        boxstyle="round,pad=0.2",
        linewidth=1.5,
        edgecolor='black',
        facecolor='#d6eaf8'
    )
    ax.add_patch(box)
    ax.text(x + 0.8, y + 0.45, text,
            ha='center', va='center', fontsize=8)

# Function to draw arrows
def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        linewidth=1.2
    )
    ax.add_patch(arrow)

# ---------------------------
# Encoder (Downsampling)
# ---------------------------
encoder_y = [8, 6.5, 5, 3.5]
for i, y in enumerate(encoder_y):
    draw_block(1, y, f"Enc {i+1}\nConv+BN+ReLU")

# ---------------------------
# Decoder (Upsampling)
# ---------------------------
decoder_y = [3.5, 5, 6.5, 8]
for i, y in enumerate(decoder_y):
    draw_block(9, y, f"Dec {i+1}\nUp+Conv")

# ---------------------------
# Bottleneck
# ---------------------------
draw_block(5, 2.5, "Bottleneck\nMulti-scale")

# ---------------------------
# Dense Skip Connections (U-Net++)
# ---------------------------
for i in range(4):
    draw_block(4 + i, 8 - i*1.5, "Skip\nFusion")

# ---------------------------
# Input & Output
# ---------------------------
draw_block(0.5, 8, "Input GPR\n(B-scan)")
draw_block(11, 8, "Reconstructed\nGPR")

# ---------------------------
# Arrows: Encoder Flow
# ---------------------------
for i in range(3):
    draw_arrow(1.8, encoder_y[i], 1.8, encoder_y[i+1] + 0.9)

# ---------------------------
# Arrows: Encoder → Bottleneck
# ---------------------------
draw_arrow(2.5, 3.5, 5, 3)

# ---------------------------
# Bottleneck → Decoder
# ---------------------------
draw_arrow(6.6, 3, 9, 3.5)

# ---------------------------
# Decoder Flow
# ---------------------------
for i in range(3):
    draw_arrow(9.8, decoder_y[i], 9.8, decoder_y[i+1] + 0.9)

# ---------------------------
# Skip Connections (horizontal)
# ---------------------------
for i in range(4):
    draw_arrow(2.5, encoder_y[i], 4 + i, 8 - i*1.5)
    draw_arrow(4 + i + 1.2, 8 - i*1.5, 9, decoder_y[i])

# ---------------------------
# Input → Encoder
# ---------------------------
draw_arrow(1, 8.4, 1, 8.9)

# ---------------------------
# Decoder → Output
# ---------------------------
draw_arrow(10.6, 8.4, 11, 8.4)

# ---------------------------
# Titles
# ---------------------------
ax.text(6, 9.5, "U-Net++ Architecture for GPR Data Reconstruction",
        fontsize=14, ha='center', fontweight='bold')

ax.text(6, 9.0, "Multi-scale Feature Fusion with Dense Skip Connections",
        fontsize=10, ha='center')

# ---------------------------
# Save Image
# ---------------------------
plt.savefig("unetpp_architecture.png", dpi=300, bbox_inches='tight')
plt.show()