import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# (a) Generate synthetic loss curves
# -----------------------------
epochs = np.arange(1, 101)

# Smooth decreasing curves
train_loss = np.exp(-epochs / 30) + 0.02 * np.random.rand(100)
val_loss = np.exp(-epochs / 28) + 0.03 * np.random.rand(100)

# -----------------------------
# (b) Quantitative metrics
# -----------------------------
methods = ['POCS', 'U-Net', 'U-Net++']

MAE = [0.045, 0.032, 0.025]
SNR = [18.3, 21.3, 23.1]
PSNR = [25.1, 28.5, 30.2]

x = np.arange(len(methods))
width = 0.25

# -----------------------------
# Create figure
# -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# -------- (a) Loss curves --------
axs[0].plot(epochs, train_loss, linewidth=2, label='Training Loss')
axs[0].plot(epochs, val_loss, linewidth=2, linestyle='--', label='Validation Loss')

axs[0].set_title('(a) Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid(True)

# -------- (b) Metrics comparison --------
axs[1].bar(x - width, MAE, width, label='MAE')
axs[1].bar(x, SNR, width, label='SNR')
axs[1].bar(x + width, PSNR, width, label='PSNR')

axs[1].set_title('(b) Quantitative Comparison')
axs[1].set_xticks(x)
axs[1].set_xticklabels(methods)
axs[1].set_ylabel('Metric Value')
axs[1].legend()
axs[1].grid(True)

# -----------------------------
# Layout & Save
# -----------------------------
plt.tight_layout()
plt.savefig("training_metrics_comparison.png", dpi=300)
plt.show()