import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulated time axis (ns)
# -----------------------------
time = np.linspace(0, 80, 400)  # 0 to 80 ns

# -----------------------------
# Original waveform
# -----------------------------
original = np.sin(0.2 * time) * np.exp(-0.02 * time)

# -----------------------------
# Simulated reconstructions
# -----------------------------
# POCS (noisy, distorted)
pocs = original + 0.25 * np.random.randn(len(time))

# U-Net (moderate improvement)
unet = original + 0.12 * np.random.randn(len(time))

# U-Net++ (best match)
unetpp = original + 0.05 * np.random.randn(len(time))

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(time, original, linewidth=2, label='Original')
plt.plot(time, pocs, linestyle='--', label='POCS')
plt.plot(time, unet, linestyle='-.', label='U-Net')
plt.plot(time, unetpp, linewidth=2, label='U-Net++')

# -----------------------------
# Highlight region (28–38 ns)
# -----------------------------
plt.axvspan(28, 38, alpha=0.2)

# -----------------------------
# Labels & Title
# -----------------------------
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title('Single-Trace Waveform Comparison at 35 cm Lateral Position')
plt.legend()
plt.grid(True)

# -----------------------------
# Save high resolution
# -----------------------------
plt.savefig("waveform_comparison.png", dpi=300, bbox_inches='tight')

plt.show()