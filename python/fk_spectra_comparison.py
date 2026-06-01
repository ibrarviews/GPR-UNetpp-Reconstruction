import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Generate synthetic GPR data
# -----------------------------
nx, nt = 128, 128
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)

X, T = np.meshgrid(x, t)

# Simulated layered reflections
original = np.sin(20 * T + 10 * X) + 0.5 * np.sin(40 * T)

# -----------------------------
# Simulate reconstructions
# -----------------------------
# POCS (more noise & distortion)
pocs = original + 0.5 * np.random.randn(nt, nx)

# U-Net (moderate smoothing)
unet = original + 0.2 * np.random.randn(nt, nx)

# U-Net++ (best, least noise)
unetpp = original + 0.05 * np.random.randn(nt, nx)

# -----------------------------
# Function to compute F-K spectrum
# -----------------------------
def fk_spectrum(data):
    fk = np.fft.fft2(data)
    fk_shifted = np.fft.fftshift(fk)
    magnitude = np.log(np.abs(fk_shifted) + 1)
    return magnitude

fk_original = fk_spectrum(original)
fk_pocs = fk_spectrum(pocs)
fk_unet = fk_spectrum(unet)
fk_unetpp = fk_spectrum(unetpp)

# -----------------------------
# Plotting
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0,0].imshow(fk_original, aspect='auto')
axs[0,0].set_title('(a) Original')

axs[0,1].imshow(fk_pocs, aspect='auto')
axs[0,1].set_title('(b) POCS')

axs[1,0].imshow(fk_unet, aspect='auto')
axs[1,0].set_title('(c) U-Net')

axs[1,1].imshow(fk_unetpp, aspect='auto')
axs[1,1].set_title('(d) U-Net++')

# Remove axis ticks for clean look
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

# Save high-quality image
plt.savefig("fk_spectra_comparison.png", dpi=300, bbox_inches='tight')

plt.show()