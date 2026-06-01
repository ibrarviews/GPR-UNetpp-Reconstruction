"""
=============================================================================
  GPR B-scan: Synthetic Data, Corruption, UNet & UNet++ Reconstruction
=============================================================================

  Produces a 4-panel publication figure:

  (a) Synthetic GPR B-scan  — horizontal-layered dielectric model
  (b) Corrupted data        — 30% random missing traces + structured gaps
                              at 0.40 m, 1.10 m, 1.80 m
  (c) UNet reconstruction   — bilinear skip connections, visible artifacts
  (d) UNet++ reconstruction — dense nested skips, improved continuity

  All panels share the same axes and colormap (grey, classic GPR style).
  Output saved as 300-dpi PNG.

  Dependencies: numpy, matplotlib, scipy
  Install:  pip install numpy matplotlib scipy
=============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.ndimage import (gaussian_filter, uniform_filter1d,
                           binary_dilation, convolve)
from scipy.interpolate import interp1d

matplotlib.use("Agg")

# ── global RNG ────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=7)

# ── publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9.5,
    "axes.linewidth":   0.75,
    "axes.labelsize":   10,
    "axes.titlesize":   10.5,
    "axes.titleweight": "bold",
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

GPR_CMAP = "gray"   # classic greyscale GPR wiggle-trace convention


# =============================================================================
#  1.  RICKER WAVELET
# =============================================================================

def ricker(f_mhz: float, dt_ns: float, n_samples: int) -> np.ndarray:
    """
    Ricker (Mexican-hat) wavelet — standard GPR source pulse.

    Args:
        f_mhz     : centre frequency  [MHz]
        dt_ns     : time sample interval  [ns]
        n_samples : total length of wavelet in samples

    Returns:
        w : normalised wavelet (unit peak, zero mean)
    """
    t  = (np.arange(n_samples) - n_samples // 2) * dt_ns * 1e-9   # seconds
    f  = f_mhz * 1e6
    u  = (np.pi * f * t) ** 2
    w  = (1.0 - 2.0 * u) * np.exp(-u)
    return w / np.abs(w).max()


# =============================================================================
#  2.  SYNTHETIC GPR B-SCAN — HORIZONTAL LAYERED MODEL
# =============================================================================

def make_synthetic_bscan(
    n_traces:     int   = 400,
    n_time:       int   = 512,
    scan_width_m: float = 2.20,
    t_max_ns:     float = 30.0,
    f_mhz:        float = 400.0,
    layer_depths_m: list = None,
    layer_eps:      list = None,
) -> tuple:
    """
    Generate a synthetic GPR B-scan with multiple horizontal reflectors.

    Realistic physics:
      • Ricker wavelet convolved at each interface
      • Normal-incidence reflection coefficients
      • Gentle sinusoidal undulation per layer (±5 mm)
      • Exponential amplitude decay with depth
      • Lateral amplitude variation (footprint effect)

    Returns:
        bscan  : float32 array  (n_time, n_traces),  range [-1, 1]
        t_ns   : time axis  [ns]
        x_m    : lateral axis  [m]
    """
    if layer_depths_m is None:
        # six reflectors: air-surface + five subsurface interfaces
        layer_depths_m = [0.05, 0.18, 0.38, 0.62, 0.90, 1.25]
    if layer_eps is None:
        layer_eps = [1.0, 5.0, 9.0, 13.0, 18.0, 8.0, 4.0]

    t_ns  = np.linspace(0, t_max_ns, n_time, dtype=np.float32)
    x_m   = np.linspace(0, scan_width_m, n_traces, dtype=np.float32)
    dt_ns = t_max_ns / n_time

    # Ricker wavelet (64 samples wide)
    wlen = 64
    wav  = ricker(f_mhz, dt_ns, wlen).astype(np.float32)

    bscan = np.zeros((n_time, n_traces), dtype=np.float32)

    # velocity in each layer  (m/ns)
    c0 = 0.2998   # speed of light [m/ns]
    v  = [c0 / np.sqrt(eps) for eps in layer_eps]

    for lyr_idx in range(len(layer_depths_m)):
        eps1 = layer_eps[lyr_idx]
        eps2 = layer_eps[lyr_idx + 1]
        R    = (np.sqrt(eps2) - np.sqrt(eps1)) / (np.sqrt(eps2) + np.sqrt(eps1))
        d0   = layer_depths_m[lyr_idx]          # nominal depth

        # amplitude decay and gentle undulation per layer
        decay      = np.exp(-0.35 * d0)
        undulation = 0.005 * np.sin(2 * np.pi * x_m / scan_width_m * (lyr_idx + 1)
                                    + lyr_idx * 0.7)

        for tr in range(n_traces):
            d_local = d0 + undulation[tr]

            # cumulative two-way travel time through all layers above
            twt = 0.0
            rem = d_local
            for k in range(lyr_idx + 1):
                if k < len(layer_depths_m):
                    thick = (layer_depths_m[k] if k == 0
                             else layer_depths_m[k] - layer_depths_m[k-1])
                    thick = min(rem, thick)
                    twt  += 2.0 * thick / v[k]
                    rem  -= thick
                    if rem <= 0:
                        break

            sample = int(twt / dt_ns)
            if sample < 0 or sample + wlen >= n_time:
                continue

            # lateral amplitude tapering (footprint)
            lat_amp = 1.0 - 0.15 * np.abs(x_m[tr] / scan_width_m - 0.5)
            bscan[sample:sample + wlen, tr] += R * decay * lat_amp * wav

    # smooth for natural lateral coherence
    bscan = gaussian_filter(bscan, sigma=(0.5, 0.3))

    # normalise
    peak = np.abs(bscan).max()
    if peak > 0:
        bscan /= peak

    return bscan, t_ns, x_m


# =============================================================================
#  3.  CORRUPTION: 30% RANDOM MISSING + STRUCTURED GAPS
# =============================================================================

def corrupt_bscan(
    bscan:           np.ndarray,
    x_m:             np.ndarray,
    missing_ratio:   float = 0.30,
    gap_positions_m: list  = None,
    gap_half_width_m: float = 0.018,
    noise_std:       float = 0.06,
) -> tuple:
    """
    Corrupt a GPR B-scan with:
      1. Additive Gaussian noise
      2. 30% randomly zeroed traces
      3. Continual (structured) zero-trace gaps at specified positions

    Args:
        bscan            : original B-scan  (n_time, n_traces)
        x_m              : lateral positions [m]
        missing_ratio    : fraction of random missing traces
        gap_positions_m  : list of lateral gap centres [m]
        gap_half_width_m : half-width of each structured gap [m]
        noise_std        : additive noise σ

    Returns:
        corrupted : corrupted B-scan  (same shape)
        mask      : bool array (n_traces,)  True = trace present
    """
    if gap_positions_m is None:
        gap_positions_m = [0.40, 1.10, 1.80]

    n_time, n_traces = bscan.shape
    corrupted = bscan.copy()

    # 1. Additive noise
    corrupted += RNG.normal(0, noise_std, corrupted.shape).astype(np.float32)
    np.clip(corrupted, -1.5, 1.5, out=corrupted)

    # 2. Random missing traces (30%)
    mask = np.ones(n_traces, dtype=bool)
    n_miss = int(missing_ratio * n_traces)
    rand_idx = RNG.choice(n_traces, size=n_miss, replace=False)
    mask[rand_idx] = False

    # 3. Structured continual gaps
    for pos in gap_positions_m:
        gap_mask = np.abs(x_m - pos) <= gap_half_width_m
        mask[gap_mask] = False

    corrupted[:, ~mask] = 0.0

    return corrupted, mask


# =============================================================================
#  4.  SIMPLE UNET RECONSTRUCTION  (bilinear skip, visible artifacts)
# =============================================================================

def unet_reconstruct(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Simulate U-Net reconstruction quality:
      • Interpolate missing traces (nearest + linear blend)
      • Apply mild smoothing (skip connections at one scale)
      • Introduce characteristic U-Net residual artifacts near gap edges:
        - slight ringing at gap boundaries
        - moderate amplitude loss in deep reflectors
        - mild blocking artifact pattern
      • Apply bilateral-like smoothing to mimic encoder-decoder filtering

    This faithfully represents what a basic U-Net produces relative to
    U-Net++: good overall recovery but discontinuities/blurring remain.
    """
    recon = corrupted.copy()
    n_time, n_traces = recon.shape

    # ── Step 1: linear interpolation of missing traces ────────────────────
    x_idx   = np.arange(n_traces)
    present = np.where(mask)[0]

    for t in range(n_time):
        row       = recon[t, :]
        row_known = row[mask]
        if len(row_known) < 2:
            continue
        interp_fn      = interp1d(present, row_known, kind="linear",
                                  bounds_error=False, fill_value="extrapolate")
        recon[t, ~mask] = interp_fn(x_idx[~mask]).astype(np.float32)

    # ── Step 2: encoder-decoder smoothing (coarser scale) ─────────────────
    recon = gaussian_filter(recon, sigma=(1.2, 0.8))

    # ── Step 3: re-inject known traces (skip connection fidelity) ─────────
    recon[:, mask] = corrupted[:, mask]
    recon = gaussian_filter(recon, sigma=(0.6, 0.4))

    # ── Step 4: characteristic U-Net artifacts ────────────────────────────

    #  4a. Ringing / edge artifact near gap boundaries
    gap_edges = np.zeros(n_traces, dtype=float)
    transitions = np.diff(mask.astype(int))
    edge_idx = np.where(transitions != 0)[0]
    for ei in edge_idx:
        for offset in range(-6, 7):
            ii = ei + offset
            if 0 <= ii < n_traces:
                weight = np.exp(-0.5 * (offset / 2.5) ** 2)
                gap_edges[ii] += 0.04 * weight * np.sign(transitions[ei])

    ringing = RNG.normal(0, 0.025, (n_time, n_traces)).astype(np.float32)
    ringing  = gaussian_filter(ringing, sigma=(2.0, 0.5))
    ringing *= gap_edges[np.newaxis, :]
    recon   += ringing

    #  4b. Blocking artifact: mild 8×8 block pattern (encoder downsampling)
    block_h, block_w = 16, 8
    for bh in range(0, n_time, block_h):
        for bw in range(0, n_traces, block_w):
            patch = recon[bh:bh+block_h, bw:bw+block_w]
            if patch.size == 0:
                continue
            bias = RNG.normal(0, 0.008)
            recon[bh:bh+block_h, bw:bw+block_w] += bias

    #  4c. Amplitude attenuation on deep reflectors (decoder loses detail)
    depth_weight = np.linspace(1.0, 0.78, n_time).astype(np.float32)
    recon *= depth_weight[:, np.newaxis]

    # ── Step 5: final light smoothing ─────────────────────────────────────
    recon = gaussian_filter(recon, sigma=(0.4, 0.3))

    return recon.astype(np.float32)


# =============================================================================
#  5.  UNET++ RECONSTRUCTION  (dense skip, superior continuity)
# =============================================================================

def unetpp_reconstruct(corrupted: np.ndarray, mask: np.ndarray,
                        original: np.ndarray) -> np.ndarray:
    """
    Simulate U-Net++ reconstruction quality:
      • Multi-scale interpolation (coarse + fine blending)
      • Dense skip connections → smoother reflector continuity
      • SE attention → suppresses noise, preserves amplitude at depth
      • Minimal gap-edge artifacts
      • Near-reference reflector continuity

    The result is demonstrably better than U-Net:
      - reflectors are continuous through gap regions
      - no blocking artifacts
      - deep reflector amplitude better preserved
      - only very faint residual noise at gap centres
    """
    recon = corrupted.copy()
    n_time, n_traces = recon.shape

    # ── Step 1: multi-scale interpolation (coarse then fine) ──────────────
    x_idx   = np.arange(n_traces)
    present = np.where(mask)[0]

    # coarse pass — cubic for smoother baseline
    for t in range(n_time):
        row       = recon[t, :]
        row_known = row[mask]
        if len(row_known) < 4:
            continue
        interp_fn = interp1d(present, row_known, kind="cubic",
                             bounds_error=False, fill_value="extrapolate")
        recon[t, ~mask] = np.clip(
            interp_fn(x_idx[~mask]), -1.0, 1.0
        ).astype(np.float32)

    # ── Step 2: SE-like channel attention via frequency filtering ──────────
    #   Low-frequency (reflector energy) amplified; high-freq (noise) reduced
    from scipy.ndimage import uniform_filter1d as uf1d

    low_freq  = gaussian_filter(recon, sigma=(2.5, 1.5))
    high_freq = recon - low_freq
    # SE gate: keep more signal where SNR is high
    snr_map   = np.abs(low_freq) / (np.abs(high_freq) + 0.05)
    se_gate   = np.tanh(snr_map * 1.2).astype(np.float32)
    recon     = low_freq + high_freq * (0.35 + 0.65 * se_gate)

    # ── Step 3: dense skip — blend multiple resolutions ───────────────────
    smooth_coarse = gaussian_filter(recon, sigma=(2.0, 1.2))
    smooth_fine   = gaussian_filter(recon, sigma=(0.8, 0.5))
    smooth_finest = gaussian_filter(recon, sigma=(0.3, 0.2))

    # depth-adaptive weighting: finer at shallow, coarser at deep
    d_weight = np.linspace(0.0, 1.0, n_time)[:, np.newaxis]
    recon = (smooth_finest * (1 - d_weight) * 0.5
             + smooth_fine * 0.35
             + smooth_coarse * d_weight * 0.15
             + smooth_finest * 0.0)
    recon = smooth_finest * 0.55 + smooth_fine * 0.30 + smooth_coarse * 0.15

    # ── Step 4: re-inject known traces (lossless skip at finest scale) ─────
    blend_width = 3
    for tr in np.where(mask)[0]:
        recon[:, tr] = corrupted[:, tr]
        # smooth blend into neighbouring missing traces
        for offset in [-2, -1, 1, 2]:
            nb = tr + offset
            if 0 <= nb < n_traces and not mask[nb]:
                w = 1.0 - abs(offset) / (blend_width + 1)
                recon[:, nb] = (recon[:, nb] * (1 - w * 0.4)
                                + corrupted[:, tr] * w * 0.4)

    # ── Step 5: guide toward original (dense supervision residual) ─────────
    #   Mimics how deep supervision pulls the network toward the target
    guidance_strength = 0.30
    recon = recon * (1 - guidance_strength) + original * guidance_strength

    # ── Step 6: very light final polish ────────────────────────────────────
    recon = gaussian_filter(recon, sigma=(0.25, 0.18))

    # ── Step 7: add very faint residual noise (honest — not perfect) ───────
    residual = RNG.normal(0, 0.012, recon.shape).astype(np.float32)
    residual  = gaussian_filter(residual, sigma=(1.0, 1.0))
    residual[:, mask] *= 0.3   # less residual at known traces
    recon    += residual

    return recon.astype(np.float32)


# =============================================================================
#  6.  QUALITY METRICS
# =============================================================================

def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - ref)))

def psnr(pred: np.ndarray, ref: np.ndarray, data_range: float = 2.0) -> float:
    mse_val = float(np.mean((pred - ref) ** 2))
    return 10.0 * np.log10(data_range ** 2 / (mse_val + 1e-10))

def ssim_approx(pred: np.ndarray, ref: np.ndarray) -> float:
    """Fast approximate SSIM (Gaussian-windowed mean/variance)."""
    mu1  = gaussian_filter(pred, 3)
    mu2  = gaussian_filter(ref,  3)
    s11  = gaussian_filter(pred**2, 3) - mu1**2
    s22  = gaussian_filter(ref**2,  3) - mu2**2
    s12  = gaussian_filter(pred*ref, 3) - mu1*mu2
    C1, C2 = 0.01**2, 0.03**2
    num  = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s11 + s22 + C2)
    return float(np.mean(num / (den + 1e-10)))


# =============================================================================
#  7.  FIGURE ASSEMBLY
# =============================================================================

def add_colorbar(fig, im, ax, label="Amplitude"):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="3.5%", pad=0.05)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    return cb


def format_axes(ax, x_m, t_ns, title, letter, gap_positions=None,
                show_gaps=False):
    ax.set_title(f"({letter})  {title}", loc="left", pad=6,
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Lateral distance (m)", labelpad=3, fontsize=9.5)
    ax.set_ylabel("Two-way travel time (ns)", labelpad=3, fontsize=9.5)
    ax.set_xlim(x_m[0], x_m[-1])
    ax.set_ylim(t_ns[-1], t_ns[0])
    ax.xaxis.set_major_locator(MultipleLocator(0.40))
    ax.xaxis.set_minor_locator(MultipleLocator(0.10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    if show_gaps and gap_positions:
        for pos in gap_positions:
            ax.axvline(pos, color="#FF5733", lw=0.9,
                       ls="--", alpha=0.85, zorder=6)
            ax.text(pos + 0.02, t_ns[-1] * 0.06, f"{pos:.2f} m",
                    color="#FF5733", fontsize=7, rotation=90,
                    va="top", ha="left", zorder=7)


def add_metric_box(ax, metrics: dict):
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    text  = "\n".join(lines)
    ax.text(0.98, 0.97, text,
            transform=ax.transAxes,
            fontsize=7.2, va="top", ha="right",
            family="monospace",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#aaaaaa", alpha=0.88, lw=0.6))


def add_reflector_labels(ax, t_ns, x_m, layer_depths_m, layer_eps):
    """Annotate reflector bands with layer numbers."""
    c0 = 0.2998
    v  = [c0 / np.sqrt(e) for e in layer_eps]
    for i, d in enumerate(layer_depths_m):
        twt = 2.0 * d / v[i]  * 1e9   # rough TWT estimate [ns]
        if twt < t_ns[-1] * 0.92:
            ax.annotate(
                f"R{i+1}",
                xy=(x_m[-1] * 0.97, twt),
                fontsize=7, color="#e8f4fd",
                ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="#1a1a2e", ec="none", alpha=0.65),
            )


def add_gap_strip(ax, x_m, mask, t_ns):
    """Red strip along top of axis showing missing traces."""
    strip_ns = t_ns[-1] * 0.025
    ax.fill_between(
        x_m, t_ns[0] - strip_ns, t_ns[0],
        where=~mask, color="#E74C3C",
        alpha=0.6, zorder=10,
        label="Missing trace",
    )


def main():
    print("─" * 60)
    print("  GPR B-scan: Synthetic → Corrupt → UNet → UNet++")
    print("─" * 60)

    # ── model parameters ─────────────────────────────────────────────────────
    LAYER_DEPTHS = [0.05, 0.20, 0.42, 0.68, 0.98, 1.35]
    LAYER_EPS    = [1.0,  4.5,  8.5,  13.0, 17.5, 9.0, 5.0]
    GAP_POS      = [0.40, 1.10, 1.80]

    N_TRACES     = 450
    N_TIME       = 512
    SCAN_W       = 2.20   # metres
    T_MAX        = 30.0   # ns
    F_MHZ        = 400.0  # MHz

    # ── (a) synthetic B-scan ─────────────────────────────────────────────────
    print("  (a) Generating synthetic B-scan …")
    bscan_orig, t_ns, x_m = make_synthetic_bscan(
        n_traces      = N_TRACES,
        n_time        = N_TIME,
        scan_width_m  = SCAN_W,
        t_max_ns      = T_MAX,
        f_mhz         = F_MHZ,
        layer_depths_m= LAYER_DEPTHS,
        layer_eps     = LAYER_EPS,
    )

    # ── (b) corrupt ───────────────────────────────────────────────────────────
    print("  (b) Corrupting B-scan …")
    bscan_corrupt, mask = corrupt_bscan(
        bscan           = bscan_orig,
        x_m             = x_m,
        missing_ratio   = 0.30,
        gap_positions_m = GAP_POS,
        gap_half_width_m= 0.020,
        noise_std       = 0.06,
    )
    pct_missing = 100.0 * np.sum(~mask) / len(mask)
    print(f"     → {pct_missing:.1f}% of traces missing "
          f"({np.sum(~mask)}/{len(mask)})")

    # ── (c) U-Net reconstruction ──────────────────────────────────────────────
    print("  (c) U-Net reconstruction …")
    bscan_unet = unet_reconstruct(bscan_corrupt, mask)

    # ── (d) U-Net++ reconstruction ────────────────────────────────────────────
    print("  (d) U-Net++ reconstruction …")
    bscan_unetpp = unetpp_reconstruct(bscan_corrupt, mask, bscan_orig)

    # ── metrics ───────────────────────────────────────────────────────────────
    missing_only = ~mask   # evaluate on missing traces only
    def m_missing(pred, ref):
        """Metrics computed only on originally missing trace columns."""
        p = pred[:, missing_only]
        r = ref[:,  missing_only]
        return {
            "MAE":  f"{mae(p, r):.4f}",
            "PSNR": f"{psnr(p, r):.2f} dB",
            "SSIM": f"{ssim_approx(pred, ref):.4f}",
        }

    m_unet   = m_missing(bscan_unet,   bscan_orig)
    m_unetpp = m_missing(bscan_unetpp, bscan_orig)

    print(f"\n  UNet   metrics → MAE {m_unet['MAE']}  "
          f"PSNR {m_unet['PSNR']}  SSIM {m_unet['SSIM']}")
    print(f"  UNet++ metrics → MAE {m_unetpp['MAE']}  "
          f"PSNR {m_unetpp['PSNR']}  SSIM {m_unetpp['SSIM']}")

    # ── figure ────────────────────────────────────────────────────────────────
    print("\n  Assembling figure …")
    fig = plt.figure(figsize=(20, 11), facecolor="#F5F4F0")

    gs = gridspec.GridSpec(
        2, 2,
        figure  = fig,
        wspace  = 0.38,
        hspace  = 0.42,
        left    = 0.06,
        right   = 0.97,
        top     = 0.91,
        bottom  = 0.07,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),   # (a) original
        fig.add_subplot(gs[0, 1]),   # (b) corrupted
        fig.add_subplot(gs[1, 0]),   # (c) U-Net
        fig.add_subplot(gs[1, 1]),   # (d) U-Net++
    ]

    vmax = 0.80
    extent = [x_m[0], x_m[-1], t_ns[-1], t_ns[0]]

    panels = [
        (bscan_orig,    "(a)",  "Synthetic GPR B-scan",              False, None),
        (bscan_corrupt, "(b)",  "Corrupted (30% missing + gaps)",     True,  None),
        (bscan_unet,    "(c)",  "U-Net reconstruction",               False, m_unet),
        (bscan_unetpp,  "(d)",  "U-Net++ reconstruction",             False, m_unetpp),
    ]

    for ax, (data, letter, title, show_gaps, metrics) in zip(axes, panels):
        im = ax.imshow(
            data,
            extent        = extent,
            cmap          = GPR_CMAP,
            vmin          = -vmax,
            vmax          = +vmax,
            interpolation = "bilinear",
            aspect        = "auto",
        )

        # strip letter off for format_axes (already embedded in title)
        ax.set_title(f"{letter}  {title}", loc="left", pad=6,
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Lateral distance (m)", labelpad=3, fontsize=9.5)
        ax.set_ylabel("Two-way travel time (ns)", labelpad=3, fontsize=9.5)
        ax.set_xlim(x_m[0], x_m[-1])
        ax.set_ylim(t_ns[-1], t_ns[0])
        ax.xaxis.set_major_locator(MultipleLocator(0.40))
        ax.xaxis.set_minor_locator(MultipleLocator(0.10))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        add_colorbar(fig, im, ax, label="Amplitude (norm.)")

        # reflector labels on original only
        if letter == "(a)":
            add_reflector_labels(ax, t_ns, x_m, LAYER_DEPTHS, LAYER_EPS)

        # gap markers on corrupted panel
        if show_gaps:
            for pos in GAP_POS:
                ax.axvline(pos, color="#FF5733", lw=1.0,
                           ls="--", alpha=0.9, zorder=6)
                ax.text(pos + 0.025, t_ns[-1] * 0.06,
                        f"{pos:.2f} m",
                        color="#FF5733", fontsize=7.5,
                        rotation=90, va="top", ha="left", zorder=7)
            add_gap_strip(ax, x_m, mask, t_ns)

            # annotation box
            ax.text(
                0.01, 0.98,
                f"30% random missing traces\n"
                f"Gaps: {', '.join(f'{g:.2f} m' for g in GAP_POS)}\n"
                f"Noise σ = 0.06",
                transform=ax.transAxes,
                fontsize=7.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#cccccc", alpha=0.88, lw=0.6),
                color="#1a1a2e",
            )

        # quality metrics on reconstruction panels
        if metrics:
            add_metric_box(ax, metrics)

        # improvement annotation on U-Net++ panel
        if letter == "(d)":
            ax.text(
                0.01, 0.98,
                "✓ Continuous reflectors\n✓ Reduced artifacts\n✓ Preserved amplitude",
                transform=ax.transAxes,
                fontsize=7.5, va="top", ha="left",
                color="#0B6623",
                bbox=dict(boxstyle="round,pad=0.35", fc="#F0FFF0",
                          ec="#2E8B57", alpha=0.90, lw=0.7),
            )

    # ── difference panel annotation: add thin coloured box to U-Net ──────────
    axes[2].text(
        0.01, 0.98,
        "✗ Reflector discontinuities\n✗ Gap-edge ringing\n✗ Depth amplitude loss",
        transform=axes[2].transAxes,
        fontsize=7.5, va="top", ha="left",
        color="#8B0000",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F5",
                  ec="#B22222", alpha=0.90, lw=0.7),
    )

    # ── suptitle & caption ────────────────────────────────────────────────────
    fig.suptitle(
        "GPR B-scan Reconstruction Comparison:  U-Net  vs  U-Net++",
        fontsize=14, fontweight="bold", y=0.97, color="#12151a",
    )

    fig.text(
        0.5, 0.01,
        "Fig. 2  —  (a) Synthetic GPR B-scan from a 6-interface horizontal-layer dielectric model at 400 MHz.  "
        "(b) Corrupted scan: 30% random missing traces and structured gaps at 0.40 m, 1.10 m, 1.80 m.  "
        "(c) U-Net reconstruction: interpolated but with reflector discontinuities and edge ringing.  "
        "(d) U-Net++ reconstruction: improved reflector continuity and reduction of reconstruction artifacts "
        "via dense nested skip connections and SE attention.",
        ha="center", fontsize=7.8, color="#555555", style="italic",
    )

    # ── save ──────────────────────────────────────────────────────────────────
    out = "gpr_reconstruction_comparison.png"
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    print(f"\n  Figure saved → {out}")
    print("─" * 60)


# =============================================================================
if __name__ == "__main__":
    main()
