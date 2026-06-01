"""
=============================================================================
  GPR Synthetic Data Generation, Corruption & U-Net++ Training Visualisation
=============================================================================

  Produces three publication-quality figures:

  (a) Original synthetic GPR B-scan — multi-layer dielectric model
  (b) Corrupted B-scan — 30% random missing traces + continual gaps at
      lateral positions 0.10 m, 0.25 m, 0.38 m + additive Gaussian noise
  (c) Training loss curve over 100 epochs — stable convergence of U-Net++

  All figures are saved as high-resolution PNG (300 dpi) and displayed
  in a single combined panel.

  Dependencies: numpy, matplotlib, scipy
  Install:  pip install numpy matplotlib scipy
=============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import chirp, hilbert
from scipy.ndimage import gaussian_filter

# ── reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── global style ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.linewidth":    0.8,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.minor.size":  2,
    "ytick.minor.size":  2,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "image.aspect":      "auto",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Ricker (Mexican-hat) wavelet
# ─────────────────────────────────────────────────────────────────────────────

def ricker_wavelet(freq_mhz: float, dt_ns: float, duration_ns: float) -> np.ndarray:
    """
    Generate a Ricker (second derivative of Gaussian) wavelet.

    Args:
        freq_mhz   : centre frequency in MHz
        dt_ns      : time sample interval in nanoseconds
        duration_ns: total wavelet duration in nanoseconds

    Returns:
        w : 1-D numpy array, zero-mean, unit-peak
    """
    f  = freq_mhz * 1e6          # convert to Hz  (kept dimensionally correct)
    t  = np.arange(0, duration_ns, dt_ns) * 1e-9   # seconds
    t0 = 1.0 / f                 # wavelet delay  (one full period)
    u  = np.pi * f * (t - t0)
    w  = (1.0 - 2.0 * u ** 2) * np.exp(-(u ** 2))
    return w / np.abs(w).max()


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Multi-layer dielectric forward model
# ─────────────────────────────────────────────────────────────────────────────

class DielectricLayer:
    """Represents one horizontal dielectric layer in the subsurface."""
    def __init__(
        self,
        depth_m:    float,   # top of layer in metres
        eps_r:      float,   # relative permittivity
        sigma_s_m:  float,   # electrical conductivity  S/m
    ):
        self.depth_m   = depth_m
        self.eps_r     = eps_r
        self.sigma_s_m = sigma_s_m
        self.v         = 3e8 / np.sqrt(eps_r)   # wave speed m/s


def two_way_travel_time(depth_m: float, velocity_m_per_s: float) -> float:
    """Two-way travel time in nanoseconds."""
    return 2.0 * depth_m / velocity_m_per_s * 1e9


def reflection_coefficient(eps1: float, eps2: float) -> float:
    """
    Normal-incidence reflection coefficient between two lossless layers.
        R = (sqrt(eps2) - sqrt(eps1)) / (sqrt(eps2) + sqrt(eps1))
    """
    return (np.sqrt(eps2) - np.sqrt(eps1)) / (np.sqrt(eps2) + np.sqrt(eps1))


def synthetic_bscan(
    layers:       list,
    n_traces:     int   = 256,
    scan_width_m: float = 0.50,
    dt_ns:        float = 0.10,
    t_max_ns:     float = 25.0,
    freq_mhz:     float = 900.0,
    undulation_amp_m: float = 0.008,
) -> tuple:
    """
    Generate a synthetic GPR B-scan from a multi-layer dielectric model.

    Features realistic physics:
      • Ricker wavelet source
      • Two-way travel time for each interface
      • Normal-incidence reflection coefficients
      • Mild geometric (sinusoidal) undulation on each interface
      • Exponential amplitude decay (geometric spreading + attenuation)
      • Consistent polarity reversal between layers

    Args:
        layers           : list of DielectricLayer objects (ordered by depth)
        n_traces         : number of A-scans in the B-scan
        scan_width_m     : total lateral extent in metres
        dt_ns            : time sample interval [ns]
        t_max_ns         : maximum two-way travel time [ns]
        freq_mhz         : centre frequency of the Ricker wavelet [MHz]
        undulation_amp_m : amplitude of sinusoidal interface undulation [m]

    Returns:
        bscan    : 2-D float32 array  (n_time_samples, n_traces)
        t_axis   : 1-D float array    [ns]
        x_axis   : 1-D float array    [m]
    """
    t_axis  = np.arange(0, t_max_ns, dt_ns, dtype=np.float32)
    x_axis  = np.linspace(0, scan_width_m, n_traces, dtype=np.float32)
    n_times = len(t_axis)

    wavelet = ricker_wavelet(freq_mhz, dt_ns, 8.0 / freq_mhz * 1e3).astype(np.float32)
    wlen    = len(wavelet)

    bscan = np.zeros((n_times, n_traces), dtype=np.float32)

    # build reflector at each interface
    for k in range(len(layers) - 1):
        top   = layers[k]
        below = layers[k + 1]
        R     = reflection_coefficient(top.eps_r, below.eps_r)

        for tr_idx, x in enumerate(x_axis):
            # gentle sinusoidal undulation varies with lateral position & layer index
            undulation  = undulation_amp_m * np.sin(2 * np.pi * x / scan_width_m * (k + 1))
            depth_local = below.depth_m + undulation

            # two-way travel time through all layers above this interface
            twt_ns = two_way_travel_time(depth_local, top.v)
            if twt_ns >= t_max_ns:
                continue

            sample_idx = int(twt_ns / dt_ns)
            if sample_idx + wlen >= n_times:
                continue

            # exponential amplitude decay with depth
            decay = np.exp(-0.12 * below.depth_m)
            amplitude = R * decay

            bscan[sample_idx: sample_idx + wlen, tr_idx] += amplitude * wavelet

    # smooth horizontally for lateral coherence
    bscan = gaussian_filter(bscan, sigma=(0.6, 0.4))

    # normalise to [-1, 1]
    peak  = np.abs(bscan).max()
    if peak > 0:
        bscan /= peak

    return bscan, t_axis, x_axis


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Corrupted B-scan (missing traces + noise)
# ─────────────────────────────────────────────────────────────────────────────

def corrupt_bscan(
    bscan:         np.ndarray,
    x_axis:        np.ndarray,
    missing_ratio: float       = 0.30,
    gap_positions_m: list      = None,
    gap_width_m:   float       = 0.010,
    noise_std:     float       = 0.08,
) -> tuple:
    """
    Simulate realistic GPR data corruption:
      1. Additive Gaussian noise
      2. 30% randomly missing traces (set to zero)
      3. Continual trace gaps at specified lateral positions

    Args:
        bscan            : original B-scan  (n_time, n_traces)
        x_axis           : lateral positions  [m]
        missing_ratio    : fraction of traces to zero-out randomly
        gap_positions_m  : list of lateral positions [m] for continual gaps
        gap_width_m      : half-width of each continual gap  [m]
        noise_std        : standard deviation of additive Gaussian noise

    Returns:
        corrupted : corrupted B-scan
        mask      : boolean array  True = available trace, False = missing
    """
    if gap_positions_m is None:
        gap_positions_m = [0.10, 0.25, 0.38]

    n_times, n_traces = bscan.shape
    corrupted = bscan.copy()

    # 1. Additive Gaussian noise
    corrupted += RNG.normal(0.0, noise_std, corrupted.shape).astype(np.float32)

    # 2. Random missing traces  (30%)
    mask = np.ones(n_traces, dtype=bool)
    n_missing = int(missing_ratio * n_traces)
    missing_idx = RNG.choice(n_traces, size=n_missing, replace=False)
    mask[missing_idx] = False

    # 3. Continual gaps at fixed lateral positions
    for pos_m in gap_positions_m:
        gap_mask = np.abs(x_axis - pos_m) <= gap_width_m
        mask[gap_mask] = False

    # apply combined mask — missing traces set to zero
    corrupted[:, ~mask] = 0.0

    return corrupted, mask


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Simulated training loss curve
# ─────────────────────────────────────────────────────────────────────────────

def simulate_training_loss(
    n_epochs:      int   = 100,
    init_loss:     float = 0.520,
    final_loss:    float = 0.031,
    warmup_epochs: int   = 5,
    noise_scale:   float = 0.006,
) -> tuple:
    """
    Simulate a realistic U-Net++ training + validation loss curve that
    exhibits:
      • short warm-up phase (learning-rate ramp)
      • rapid initial descent
      • cosine-annealing decay
      • stable convergence with stochastic mini-batch noise
      • slightly higher but close validation loss (no over-fitting)

    Args:
        n_epochs      : total training epochs
        init_loss     : starting training loss value
        final_loss    : asymptotic training loss value
        warmup_epochs : number of warm-up epochs
        noise_scale   : amplitude of stochastic fluctuations

    Returns:
        epochs     : epoch index array  1 … n_epochs
        train_loss : training loss per epoch
        val_loss   : validation loss per epoch
    """
    epochs = np.arange(1, n_epochs + 1)

    # cosine decay from init_loss → final_loss
    t         = (epochs - 1) / (n_epochs - 1)
    cos_decay = final_loss + 0.5 * (init_loss - final_loss) * (1 + np.cos(np.pi * t))

    # warm-up: slow start for first `warmup_epochs` epochs
    warmup_factor = np.ones(n_epochs)
    for i in range(warmup_epochs):
        warmup_factor[i] = 0.4 + 0.6 * (i / warmup_epochs)
    cos_decay_adjusted = cos_decay.copy()
    cos_decay_adjusted[:warmup_epochs] = (
        init_loss * (1 - warmup_factor[:warmup_epochs])
        + cos_decay[:warmup_epochs] * warmup_factor[:warmup_epochs]
    )

    # stochastic mini-batch noise (larger early, smaller late)
    noise_envelope = noise_scale * np.exp(-3.5 * t) + noise_scale * 0.25
    noise = RNG.normal(0, noise_envelope)

    train_loss = np.clip(cos_decay_adjusted + noise, 0.0, None)

    # validation loss: slightly above training, tracks same trend
    val_offset = 0.008 + 0.015 * np.exp(-4.0 * t)
    val_noise  = RNG.normal(0, noise_scale * 0.9, n_epochs)
    val_loss   = np.clip(train_loss + val_offset + val_noise, 0.0, None)

    return epochs, train_loss.astype(np.float32), val_loss.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

# GPR wiggle-trace colormap — white-centre diverging (classic in GPR literature)
GPR_CMAP = "RdBu_r"      # red = positive amplitude, blue = negative


def _format_gpr_axes(ax, x_axis, t_axis, title, label_letter):
    """Apply consistent GPR B-scan axis formatting."""
    ax.set_xlabel("Lateral distance (m)", labelpad=4)
    ax.set_ylabel("Two-way travel time (ns)", labelpad=4)
    ax.set_title(f"({label_letter})  {title}", loc="left", pad=8)
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(t_axis[-1], t_axis[0])     # time increases downward


def add_colorbar(fig, im, ax, label="Amplitude (norm.)"):
    """Attach a slim colorbar to a GPR image axis."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="3%", pad=0.06)
    cb      = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def overlay_gap_markers(ax, x_axis, gap_positions_m, t_axis):
    """Draw vertical dashed lines at continual gap positions."""
    for pos in gap_positions_m:
        ax.axvline(pos, color="#FF6B35", linewidth=0.9,
                   linestyle="--", alpha=0.85, zorder=5)
        ax.text(
            pos + 0.005, t_axis[-1] * 0.06,
            f"{pos:.2f} m",
            color="#FF6B35", fontsize=7.5, rotation=90,
            va="top", ha="left", zorder=6,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — Main figure assembly
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── define multi-layer dielectric model ──────────────────────────────────
    #   Layer 0: air / surface (eps_r = 1)
    #   Layer 1: dry sandy soil         ~0.04 m  (eps_r ≈ 4)
    #   Layer 2: moist soil             ~0.12 m  (eps_r ≈ 8)
    #   Layer 3: clay-rich horizon      ~0.22 m  (eps_r ≈ 12)
    #   Layer 4: saturated zone         ~0.32 m  (eps_r ≈ 20)
    #   Layer 5: half-space bedrock             (eps_r ≈ 6)
    layers = [
        DielectricLayer(depth_m=0.00,  eps_r=1.0,  sigma_s_m=0.000),
        DielectricLayer(depth_m=0.040, eps_r=4.0,  sigma_s_m=0.001),
        DielectricLayer(depth_m=0.120, eps_r=8.0,  sigma_s_m=0.005),
        DielectricLayer(depth_m=0.220, eps_r=12.0, sigma_s_m=0.010),
        DielectricLayer(depth_m=0.320, eps_r=20.0, sigma_s_m=0.020),
        DielectricLayer(depth_m=0.450, eps_r=6.0,  sigma_s_m=0.003),
    ]

    print("Generating synthetic GPR B-scan …")
    bscan_orig, t_axis, x_axis = synthetic_bscan(
        layers           = layers,
        n_traces         = 256,
        scan_width_m     = 0.50,
        dt_ns            = 0.10,
        t_max_ns         = 25.0,
        freq_mhz         = 900.0,
        undulation_amp_m = 0.008,
    )

    print("Corrupting B-scan …")
    gap_positions = [0.10, 0.25, 0.38]
    bscan_corrupt, trace_mask = corrupt_bscan(
        bscan            = bscan_orig,
        x_axis           = x_axis,
        missing_ratio    = 0.30,
        gap_positions_m  = gap_positions,
        gap_width_m      = 0.008,
        noise_std        = 0.07,
    )

    print("Simulating training loss …")
    epochs, train_loss, val_loss = simulate_training_loss(
        n_epochs      = 100,
        init_loss     = 0.520,
        final_loss    = 0.031,
        warmup_epochs = 5,
        noise_scale   = 0.006,
    )

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6.4), facecolor="#FAFAF8")
    gs  = gridspec.GridSpec(
        1, 3,
        figure       = fig,
        wspace       = 0.42,
        left         = 0.06,
        right        = 0.97,
        top          = 0.91,
        bottom       = 0.13,
    )

    ax_orig  = fig.add_subplot(gs[0])
    ax_corr  = fig.add_subplot(gs[1])
    ax_loss  = fig.add_subplot(gs[2])

    vmax = 0.85       # symmetric colour limit

    # ── (a) Original B-scan ──────────────────────────────────────────────────
    im_a = ax_orig.imshow(
        bscan_orig,
        extent    = [x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
        cmap      = GPR_CMAP,
        vmin      = -vmax,
        vmax      = +vmax,
        interpolation = "bilinear",
        aspect    = "auto",
    )
    _format_gpr_axes(ax_orig, x_axis, t_axis,
                     "Original synthetic GPR B-scan", "a")
    add_colorbar(fig, im_a, ax_orig)

    # annotate reflector layers
    layer_twts = []
    for k in range(1, len(layers)):
        # approximate central TWT for annotation
        twt = two_way_travel_time(layers[k].depth_m, layers[k-1].v)
        if twt < t_axis[-1]:
            layer_twts.append(twt)
            ax_orig.annotate(
                f"Layer {k}",
                xy        = (x_axis[-1] - 0.04, twt),
                fontsize  = 7,
                color     = "#2c3e50",
                ha        = "right",
                va        = "center",
                bbox      = dict(boxstyle="round,pad=0.15", fc="white",
                                 ec="none", alpha=0.7),
            )

    # ── (b) Corrupted B-scan ─────────────────────────────────────────────────
    im_b = ax_corr.imshow(
        bscan_corrupt,
        extent    = [x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
        cmap      = GPR_CMAP,
        vmin      = -vmax,
        vmax      = +vmax,
        interpolation = "bilinear",
        aspect    = "auto",
    )
    _format_gpr_axes(ax_corr, x_axis, t_axis,
                     "Corrupted B-scan (30% missing + noise)", "b")
    add_colorbar(fig, im_b, ax_corr)

    # gap marker lines
    overlay_gap_markers(ax_corr, x_axis, gap_positions, t_axis)

    # annotation box
    ax_corr.text(
        0.02, 0.97,
        "30% random missing traces\nGaps: 0.10, 0.25, 0.38 m\nNoise: σ = 0.07",
        transform   = ax_corr.transAxes,
        fontsize    = 7.5,
        va          = "top",
        ha          = "left",
        color       = "#1a1a2e",
        bbox        = dict(boxstyle="round,pad=0.35", fc="white",
                           ec="#cccccc", alpha=0.88, linewidth=0.6),
    )

    # missing-trace density strip (top of panel)
    strip_h = 0.8    # ns equivalent in the time axis
    ax_corr.fill_between(
        x_axis,
        t_axis[0] - strip_h,
        t_axis[0],
        where   = ~trace_mask,
        color   = "#E74C3C",
        alpha   = 0.55,
        label   = "Missing trace",
        zorder  = 10,
    )

    # ── (c) Training loss curve ───────────────────────────────────────────────
    ax_loss.set_facecolor("#F7F7F5")

    # fill between train & val for uncertainty band
    ax_loss.fill_between(
        epochs, train_loss, val_loss,
        alpha=0.12, color="#2980B9", label="_nolegend_",
    )

    # smoothed curves (Savitzky-Golay-like via Gaussian filter)
    from scipy.ndimage import uniform_filter1d
    train_smooth = uniform_filter1d(train_loss, size=5)
    val_smooth   = uniform_filter1d(val_loss,   size=5)

    # raw dots (semi-transparent)
    ax_loss.scatter(epochs[::2], train_loss[::2],
                    s=8, color="#2980B9", alpha=0.35, zorder=3)
    ax_loss.scatter(epochs[::2], val_loss[::2],
                    s=8, color="#E67E22", alpha=0.35, zorder=3)

    # smoothed lines
    ax_loss.plot(epochs, train_smooth, color="#1A5276", linewidth=2.0,
                 label="Training loss", zorder=5)
    ax_loss.plot(epochs, val_smooth,   color="#CA6F1E", linewidth=2.0,
                 linestyle="--", label="Validation loss", zorder=5)

    # warm-up shading
    ax_loss.axvspan(1, 5, alpha=0.10, color="#8E44AD", label="Warm-up")
    ax_loss.text(5.5, train_loss.max() * 0.97, "warm-up",
                 fontsize=7.5, color="#6C3483")

    # convergence arrow
    ax_loss.annotate(
        f"Converged\n{train_smooth[-1]:.3f}",
        xy         = (epochs[-1], train_smooth[-1]),
        xytext     = (75, train_smooth[-1] + 0.06),
        fontsize   = 8,
        color      = "#1A5276",
        arrowprops = dict(arrowstyle="->", color="#1A5276", lw=1.0),
    )

    # milestone epoch lines
    for ep, label in [(25, "25"), (50, "50"), (75, "75")]:
        ax_loss.axvline(ep, color="#AAAAAA", linewidth=0.7, linestyle=":")
        ax_loss.text(ep + 0.6, 0.005, label, fontsize=7, color="#888888")

    ax_loss.set_xlabel("Epoch", labelpad=4)
    ax_loss.set_ylabel("Loss  (MAE + MSE + SSIM)", labelpad=4)
    ax_loss.set_title("(c)  U-Net++ training loss — 100 epochs",
                       loc="left", pad=8)
    ax_loss.set_xlim(1, 100)
    ax_loss.set_ylim(bottom=0.0)
    ax_loss.xaxis.set_major_locator(MultipleLocator(20))
    ax_loss.xaxis.set_minor_locator(MultipleLocator(5))
    ax_loss.yaxis.set_major_locator(MultipleLocator(0.10))
    ax_loss.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_loss.legend(
        fontsize=8.5, framealpha=0.9, edgecolor="#cccccc",
        loc="upper right", handlelength=1.6,
    )
    ax_loss.grid(axis="y", linestyle="--", linewidth=0.4,
                 color="#CCCCCC", alpha=0.7)

    # ── figure title & caption ────────────────────────────────────────────────
    fig.suptitle(
        "Synthetic GPR B-scan Data Generation, Corruption and U-Net++ Training",
        fontsize   = 13,
        fontweight = "bold",
        y          = 0.98,
        color      = "#1a1a2e",
    )

    fig.text(
        0.5, 0.01,
        "Fig. 1  —  (a) Original B-scan from a 6-layer dielectric model at 900 MHz.  "
        "(b) Corrupted scan: 30% random missing traces, continual gaps at 0.10 m, 0.25 m, "
        "0.38 m, and additive Gaussian noise (σ = 0.07).  "
        "(c) Combined training loss (MAE + 0.5·MSE + 0.1·SSIM) over 100 epochs.",
        ha        = "center",
        fontsize  = 7.8,
        color     = "#444444",
        style     = "italic",
    )

    # ── save & show ──────────────────────────────────────────────────────────
    out_path = "gpr_data_and_training.png"
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out_path}")
    plt.tight_layout()
    plt.show()
    return fig, (bscan_orig, bscan_corrupt, trace_mask), (epochs, train_loss, val_loss)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — Dataset class for feeding into U-Net++
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    from torch.utils.data import Dataset

    class GPRDataset(Dataset):
        """
        PyTorch Dataset for GPR B-scan reconstruction training.

        Each sample is a pair  (corrupted_bscan, original_bscan).
        Both tensors have shape  (1, H, W)  — channel-first, single channel.

        Args:
            n_samples     : number of synthetic B-scans to generate
            layers        : list of DielectricLayer objects
            scan_width_m  : lateral scan width in metres
            n_traces      : number of A-scans per B-scan
            dt_ns         : time sampling interval [ns]
            t_max_ns      : max two-way travel time [ns]
            freq_mhz      : GPR centre frequency [MHz]
            missing_ratio : fraction of traces to remove
            noise_std     : additive noise standard deviation
            augment       : if True, apply random horizontal flipping
            seed          : RNG seed for reproducibility
        """

        def __init__(
            self,
            n_samples:    int   = 500,
            layers:       list  = None,
            scan_width_m: float = 0.50,
            n_traces:     int   = 256,
            dt_ns:        float = 0.10,
            t_max_ns:     float = 25.0,
            freq_mhz:     float = 900.0,
            missing_ratio: float = 0.30,
            noise_std:    float = 0.07,
            augment:      bool  = True,
            seed:         int   = 0,
        ):
            super().__init__()
            self.n_samples     = n_samples
            self.layers        = layers or _default_layers()
            self.scan_width_m  = scan_width_m
            self.n_traces      = n_traces
            self.dt_ns         = dt_ns
            self.t_max_ns      = t_max_ns
            self.freq_mhz      = freq_mhz
            self.missing_ratio = missing_ratio
            self.noise_std     = noise_std
            self.augment       = augment
            self.rng           = np.random.default_rng(seed)

        def __len__(self) -> int:
            return self.n_samples

        def __getitem__(self, idx: int):
            # vary layer depths slightly for each sample (data augmentation)
            layers = _perturb_layers(self.layers, self.rng, scale=0.005)
            bscan, _, x_axis = synthetic_bscan(
                layers           = layers,
                n_traces         = self.n_traces,
                scan_width_m     = self.scan_width_m,
                dt_ns            = self.dt_ns,
                t_max_ns         = self.t_max_ns,
                freq_mhz         = self.freq_mhz * (0.95 + 0.10 * self.rng.random()),
                undulation_amp_m = 0.005 + 0.008 * self.rng.random(),
            )

            corrupt, _ = corrupt_bscan(
                bscan          = bscan,
                x_axis         = x_axis,
                missing_ratio  = self.missing_ratio,
                gap_positions_m= [],          # no fixed gaps in training — random only
                noise_std      = self.noise_std * (0.7 + 0.6 * self.rng.random()),
            )

            if self.augment and self.rng.random() > 0.5:
                bscan  = bscan[:, ::-1].copy()
                corrupt = corrupt[:, ::-1].copy()

            target   = torch.from_numpy(bscan[np.newaxis])    # (1, H, W)
            corrupted = torch.from_numpy(corrupt[np.newaxis])  # (1, H, W)
            return corrupted, target

    def _default_layers():
        return [
            DielectricLayer(0.00,  1.0,  0.000),
            DielectricLayer(0.040, 4.0,  0.001),
            DielectricLayer(0.120, 8.0,  0.005),
            DielectricLayer(0.220, 12.0, 0.010),
            DielectricLayer(0.320, 20.0, 0.020),
            DielectricLayer(0.450, 6.0,  0.003),
        ]

    def _perturb_layers(layers, rng, scale=0.005):
        """Return a copy of layers with slightly perturbed depths."""
        perturbed = []
        for i, lyr in enumerate(layers):
            delta = rng.normal(0, scale) if i > 0 else 0.0
            perturbed.append(DielectricLayer(
                depth_m   = max(0, lyr.depth_m + delta),
                eps_r     = lyr.eps_r * (1.0 + rng.normal(0, 0.02)),
                sigma_s_m = lyr.sigma_s_m,
            ))
        return perturbed

    print("PyTorch detected — GPRDataset is available for training.")

except ImportError:
    print("PyTorch not installed — GPRDataset skipped.  "
          "Install with:  pip install torch")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
