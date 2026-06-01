# GPR-UNetpp-Reconstruction

**Multi-Scale U-Net++ Framework for Robust Reconstruction of Incomplete Ground-Penetrating Radar Data**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2019b%2B-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()

## Overview

This repository contains the complete MATLAB and Python source code for the paper:

> **A Multi-Scale U-Net++ Framework for Robust Reconstruction of 
> Incomplete Ground-Penetrating Radar Data**  
> Ibrar Iqbal, Wang Honghua, Bin Xiong, Lu Yuguo  
> *Journal of Applied Geophysics*, 2026  
> Manuscript No. APPGEO-D-25-00590

The proposed framework reconstructs missing and corrupted GPR B-scan traces 
using a U-Net++ architecture enhanced with nested dense skip connections, 
multi-scale feature aggregation, and squeeze-and-excitation (SE) attention 
blocks. It is evaluated against linear trace interpolation, curvelet-domain 
POCS, and standard U-Net baselines on both synthetic and field-acquired GPR 
datasets.

---

## Repository Structure
GPR-UNetpp-Reconstruction/
│
├── matlab/                          # MATLAB scripts for figures and analysis
│   ├── ricker_wavelet.m             # Ricker wavelet generation (300 MHz)
│   ├── gpr_synthetic_figure6.m      # Synthetic GPR B-scan generation (Fig. 6)
│   ├── generate_figure5.m           # Training convergence + bar chart (Fig. 5)
│   ├── generate_figure10.m          # Single-trace waveform comparison (Fig. 10)
│   ├── generate_figure14.m          # Field waveform comparison (Fig. 14)
│   ├── generate_FK_spectra_figure13.m  # Field F-K spectra (Fig. 13)
│   ├── linear_interp_reconstruction.m  # Linear interpolation baseline
│   ├── statistical_significance_figure.m  # Box plots + Wilcoxon test (Fig. 16)
│   ├── simulate_loss.m              # Trace removal and gap simulation
│   ├── compute_fk.m                 # 2D F-K transform computation
│   ├── load_bscan.m                 # B-scan data loading utility
│   ├── custom_fk_colormap.m         # Custom colormap for F-K plots
│   └── redwhiteblue.m               # Red-white-blue diverging colormap
│
├── python/                          # Python scripts for deep learning
│   ├── unetpp_gpr.py                # Core U-Net++ model definition (PyTorch)
│   ├── unetpp_architecture.py       # Architecture diagram generation
│   ├── gpr_data_and_training.py     # Dataset class, training loop, evaluation
│   ├── gpr_reconstruction_comparison.py  # Reconstruction comparison pipeline
│   ├── fk_spectra_comparison.py     # F-K spectral analysis
│   ├── waveform_comparison.py       # Single-trace waveform plots
│   ├── training_metrics_comparison.py   # Training metrics visualization
│   └── generate_figure.py           # General figure generation utilities
│
├── data/                            # Data placeholder (see Zenodo link below)
│   └── README.md                    # Data access instructions
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md

---

## Key Features

- **U-Net++ with SE attention** — squeeze-and-excitation blocks at all five 
  nested intermediate nodes (reduction ratio r = 16)
- **Composite loss function** — MAE (λ₁ = 1.0) + SSIM (λ₂ = 0.5) + 
  Total Variation (λ₃ = 0.1)
- **Four-method comparison** — linear interpolation, curvelet POCS, U-Net, 
  and proposed U-Net++
- **Statistical validation** — Wilcoxon signed-rank tests across 100 test 
  samples, p < 0.001
- **Architecture:** 3 encoder levels (64/128/256 ch), bottleneck (512 ch), 
  5 nested nodes with SE, 3 decoder levels, ~2 million trainable parameters

---

## Results Summary

| Method | MAE ± std | SNR ± std (dB) | PSNR ± std (dB) |
|---|---|---|---|
| Linear Interpolation | 0.2515 ± 0.031 | −3.31 ± 1.18 | 7.12 ± 1.24 |
| Curvelet POCS | 0.0312 ± 0.0008 | 22.67 ± 0.06 | 28.53 ± 0.10 |
| U-Net | 0.0184 ± 0.0019 | 27.21 ± 0.93 | 33.12 ± 0.93 |
| **U-Net++ (proposed)** | **0.0117 ± 0.0023** | **30.86 ± 0.95** | **36.97 ± 0.95** |

Evaluated on 100 synthetic test samples. All improvements of U-Net++ over 
baselines are statistically significant (p < 0.001, paired one-sided 
Wilcoxon signed-rank test).

---

## Dataset

The synthetic and field GPR datasets used in this study are archived on Zenodo:

> **Zenodo DOI:** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)  
> *(Update this DOI once the Zenodo deposit is complete)*

The field dataset is available upon reasonable request to the corresponding 
author due to data sharing restrictions of the acquisition site.

---

## Requirements

### Python (deep learning)
torch>=1.9.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-image>=0.18.0
h5py>=3.3.0
Install via:
```bash
pip install -r requirements.txt
```

### MATLAB (figure generation and signal processing)
- MATLAB R2019b or later
- Signal Processing Toolbox
- Image Processing Toolbox

---

## Usage

### Training the U-Net++ model
```bash
python python/gpr_data_and_training.py --epochs 100 --lr 1e-4 --batch_size 8
```

### Running reconstruction comparison
```bash
python python/gpr_reconstruction_comparison.py --input data/synthetic_test/
```

### Generating MATLAB figures
```matlab
% In MATLAB, run from the matlab/ directory
run('generate_figure5.m')   % Training convergence + bar chart
run('gpr_synthetic_figure6.m')   % Synthetic data generation figure
run('statistical_significance_figure.m')  % Box plots (Fig. 16)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{iqbal2026unetpp,
  title   = {A Multi-Scale U-Net++ Framework for Robust Reconstruction 
             of Incomplete Ground-Penetrating Radar Data},
  author  = {Iqbal, Ibrar and Honghua, Wang and Xiong, Bin and Yuguo, Lu},
  journal = {Journal of Applied Geophysics},
  year    = {2026},
  note    = {Manuscript No. APPGEO-D-25-00590}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) 
file for details.

---

## Contact

**Ibrar Iqbal** (Corresponding Author)  
Associate Professor, College of Earth Sciences  
Guilin University of Technology, China  
📧 ibrariqbal@glut.edu.cn
