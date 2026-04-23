"""
utils.py  —  Physical constants, unit helpers, matplotlib style.
"""

import numpy as np
import os

# ── SI constants ─────────────────────────────────────────────────────
C_SI   = 2.99792458e8    # speed of light [m/s]
HBAR   = 1.054571817e-34 # J·s
E_Q    = 1.602176634e-19 # C
KB     = 1.380649e-23    # J/K
EPS0   = 8.8541878e-12   # F/m

# ── Unit conversions ─────────────────────────────────────────────────
def cm1_to_THz(omega_cm):
    """Wavenumber [cm⁻¹] → frequency [THz]."""
    return np.asarray(omega_cm) * C_SI * 1e-10  # 1e2 * 1e-12

def THz_to_cm1(freq_THz):
    """Frequency [THz] → wavenumber [cm⁻¹]."""
    return np.asarray(freq_THz) * 1e12 / (C_SI * 100.0)

def eV_to_J(eV):
    return np.asarray(eV) * E_Q

def J_to_eV(J):
    return np.asarray(J) / E_Q

def nm_to_m(nm):
    return np.asarray(nm) * 1e-9

# ── Plotting style ───────────────────────────────────────────────────
def set_paper_style():
    """Apply matplotlib style for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":    "serif",
        "font.serif":     ["Times New Roman", "DejaVu Serif"],
        "font.size":      10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi":     150,
        "savefig.dpi":    300,
        "savefig.bbox":   "tight",
        "lines.linewidth": 1.5,
    })

# ── CSV I/O ──────────────────────────────────────────────────────────
def save_csv(data_dict, path):
    """Save dict of 1-D arrays to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    keys = list(data_dict.keys())
    arr  = np.column_stack([data_dict[k] for k in keys])
    np.savetxt(path, arr, delimiter=",", header=",".join(keys), comments="")
