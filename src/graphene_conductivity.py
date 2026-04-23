"""
graphene_conductivity.py
------------------------
Optical conductivity of monolayer graphene via the Kubo formula.

σ(ω, E_F, T) = σ_intra(ω) + σ_inter(ω)

  σ_intra : Drude-like intraband term (dominant at THz/far-IR)
  σ_inter : interband term (onset at 2E_F)

The THz conductivity is primarily intraband:
    σ_intra(ω) = (ie²/πℏ²) · E_F / (ω + iτ⁻¹)

Gate voltage V_g shifts E_F → ΔG = Δσ · (W/L)

References
----------
Falkovsky & Varlamov, Eur. Phys. J. B 56, 281 (2007)
Nair et al., Science 320, 1308 (2008)
"""

import numpy as np

# ── Physical constants (SI) ─────────────────────────────────────────
HBAR  = 1.0546e-34   # J·s
E_Q   = 1.6022e-19   # C
KB    = 1.3806e-23   # J/K
EPS0  = 8.8542e-12   # F/m
SIGMA0 = E_Q**2 / (4.0 * HBAR)   # universal conductance ~6.085e-5 S  (= πe²/2h)

# ── Default parameters ──────────────────────────────────────────────
V_FERMI   = 1.0e6    # Fermi velocity [m/s]
TAU_DEFAULT = 1.0e-13  # scattering time [s]  (~100 fs, high-quality graphene)
TEMP_DEFAULT = 300.0   # temperature [K]


def graphene_sigma(omega_rad, E_F_eV, T=TEMP_DEFAULT, tau=TAU_DEFAULT):
    """
    Complex optical conductivity of graphene σ(ω) [S].

    Parameters
    ----------
    omega_rad : float or array  — angular frequency [rad/s]
    E_F_eV    : float            — Fermi energy [eV]
    T         : float            — temperature [K]
    tau       : float            — momentum relaxation time [s]

    Returns
    -------
    sigma : complex ndarray  [S]  (per square, 2D)
    """
    omega = np.asarray(omega_rad, dtype=complex)
    E_F   = E_F_eV * E_Q          # convert eV → J
    mu    = E_F                    # chemical potential = E_F at T=0

    # ── Intraband (Drude) ────────────────────────────────────────────
    # σ_intra = (ie²/πℏ²) · kT · ln(2cosh(E_F/2kT)) / (ω + i/τ)
    kT = KB * T
    ln_term = np.log(2.0 * np.cosh(E_F / (2.0 * kT)))
    sigma_intra = (1j * E_Q**2 / (np.pi * HBAR**2)) * kT * ln_term / (omega + 1j / tau)

    # ── Interband ────────────────────────────────────────────────────
    # σ_inter = (e²/4ℏ) · [H(ω/2) + (4iω/π) ∫ dE (H(E)-H(ω/2))/(ω²-4E²)]
    # Simplified real part (step at ℏω = 2E_F):
    H = lambda E: np.tanh((HBAR * E / 2.0 + mu) / (2.0 * kT)) + \
                  np.tanh((HBAR * E / 2.0 - mu) / (2.0 * kT))

    omega_real = np.real(omega)
    sigma_inter_re = SIGMA0 * H(omega_real)
    sigma_inter_im = -(2.0 * SIGMA0 / np.pi) * np.log(
        np.abs(2.0 * mu - HBAR * omega_real + 1e-30) /
        np.abs(2.0 * mu + HBAR * omega_real + 1e-30)
    )
    sigma_inter = sigma_inter_re + 1j * sigma_inter_im

    return sigma_intra + sigma_inter


def fermi_level_from_density(n_cm2):
    """
    Compute graphene Fermi energy from carrier density.

    For linear Dirac dispersion:  E_F = ℏ v_F √(π |n|)

    Parameters
    ----------
    n_cm2 : float  — 2D carrier density [cm⁻²]  (positive = electron)

    Returns
    -------
    E_F : float  — Fermi energy [eV]
    """
    n_m2 = np.abs(n_cm2) * 1.0e4   # cm⁻² → m⁻²
    E_F_J = HBAR * V_FERMI * np.sqrt(np.pi * n_m2)
    return float(E_F_J / E_Q)       # J → eV


def sheet_conductance(E_F_eV, omega_THz, tau=TAU_DEFAULT):
    """
    Sheet conductance G_sq (THz regime, intraband dominant).

    Parameters
    ----------
    E_F_eV   : float  — Fermi energy [eV]
    omega_THz: float  — frequency [THz]
    tau      : float  — scattering time [s]

    Returns
    -------
    G_sq : float  [S/sq]  (real part)
    """
    omega = 2.0 * np.pi * omega_THz * 1.0e12  # THz → rad/s
    sig   = graphene_sigma(omega, E_F_eV, tau=tau)
    return float(np.real(sig))


def delta_conductance(delta_EF_eV, omega_THz=1.0, tau=TAU_DEFAULT):
    """
    Differential conductance change ΔG for a Fermi level shift ΔE_F.

    Used to map PhP-induced E_F shift → synaptic weight update.

    Parameters
    ----------
    delta_EF_eV : float  — Fermi level shift [eV]
    omega_THz   : float  — operating frequency [THz]

    Returns
    -------
    dG : float  [S/sq]
    """
    E_F0  = 0.10   # baseline Fermi energy [eV]
    G0    = sheet_conductance(E_F0,                 omega_THz, tau)
    G1    = sheet_conductance(E_F0 + delta_EF_eV,   omega_THz, tau)
    return G1 - G0
