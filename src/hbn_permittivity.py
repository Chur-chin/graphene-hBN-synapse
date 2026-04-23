"""
hbn_permittivity.py
-------------------
Anisotropic (uniaxial) dielectric function of hexagonal boron nitride (hBN).

hBN is a natural hyperbolic material with TWO Reststrahlen bands:

  Band I  (in-plane,  ε⊥):  760 – 825  cm⁻¹   (~22.8–24.7 THz)
  Band II (out-of-plane, ε∥): 1370 – 1610 cm⁻¹  (~41.1–48.3 THz)

In each band one permittivity component is NEGATIVE while the other
remains positive → hyperbolic dispersion → confined phonon-polaritons.

Drude-Lorentz model:
    ε_j(ω) = ε_j,∞ · [1 + (ω_LO,j² - ω_TO,j²) / (ω_TO,j² - ω² - iγ_j·ω)]

References
----------
Caldwell et al., Nat. Commun. 5, 5221 (2014)
Dai et al., Science 343, 1125 (2014)
"""

import numpy as np

# ── Physical constants ──────────────────────────────────────────────
C_CM = 2.998e10          # speed of light [cm/s]

# ── hBN Drude-Lorentz parameters ───────────────────────────────────
# in-plane (perpendicular to c-axis):  ε⊥
PARAMS_PERP = dict(
    eps_inf =  4.87,           # high-frequency permittivity
    omega_TO=  760.0,          # TO phonon [cm⁻¹]  Band I lower edge
    omega_LO=  825.0,          # LO phonon [cm⁻¹]  Band I upper edge
    gamma   =    5.0,          # damping   [cm⁻¹]
)

# out-of-plane (parallel to c-axis):  ε∥
PARAMS_PAR = dict(
    eps_inf =  2.95,
    omega_TO= 1370.0,          # Band II lower edge
    omega_LO= 1610.0,          # Band II upper edge
    gamma   =    5.0,
)


def _drude_lorentz(omega_cm, eps_inf, omega_TO, omega_LO, gamma):
    """
    Single-oscillator Drude-Lorentz permittivity.

    Parameters
    ----------
    omega_cm : float or array  — frequency [cm⁻¹]

    Returns
    -------
    eps : complex permittivity
    """
    omega  = np.asarray(omega_cm, dtype=complex)
    num    = omega_LO**2 - omega_TO**2
    denom  = omega_TO**2 - omega**2 - 1j * gamma * omega
    return eps_inf * (1.0 + num / denom)


def hbn_epsilon_perp(omega_cm):
    """
    In-plane permittivity ε⊥(ω).  Negative inside Band I (760–825 cm⁻¹).

    Parameters
    ----------
    omega_cm : float or array  [cm⁻¹]

    Returns
    -------
    eps_perp : complex ndarray
    """
    return _drude_lorentz(omega_cm, **PARAMS_PERP)


def hbn_epsilon_par(omega_cm):
    """
    Out-of-plane permittivity ε∥(ω).  Negative inside Band II (1370–1610 cm⁻¹).

    Parameters
    ----------
    omega_cm : float or array  [cm⁻¹]

    Returns
    -------
    eps_par : complex ndarray
    """
    return _drude_lorentz(omega_cm, **PARAMS_PAR)


def hbn_epsilon_tensor(omega_cm):
    """
    Full uniaxial permittivity tensor diag(ε⊥, ε⊥, ε∥).

    Returns
    -------
    eps : ndarray shape (..., 3, 3) — diagonal tensor
    """
    ep = hbn_epsilon_perp(omega_cm)
    ez = hbn_epsilon_par(omega_cm)
    omega_arr = np.atleast_1d(omega_cm)
    n = omega_arr.shape[0]
    tensor = np.zeros((n, 3, 3), dtype=complex)
    tensor[:, 0, 0] = ep
    tensor[:, 1, 1] = ep
    tensor[:, 2, 2] = ez
    return tensor


def hyperbolic_type(omega_cm):
    """
    Return the hyperbolic type at frequency omega_cm.

    Type I  (Band II): ε∥ < 0, ε⊥ > 0  → out-of-plane confinement
    Type II (Band I) : ε⊥ < 0, ε∥ > 0  → in-plane confinement
    Elliptic: both positive (no surface PhP)

    Returns
    -------
    label : str  'Type-I' | 'Type-II' | 'Elliptic'
    """
    ep = hbn_epsilon_perp(omega_cm).real
    ez = hbn_epsilon_par(omega_cm).real
    if ep > 0 and ez < 0:
        return "Type-I"
    elif ep < 0 and ez > 0:
        return "Type-II"
    else:
        return "Elliptic"
