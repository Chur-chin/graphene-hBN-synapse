"""
thz_polariton.py
----------------
THz phonon-polariton (PhP) dispersion in hBN and
polarization-selective near-field coupling to graphene.

KEY PHYSICS
-----------
1. Polariton dispersion in hyperbolic hBN slab (thickness d):
      q(ω) = (i/d) · arctan(ε_top / ε_hBN,z) + nπ/d
   where q is the in-plane wavevector.

2. Polarization angle θ of incoming THz field:
      E_in = E₀ [cos θ  x̂ + sin θ  ẑ]
   - θ = 0°  → in-plane E → excites Band I  (Type-II PhP)
   - θ = 90° → out-of-plane E → excites Band II (Type-I PhP)

3. Near-field coupling to graphene:
      ΔE_F ∝ Im[r_p(q, ω)] · I_THz · f(θ, band)
   where r_p is the p-polarized reflection from graphene/hBN stack.

References
----------
Li et al., Nature 562, 91 (2018)
Basov et al., Science 354, aag1992 (2016)
"""

import numpy as np
from .hbn_permittivity    import hbn_epsilon_perp, hbn_epsilon_par
from .graphene_conductivity import graphene_sigma

# ── Constants ────────────────────────────────────────────────────────
C_SI   = 2.998e8    # m/s
EPS0   = 8.8542e-12 # F/m
CM2RAD = 2.0 * np.pi * C_SI * 100.0   # cm⁻¹ → rad/s


def _omega_rad(omega_cm):
    return np.asarray(omega_cm) * CM2RAD


def polariton_dispersion(omega_cm, d_nm=10.0, n_mode=1, eps_top=1.0, eps_bot=1.0):
    """
    In-plane wavevector q of hBN slab phonon-polariton.

    Uses the thin-slab quantization condition for a uniaxial slab:
        q = (1/d) [n·π - i·arctan(ε_top·√(ε⊥/(-ε∥)) / ε_top)]

    Parameters
    ----------
    omega_cm : float or array  — frequency [cm⁻¹]
    d_nm     : float            — hBN slab thickness [nm]
    n_mode   : int              — mode index (1 = fundamental)
    eps_top  : float            — permittivity of top cladding
    eps_bot  : float            — permittivity of bottom cladding

    Returns
    -------
    q_m1 : complex array  — in-plane wavevector [m⁻¹]
    """
    d   = d_nm * 1.0e-9           # nm → m
    ep  = hbn_epsilon_perp(omega_cm)   # ε⊥
    ez  = hbn_epsilon_par(omega_cm)    # ε∥

    # Hyperbolic ratio
    ratio = np.sqrt(-ep / (ez + 1e-30 * 1j))

    # Quantization: q·d = n·π + arctan(eps_top / (eps_hBN_eff))
    phi = np.arctan(eps_top / (ratio + 1e-30))
    q   = (n_mode * np.pi + phi) / d

    return q   # [m⁻¹]


def polarization_weight(theta_deg, band='I'):
    """
    Coupling efficiency of THz polarization angle to each PhP band.

    Band I  (Type-II, in-plane):    excited by in-plane E  (θ→0°)
    Band II (Type-I,  out-of-plane): excited by out-of-plane E (θ→90°)

    Uses a cosine-squared projection:
        w_I   = cos²(θ)
        w_II  = sin²(θ)

    Parameters
    ----------
    theta_deg : float or array  — THz polarization angle [degrees]
    band      : 'I' or 'II'

    Returns
    -------
    weight : float ∈ [0, 1]
    """
    th = np.asarray(theta_deg) * np.pi / 180.0
    if band == 'I':
        return np.cos(th) ** 2
    elif band == 'II':
        return np.sin(th) ** 2
    else:
        raise ValueError("band must be 'I' or 'II'")


def near_field_coupling(omega_cm, E_F_eV, d_nm=10.0, eps_top=1.0):
    """
    Near-field coupling strength between hBN PhP and graphene.

    Computed as the imaginary part of the p-polarized reflection
    coefficient of the graphene/hBN interface (loss function):

        Im[r_p] = Im[ (q - q_gr) / (q + q_gr) ]

    where q_gr = σ·q / (2·ε₀·ω·ε_eff) is the graphene renormalization.

    Parameters
    ----------
    omega_cm : float or array  [cm⁻¹]
    E_F_eV   : float            [eV]
    d_nm     : float            — hBN thickness [nm]

    Returns
    -------
    Im_rp : real array  — loss function (proportional to ΔE_F coupling)
    """
    omega = np.asarray(omega_cm, dtype=float)
    omega_rad = omega * CM2RAD

    q = polariton_dispersion(omega_cm, d_nm=d_nm, eps_top=eps_top)

    sigma = graphene_sigma(omega_rad + 0j, E_F_eV)

    eps_eff = (hbn_epsilon_perp(omega_cm) + eps_top) / 2.0

    # Graphene wavevector renormalization
    q_gr = q - 1j * sigma * q / (2.0 * EPS0 * omega_rad * eps_eff + 1e-30)

    r_p = (q - q_gr) / (q + q_gr + 1e-30)
    return np.imag(r_p)


def fermi_level_shift(omega_cm, theta_deg, I_THz_Wm2,
                      E_F0_eV=0.10, d_nm=10.0,
                      coupling_const=5.0e-3):
    """
    Estimate Fermi level shift ΔE_F induced by THz pulse.

    ΔE_F = α · Im[r_p(ω)] · I_THz · [w_I(θ) - w_II(θ)]

    Positive → Band I dominant → LTP (excitatory)
    Negative → Band II dominant → LTD (inhibitory)

    Parameters
    ----------
    omega_cm      : float  — representative frequency [cm⁻¹]
    theta_deg     : float  — THz polarization angle [°]
    I_THz_Wm2    : float  — THz intensity [W/m²]
    E_F0_eV      : float  — baseline Fermi energy [eV]
    coupling_const: float  — phenomenological coupling [eV·m²/W]

    Returns
    -------
    delta_EF : float  [eV]
    """
    Im_rp = near_field_coupling(omega_cm, E_F0_eV, d_nm=d_nm)
    wI    = polarization_weight(theta_deg, 'I')
    wII   = polarization_weight(theta_deg, 'II')
    delta_EF = coupling_const * float(np.real(Im_rp)) * I_THz_Wm2 * (wI - wII)
    return delta_EF
