"""
synapse.py
----------
Graphene / hBN artificial synapse with THz-polarization-controlled
synaptic weight dynamics.

Plasticity rules implemented
-----------------------------
1. Analog weight update  : ΔG driven by THz pulse (theta, intensity)
2. STDP kernel           : spike-timing-dependent plasticity
3. LTP / LTD             : θ = 0° / 90° polarization protocol

Synaptic weight w ≡ sheet conductance G_sq [S/sq] of graphene channel.
"""

import numpy as np
from .graphene_conductivity import (
    graphene_sigma, fermi_level_from_density,
    sheet_conductance, delta_conductance, E_Q, HBAR
)
from .thz_polariton import fermi_level_shift


# ── Default device parameters ────────────────────────────────────────
G_MAX   = 1.0e-3   # maximum conductance [S/sq]
G_MIN   = 1.0e-5   # minimum conductance [S/sq]
G_BASE  = 1.0e-4   # resting conductance [S/sq]
TAU_SYN = 10.0e-3  # synaptic time constant [s]  (10 ms)
OMEGA_OP_CM = 790.0  # operating frequency in Band I [cm⁻¹]


class GrapheneHBNSynapse:
    """
    THz-polarization-controlled graphene/hBN artificial synapse.

    Parameters
    ----------
    G_init   : initial conductance (synaptic weight) [S/sq]
    d_nm     : hBN slab thickness [nm]
    E_F0_eV  : baseline Fermi energy [eV]
    tau      : graphene scattering time [s]
    G_min/max: conductance bounds [S/sq]
    """

    def __init__(self, G_init=G_BASE, d_nm=10.0,
                 E_F0_eV=0.10, tau=1e-13,
                 G_min=G_MIN, G_max=G_MAX):
        self.G      = float(G_init)
        self.d_nm   = d_nm
        self.E_F    = E_F0_eV
        self.tau    = tau
        self.G_min  = G_min
        self.G_max  = G_max
        self.history = []   # list of (time, G, theta, delta_EF)

    # ── Core update ──────────────────────────────────────────────────
    def thz_pulse(self, theta_deg, I_THz_Wm2,
                  omega_cm=OMEGA_OP_CM, t=None):
        """
        Apply a single THz pulse and update synaptic weight.

        θ = 0°  → Band I (LTP)  : G increases
        θ = 90° → Band II (LTD) : G decreases
        θ ∈ (0°,90°) → graded analog update

        Parameters
        ----------
        theta_deg  : float  — THz polarization angle [°]
        I_THz_Wm2 : float  — THz intensity [W/m²]
        omega_cm   : float  — carrier frequency [cm⁻¹]
        t          : float  — current time [s] (for history)

        Returns
        -------
        delta_G : float  [S/sq]  — conductance change
        """
        delta_EF = fermi_level_shift(
            omega_cm, theta_deg, I_THz_Wm2,
            E_F0_eV=self.E_F, d_nm=self.d_nm
        )
        dG = delta_conductance(delta_EF)

        self.E_F += delta_EF
        self.G    = float(np.clip(self.G + dG, self.G_min, self.G_max))

        if t is not None:
            self.history.append((t, self.G, theta_deg, delta_EF))

        return dG

    def potentiate(self, n_pulses=10, I_THz=1e6, omega_cm=OMEGA_OP_CM):
        """LTP protocol: n pulses at θ=0° (Band I)."""
        for i in range(n_pulses):
            self.thz_pulse(0.0, I_THz, omega_cm, t=i)
        return self.G

    def depress(self, n_pulses=10, I_THz=1e6, omega_cm=1500.0):
        """LTD protocol: n pulses at θ=90° (Band II)."""
        for i in range(n_pulses):
            self.thz_pulse(90.0, I_THz, omega_cm, t=i)
        return self.G

    def analog_sweep(self, theta_array, I_THz=5e5, omega_cm=OMEGA_OP_CM):
        """
        Sweep polarization angle and record conductance response.

        Parameters
        ----------
        theta_array : array of polarization angles [°]
        I_THz       : THz intensity [W/m²]

        Returns
        -------
        G_arr : array of conductance values [S/sq]
        """
        G_arr = []
        for t, th in enumerate(theta_array):
            self.thz_pulse(th, I_THz, omega_cm, t=t)
            G_arr.append(self.G)
        return np.array(G_arr)

    def reset(self, G_init=G_BASE, E_F0_eV=0.10):
        """Reset synapse to initial state."""
        self.G       = G_init
        self.E_F     = E_F0_eV
        self.history = []

    # ── Post-synaptic current ────────────────────────────────────────
    def epsc(self, V_drive=0.01):
        """
        Post-synaptic current: I = G · V_drive  [A/sq].

        Parameters
        ----------
        V_drive : float  — driving voltage [V]  (default 10 mV)

        Returns
        -------
        I : float  [A]
        """
        return self.G * V_drive

    def __repr__(self):
        return (f"GrapheneHBNSynapse("
                f"G={self.G:.3e} S/sq, E_F={self.E_F:.3f} eV, "
                f"d={self.d_nm} nm)")


# ── STDP kernel ──────────────────────────────────────────────────────
def stdp_kernel(delta_t_ms, A_plus=0.01, A_minus=0.012,
                tau_plus=20.0, tau_minus=20.0):
    """
    Spike-timing-dependent plasticity (STDP) weight change.

    ΔG/G_max = A+ · exp(-Δt/τ+)   if Δt > 0  (pre before post → LTP)
             = -A- · exp(+Δt/τ-)   if Δt < 0  (post before pre → LTD)

    Parameters
    ----------
    delta_t_ms : float or array  — t_post - t_pre  [ms]
    A_plus/minus : float  — plasticity amplitudes
    tau_plus/minus : float  — time constants [ms]

    Returns
    -------
    dW : float or array  — normalised weight change ΔG/G_max
    """
    dt = np.asarray(delta_t_ms, dtype=float)
    dW = np.where(dt > 0,
                  A_plus  * np.exp(-dt / tau_plus),
                  -A_minus * np.exp( dt / tau_minus))
    return dW


def multi_pulse_protocol(synapse, n_cycles=50,
                         theta_ltp=0.0, theta_ltd=90.0,
                         I_THz=5e5, ratio_ltp=0.6):
    """
    Interleaved LTP/LTD pulse protocol.

    Parameters
    ----------
    synapse    : GrapheneHBNSynapse
    n_cycles   : total pulse cycles
    theta_ltp  : LTP polarization [°]
    theta_ltd  : LTD polarization [°]
    I_THz      : THz intensity [W/m²]
    ratio_ltp  : fraction of pulses that are LTP (rest are LTD)

    Returns
    -------
    G_trace : array  [S/sq]
    theta_trace : array [°]
    """
    G_trace, theta_trace = [], []
    for i in range(n_cycles):
        if np.random.rand() < ratio_ltp:
            th = theta_ltp
        else:
            th = theta_ltd
        synapse.thz_pulse(th, I_THz, t=i)
        G_trace.append(synapse.G)
        theta_trace.append(th)
    return np.array(G_trace), np.array(theta_trace)
