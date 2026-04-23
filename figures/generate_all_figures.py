"""
generate_all_figures.py
-----------------------
Reproduce all figures for the graphene-hBN synapse paper.

Fig 1 : hBN dielectric tensor — Reststrahlen bands
Fig 2 : PhP dispersion (q vs ω) in hBN slab
Fig 3 : Graphene conductivity vs Fermi energy & frequency
Fig 4 : Polarization-angle-dependent synaptic weight update
Fig 5 : STDP kernel + multi-pulse LTP/LTD protocol

Usage:
    python figures/generate_all_figures.py
    python figures/generate_all_figures.py --fig 4
"""

import sys, os, argparse, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hbn_permittivity     import hbn_epsilon_perp, hbn_epsilon_par
from src.graphene_conductivity import graphene_sigma, sheet_conductance
from src.thz_polariton        import polariton_dispersion, polarization_weight, fermi_level_shift
from src.synapse               import GrapheneHBNSynapse, stdp_kernel, multi_pulse_protocol
from src.utils                 import cm1_to_THz, set_paper_style

set_paper_style()
OUT = os.path.dirname(os.path.abspath(__file__))


# ── Fig 1 : hBN permittivity ────────────────────────────────────────
def fig1():
    print("Fig 1: hBN dielectric tensor...")
    omega = np.linspace(600, 1700, 1200)

    ep = hbn_epsilon_perp(omega)
    ez = hbn_epsilon_par(omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(omega, ep.real, 'b-', label=r'Re[$\varepsilon_\perp$] Band I')
    ax1.plot(omega, ep.imag, 'b--', alpha=0.6, label=r'Im[$\varepsilon_\perp$]')
    ax1.axhline(0, color='k', lw=0.8, ls='--')
    ax1.axvspan(760, 825, alpha=0.15, color='blue', label='Band I (760–825 cm⁻¹)')
    ax1.set_ylabel(r'$\varepsilon_\perp(\omega)$', fontsize=11)
    ax1.set_ylim(-30, 30); ax1.legend(fontsize=8); ax1.set_title('(a)')

    ax2.plot(omega, ez.real, 'r-', label=r'Re[$\varepsilon_\parallel$] Band II')
    ax2.plot(omega, ez.imag, 'r--', alpha=0.6, label=r'Im[$\varepsilon_\parallel$]')
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.axvspan(1370, 1610, alpha=0.15, color='red', label='Band II (1370–1610 cm⁻¹)')
    ax2.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontsize=11)
    ax2.set_ylabel(r'$\varepsilon_\parallel(\omega)$', fontsize=11)
    ax2.set_ylim(-30, 30); ax2.legend(fontsize=8); ax2.set_title('(b)')

    plt.suptitle('FIG. 1.  hBN anisotropic permittivity — Reststrahlen bands', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1_hbn_permittivity.png'))
    plt.close()
    print("  -> fig1_hbn_permittivity.png")


# ── Fig 2 : PhP dispersion ──────────────────────────────────────────
def fig2():
    print("Fig 2: PhP dispersion...")
    # Band I
    omega_I  = np.linspace(762, 823, 300)
    q_I_10nm = polariton_dispersion(omega_I, d_nm=10)
    q_I_30nm = polariton_dispersion(omega_I, d_nm=30)

    # Band II
    omega_II  = np.linspace(1380, 1600, 300)
    q_II_10nm = polariton_dispersion(omega_II, d_nm=10, n_mode=1)
    q_II_30nm = polariton_dispersion(omega_II, d_nm=30, n_mode=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    ax1.plot(np.real(q_I_10nm) * 1e-6, omega_I, 'b-',  label='d = 10 nm')
    ax1.plot(np.real(q_I_30nm) * 1e-6, omega_I, 'b--', label='d = 30 nm')
    ax1.set_xlabel(r'Re[$q$]  (μm$^{-1}$)', fontsize=11)
    ax1.set_ylabel(r'Frequency (cm$^{-1}$)', fontsize=11)
    ax1.set_title('(a)  Band I PhP dispersion\n(Type-II hyperbolic, 760–825 cm⁻¹)')
    ax1.legend(); ax1.set_xlim(left=0)

    ax2.plot(np.real(q_II_10nm) * 1e-6, omega_II, 'r-',  label='d = 10 nm')
    ax2.plot(np.real(q_II_30nm) * 1e-6, omega_II, 'r--', label='d = 30 nm')
    ax2.set_xlabel(r'Re[$q$]  (μm$^{-1}$)', fontsize=11)
    ax2.set_ylabel(r'Frequency (cm$^{-1}$)', fontsize=11)
    ax2.set_title('(b)  Band II PhP dispersion\n(Type-I hyperbolic, 1370–1610 cm⁻¹)')
    ax2.legend(); ax2.set_xlim(left=0)

    plt.suptitle('FIG. 2.  hBN phonon-polariton dispersion for two slab thicknesses', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig2_php_dispersion.png'))
    plt.close()
    print("  -> fig2_php_dispersion.png")


# ── Fig 3 : Graphene conductivity ───────────────────────────────────
def fig3():
    print("Fig 3: Graphene conductivity...")
    EF_arr  = np.linspace(0.01, 0.5, 200)  # eV
    freq_THz = [0.5, 1.0, 2.0, 5.0]
    colors   = ['navy', 'royalblue', 'steelblue', 'skyblue']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for freq, col in zip(freq_THz, colors):
        G_arr = [sheet_conductance(ef, freq) for ef in EF_arr]
        ax1.plot(EF_arr, np.array(G_arr) * 1e3, color=col,
                 label=f'{freq} THz')
    ax1.set_xlabel(r'Fermi energy $E_F$ (eV)', fontsize=11)
    ax1.set_ylabel(r'Sheet conductance $G$ (mS/sq)', fontsize=11)
    ax1.set_title('(a)  G vs E_F at various THz frequencies')
    ax1.legend(fontsize=8)

    omega_THz = np.linspace(0.1, 10.0, 300)
    for EF, col in zip([0.05, 0.10, 0.20, 0.40], colors):
        G_arr = [sheet_conductance(EF, f) for f in omega_THz]
        ax2.plot(omega_THz, np.array(G_arr) * 1e3, color=col,
                 label=f'$E_F$ = {EF} eV')
    ax2.set_xlabel(r'Frequency (THz)', fontsize=11)
    ax2.set_ylabel(r'Sheet conductance $G$ (mS/sq)', fontsize=11)
    ax2.set_title('(b)  G vs frequency at various E_F')
    ax2.legend(fontsize=8)

    plt.suptitle('FIG. 3.  Graphene THz sheet conductance (Kubo formula)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig3_graphene_conductivity.png'))
    plt.close()
    print("  -> fig3_graphene_conductivity.png")


# ── Fig 4 : Polarization-controlled weight update ───────────────────
def fig4():
    print("Fig 4: Polarization sweep...")
    theta_arr = np.linspace(0, 90, 91)
    I_THz     = 5e5  # W/m²

    # Weight functions
    wI  = polarization_weight(theta_arr, 'I')
    wII = polarization_weight(theta_arr, 'II')

    # Synaptic weight (conductance) after single pulse at each angle
    syn = GrapheneHBNSynapse()
    G_arr = []
    dEF_arr = []
    for th in theta_arr:
        syn.reset()
        syn.thz_pulse(th, I_THz)
        G_arr.append(syn.G)
        dEF_arr.append(syn.E_F - 0.10)

    G_arr   = np.array(G_arr)
    dEF_arr = np.array(dEF_arr)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].plot(theta_arr, wI,  'b-', lw=2, label='Band I (LTP)')
    axes[0].plot(theta_arr, wII, 'r-', lw=2, label='Band II (LTD)')
    axes[0].set_xlabel(r'THz polarization angle $\theta$ (°)', fontsize=10)
    axes[0].set_ylabel('Coupling weight', fontsize=10)
    axes[0].set_title('(a)  Band coupling vs θ')
    axes[0].legend()

    axes[1].plot(theta_arr, dEF_arr * 1e3, 'g-', lw=2)
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    axes[1].set_xlabel(r'$\theta$ (°)', fontsize=10)
    axes[1].set_ylabel(r'$\Delta E_F$ (meV)', fontsize=10)
    axes[1].set_title(r'(b)  Fermi level shift $\Delta E_F$')

    axes[2].plot(theta_arr, G_arr * 1e4, 'm-', lw=2)
    axes[2].axhline(1.0, color='k', lw=0.8, ls='--', label='Baseline')
    axes[2].set_xlabel(r'$\theta$ (°)', fontsize=10)
    axes[2].set_ylabel(r'Conductance $G$ (×10$^{-4}$ S/sq)', fontsize=10)
    axes[2].set_title('(c)  Synaptic weight $G$ after pulse')
    axes[2].legend(fontsize=8)

    plt.suptitle(r'FIG. 4.  THz polarization $\theta$ controls synaptic weight', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_polarization_weight.png'))
    plt.close()
    print("  -> fig4_polarization_weight.png")


# ── Fig 5 : STDP + LTP/LTD protocol ────────────────────────────────
def fig5():
    print("Fig 5: STDP and multi-pulse protocol...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # STDP kernel
    dt_arr = np.linspace(-80, 80, 400)
    dW     = stdp_kernel(dt_arr)
    ax1.plot(dt_arr[dt_arr > 0],  dW[dt_arr > 0]  * 100, 'b-', lw=2, label='LTP (Δt > 0)')
    ax1.plot(dt_arr[dt_arr <= 0], dW[dt_arr <= 0] * 100, 'r-', lw=2, label='LTD (Δt < 0)')
    ax1.axhline(0, color='k', lw=0.8)
    ax1.axvline(0, color='k', lw=0.8)
    ax1.set_xlabel(r'$\Delta t = t_{\rm post} - t_{\rm pre}$ (ms)', fontsize=10)
    ax1.set_ylabel(r'$\Delta G / G_{\rm max}$ (%)', fontsize=10)
    ax1.set_title('(a)  STDP kernel')
    ax1.legend()

    # Multi-pulse LTP/LTD protocol
    np.random.seed(42)
    syn = GrapheneHBNSynapse()
    G_trace, theta_trace = multi_pulse_protocol(syn, n_cycles=100,
                                                I_THz=5e5, ratio_ltp=0.6)
    pulse_idx = np.arange(len(G_trace))
    ltp_mask  = theta_trace == 0.0

    ax2.plot(pulse_idx, G_trace * 1e4, 'k-', lw=0.8, alpha=0.7)
    ax2.scatter(pulse_idx[ltp_mask],  G_trace[ltp_mask]  * 1e4,
                s=12, c='blue',  label='LTP (θ=0°)')
    ax2.scatter(pulse_idx[~ltp_mask], G_trace[~ltp_mask] * 1e4,
                s=12, c='red',   label='LTD (θ=90°)')
    ax2.set_xlabel('Pulse index', fontsize=10)
    ax2.set_ylabel(r'Conductance $G$ (×10$^{-4}$ S/sq)', fontsize=10)
    ax2.set_title('(b)  Multi-pulse LTP / LTD protocol')
    ax2.legend(fontsize=8)

    plt.suptitle('FIG. 5.  STDP kernel and polarization-driven weight modulation', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig5_stdp_plasticity.png'))
    plt.close()
    print("  -> fig5_stdp_plasticity.png")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=int, choices=[1,2,3,4,5])
    args = parser.parse_args()
    if args.fig:
        FIGS[args.fig]()
    else:
        for f in FIGS.values():
            f()
    print("Done.")
