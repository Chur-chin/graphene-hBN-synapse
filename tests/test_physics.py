"""
test_physics.py  —  run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.hbn_permittivity      import hbn_epsilon_perp, hbn_epsilon_par, hyperbolic_type
from src.graphene_conductivity import graphene_sigma, fermi_level_from_density, sheet_conductance
from src.thz_polariton         import polarization_weight, fermi_level_shift
from src.synapse               import GrapheneHBNSynapse, stdp_kernel


# ── hBN permittivity ────────────────────────────────────────────────
def test_band_I_negative():
    """ε⊥ must be negative inside Band I (760–825 cm⁻¹)."""
    for w in [770, 790, 810]:
        assert hbn_epsilon_perp(w).real < 0, f"ε⊥ should be negative at {w} cm⁻¹"

def test_band_II_negative():
    """ε∥ must be negative inside Band II (1370–1610 cm⁻¹)."""
    for w in [1400, 1500, 1590]:
        assert hbn_epsilon_par(w).real < 0, f"ε∥ should be negative at {w} cm⁻¹"

def test_outside_bands_positive():
    """Outside both bands, permittivities should be positive."""
    assert hbn_epsilon_perp(600).real > 0
    assert hbn_epsilon_par(600).real  > 0

def test_hyperbolic_type_band_I():
    assert hyperbolic_type(790) == "Type-II"

def test_hyperbolic_type_band_II():
    assert hyperbolic_type(1500) == "Type-I"


# ── Graphene conductivity ───────────────────────────────────────────
def test_sigma_complex():
    sig = graphene_sigma(2*np.pi*1e12, 0.1)
    assert np.iscomplex(sig) or isinstance(sig, complex)

def test_sigma_increases_with_EF():
    """Higher E_F → higher intraband conductivity at THz."""
    omega = 2*np.pi*1e12
    s1 = np.real(graphene_sigma(omega, 0.05))
    s2 = np.real(graphene_sigma(omega, 0.20))
    assert s2 > s1

def test_fermi_from_density_positive():
    EF = fermi_level_from_density(1e12)
    assert EF > 0

def test_sheet_conductance_positive():
    G = sheet_conductance(0.1, 1.0)
    assert G > 0


# ── THz polarization ───────────────────────────────────────────────
def test_polarization_weight_sum_to_one():
    for th in [0, 30, 45, 60, 90]:
        wI  = polarization_weight(th, 'I')
        wII = polarization_weight(th, 'II')
        assert abs(wI + wII - 1.0) < 1e-10

def test_ltp_at_zero_degrees():
    """θ=0° → Band I dominant → LTP (ΔE_F > 0)."""
    dEF = fermi_level_shift(790, 0.0, 1e6)
    assert dEF > 0

def test_ltd_at_ninety_degrees():
    """θ=90° → Band II dominant → LTD (ΔE_F < 0)."""
    dEF = fermi_level_shift(790, 90.0, 1e6)
    assert dEF < 0


# ── Synapse ────────────────────────────────────────────────────────
def test_synapse_ltp_increases_G():
    syn = GrapheneHBNSynapse()
    G0  = syn.G
    syn.potentiate(n_pulses=5, I_THz=1e6)
    assert syn.G >= G0

def test_synapse_ltd_decreases_G():
    syn = GrapheneHBNSynapse()
    G0  = syn.G
    syn.depress(n_pulses=5, I_THz=1e6)
    assert syn.G <= G0

def test_synapse_bounds():
    syn = GrapheneHBNSynapse()
    syn.potentiate(n_pulses=1000, I_THz=1e9)
    assert syn.G <= syn.G_max
    syn.depress(n_pulses=1000, I_THz=1e9)
    assert syn.G >= syn.G_min

def test_stdp_ltp_positive():
    assert stdp_kernel(10.0) > 0

def test_stdp_ltd_negative():
    assert stdp_kernel(-10.0) < 0

def test_stdp_asymmetry():
    """LTD amplitude slightly larger than LTP (A_minus > A_plus)."""
    assert abs(stdp_kernel(-10.0)) > abs(stdp_kernel(10.0))
