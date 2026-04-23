from .hbn_permittivity    import hbn_epsilon_perp, hbn_epsilon_par, hbn_epsilon_tensor
from .graphene_conductivity import graphene_sigma, fermi_level_from_density
from .thz_polariton       import polariton_dispersion, near_field_coupling, polarization_weight
from .synapse             import GrapheneHBNSynapse, stdp_kernel

__version__ = "1.0.0"
__author__  = "Chur Chin"
__email__   = "tpotaoai@gmail.com"
