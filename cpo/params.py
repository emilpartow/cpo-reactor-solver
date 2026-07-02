"""
Physical, chemical and numerical parameters for the 1-D heterogeneous
catalytic partial-oxidation (CPO) reactor model.

All values are taken from the project's spreadsheets
(Reactor_Properties.xlsx / Modelling_Properties.xlsx) and re-expressed in
strict SI units.  Where the original data contained an obvious typo it is
corrected here and flagged with a comment.

Species order (the index convention used everywhere in the code):

    0: CH4   1: O2   2: CO2   3: CO   4: H2O   5: H2     (the 6 'reactants')
    6: N2    (inert diluent, carried analytically, not part of the state)

Reaction order:

    0: TOX   CH4 + 2 O2  -> CO2 + 2 H2O
    1: SR    CH4 + H2O  <-> CO  + 3 H2
    2: WGS   CO  + H2O  <-> CO2 + H2
    3: HOX   H2  + 1/2 O2 -> H2O
    4: COOX  CO  + 1/2 O2 -> CO2
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# universal constants
# --------------------------------------------------------------------------
R_GAS = 8.314462618          # J/mol/K
SIGMA_SB = 5.670374419e-8    # W/m^2/K^4  (Stefan-Boltzmann)
P_ATM = 101325.0             # Pa per atm  (kinetics use partial pressures in atm)

SPECIES = ["CH4", "O2", "CO2", "CO", "H2O", "H2", "N2"]
N_SP = 6                      # number of reacting species held in the state vector
IDX = {s: i for i, s in enumerate(SPECIES)}

# --------------------------------------------------------------------------
# molar masses [kg/mol]   (N2 corrected: the sheet stored 14.0067, i.e. atomic N)
# --------------------------------------------------------------------------
MW = np.array([16.043, 31.999, 44.01, 28.01, 18.015, 2.016, 28.0134]) * 1e-3

# --------------------------------------------------------------------------
# NASA-7 thermodynamic polynomials.
#   low  range : 200-1000 K   (columns A1..A7)
#   high range : 1000-6000 K  (columns B1..B7)
#   cp/R = a1 + a2 T + a3 T^2 + a4 T^3 + a5 T^4
#   H/RT = a1 + a2/2 T + a3/3 T^2 + a4/4 T^3 + a5/5 T^4 + a6/T
#   S/R  = a1 lnT + a2 T + a3/2 T^2 + a4/3 T^3 + a5/4 T^4 + a7
# (these coefficients already contain the formation enthalpy/entropy, verified
#  against the tabulated Hf, e.g. CH4 -> -74.9 kJ/mol)
# --------------------------------------------------------------------------
NASA_LOW = np.array([
    [ 0.7787415,  0.01747668, -2.783409e-05,  3.049708e-08, -1.2239307e-11, -9825.229,    13.722195],   # CH4
    [ 3.2129360,  0.0011274864,-5.75615e-07,  1.3138773e-09,-8.768554e-13,  -1005.249,     6.034737],   # O2
    [ 2.2757240,  0.009922072, -1.0409113e-05, 6.866686e-09,-2.11728e-12,  -48373.14,     10.188488],   # CO2
    [ 3.262451,   0.0015119409,-3.881755e-06, 5.581944e-09,-2.474951e-12, -14310.539,      4.848897],   # CO
    [ 3.386842,   0.003474982, -6.354696e-06, 6.968581e-09,-2.506588e-12, -30208.11,       2.590232],   # H2O
    [ 3.298124,   0.0008249441,-8.143015e-07,-9.475434e-11, 4.134872e-13,  -1012.5209,    -3.294094],   # H2
    [ 2.9266400,  0.0014879768,-5.68476e-07, -1.0097038e-10,-6.753351e-15,  -922.7977,     5.9802528],  # N2
])
NASA_HIGH = np.array([
    [ 1.683478,   0.010237236, -3.875128e-06, 6.785585e-10,-4.503423e-14, -10080.787,      9.623395],   # CH4
    [ 3.697578,   0.0006135197,-1.258842e-07, 1.775281e-11,-1.1364354e-15, -1233.9301,     3.189165],   # O2
    [ 4.453623,   0.003140168, -1.2784105e-06,2.393996e-10,-1.6690333e-14,-48966.96,      -0.9553959],  # CO2
    [ 3.025078,   0.0014426885,-5.630827e-07, 1.0185813e-10,-6.910951e-15,-14268.35,        6.108217],   # CO
    [ 2.672145,   0.003056293, -8.73026e-07,  1.2009964e-10,-6.391618e-15,-29899.21,        6.862817],   # H2O
    [ 2.991423,   0.0007000644,-5.633828e-08,-9.231578e-12, 1.5827519e-15, -835.034,       -1.35511],    # H2
    [ 2.9266400,  0.0014879768,-5.68476e-07,  1.0097038e-10,-6.753351e-15, -922.7977,       5.9805280],  # N2
])
NASA_TMID = 1000.0

# --------------------------------------------------------------------------
# transport property data (Chung viscosity / kinetic-theory conductivity /
# Fuller diffusion).  Per species, index 0..6.
# --------------------------------------------------------------------------
LJ_SIGMA = np.array([3.758, 3.467, 3.941, 3.69, 2.641, 2.827, 3.798])      # Angstrom
T_CRIT   = np.array([190.4, 154.6, 304.1, 132.9, 647.3, 33.2, 126.2])      # K
V_CRIT   = np.array([99.2, 73.4, 93.9, 93.2, 57.1, 65.1, 90.1])           # cm^3/mol
DIPOLE   = np.array([0.0, 0.0, 0.0, 0.1, 1.8, 0.0, 0.0])                  # Debye
ACENTRIC = np.array([0.011, 0.025, 0.239, 0.066, 0.344, -0.218, 0.037])
ASSOC    = np.array([0.0, 0.0, 0.0, 0.0, 0.075, 0.0, 0.0])
DIFFVOL  = np.array([24.42, 16.6, 26.9, 18.9, 12.7, 7.07, 18.5])          # Fuller diffusion volumes (N2=18.5)

# --------------------------------------------------------------------------
# kinetics  (Arrhenius referenced to 873 K; rate constants per kg catalyst,
# partial pressures in atm).  Adsorption inhibition on O2, CO, H2O.
# --------------------------------------------------------------------------
K_873 = np.array([0.103, 0.1027, 0.06239, 0.01276, 2638.0, 19.38])   # TOX,SR,WGS,RWGS,HOX,COOX
E_A   = np.array([92000., 92000., 25000., 62000., 62000., 76000.])   # J/mol

# adsorption constants (only O2, CO, H2O); index by species
K_ADS = np.zeros(N_SP)
H_ADS = np.zeros(N_SP)
K_ADS[IDX["O2"]],  H_ADS[IDX["O2"]]  = 5.461,  -73000.0
K_ADS[IDX["CO"]],  H_ADS[IDX["CO"]]  = 211.4,  -16000.0
K_ADS[IDX["H2O"]], H_ADS[IDX["H2O"]] = 390.1,  -37000.0

STAB_FACTOR = 1.0e-3          # reactant stabilisation factor sigma~ [atm]

# stoichiometric matrix  RM[reaction, species]  (5 net reactions x 6 species)
#                        CH4   O2   CO2   CO   H2O   H2
RM = np.array([
    [-1.0, -2.0,  1.0,  0.0,  2.0,  0.0],   # TOX  : CH4 + 2 O2 -> CO2 + 2 H2O
    [-1.0,  0.0,  0.0,  1.0, -1.0,  3.0],   # SR   : CH4 + H2O <-> CO + 3 H2
    [ 0.0,  0.0,  1.0, -1.0, -1.0,  1.0],   # WGS  : CO + H2O <-> CO2 + H2
    [ 0.0, -0.5,  0.0,  0.0,  1.0, -1.0],   # HOX  : H2 + 1/2 O2 -> H2O
    [ 0.0, -0.5,  1.0, -1.0,  0.0,  0.0],   # COOX : CO + 1/2 O2 -> CO2
])

# --------------------------------------------------------------------------
# catalyst / washcoat
# --------------------------------------------------------------------------
@dataclass
class Catalyst:
    xi: float = 0.084                 # catalyst (active) volume fraction [-]
    k_s: float = 3.0                  # solid thermal conductivity [W/m/K]
    a_v: float = 2800.0               # specific external surface area [1/m]
    eps_bed: float = 0.8              # bed void fraction [-]
    cp_s: float = 865.0               # solid heat capacity [J/kg/K]
    L_wc: float = 1.0e-5              # washcoat (diffusion) thickness [m]
    #   = 10 um, the midpoint of the 7-15 um range stated in the Model
    #   Description.  (The spreadsheet's 1 um is inconsistent with that doc;
    #   the result is in any case insensitive: peak T +/-5 K over 1-15 um.)
    d_macro: float = 200.0e-9         # macro pore radius [m]
    d_micro: float = 50.0e-9          # micro pore radius [m]
    eps_macro: float = 0.05
    eps_micro: float = 0.5
    rho_bed: float = 3800.0           # solid (Rho_s) density [kg/m^3] (alpha-Al2O3)
    rho_cat: float = 1500.0           # catalyst (washcoat) density [kg/m^3]
    emissivity: float = 0.8
    # reference-model heat-source scaling (RBFunction.m: UseCatalystDensityInHeatSource=true):
    # solid interphase-transfer and reaction sources carry the (1-eps) solid
    # volume fraction, and the reaction heat is scaled by Rho_cat/Rho_s.
    use_cat_density_in_heat_source: bool = True

    @property
    def rho_cat_eff(self) -> float:
        """catalyst mass per unit reactor volume [kg_cat/m^3]"""
        return self.xi * self.rho_cat

# --------------------------------------------------------------------------
# reactor geometry  (axial layout: inert inlet | catalyst | inert outlet)
# --------------------------------------------------------------------------
@dataclass
class Geometry:
    L_inlet: float = 0.015            # inert inlet length [m]
    L_cat: float = 0.020              # catalytic length [m]
    L_outlet: float = 0.015           # inert outlet length [m]
    d_h: float = 1.12e-3              # channel hydraulic diameter [m]
    n_channels: int = 236
    friction: float = 14.3

    @property
    def L_total(self) -> float:
        return self.L_inlet + self.L_cat + self.L_outlet

    @property
    def z_cat_start(self) -> float:
        return self.L_inlet

    @property
    def z_cat_end(self) -> float:
        return self.L_inlet + self.L_cat

    @property
    def channel_area(self) -> float:
        return np.pi * self.d_h ** 2 / 4.0

# --------------------------------------------------------------------------
# feed / operating conditions
# --------------------------------------------------------------------------
@dataclass
class Feed:
    # inlet mass fractions of the 6 reacting species (N2 = balance)
    w_in: np.ndarray = field(default_factory=lambda: np.array(
        [0.174, 0.19, 1e-8, 1e-8, 1e-8, 1e-8]))
    T_gas_in: float = 630.0           # K
    T_solid_in: float = 630.0         # K
    P_in: float = 1.0e5               # Pa  (1 bar)
    volume_flow_NLmin: float = 10.0   # normal litres per minute (per whole monolith)
    T_std: float = 298.15             # K, standard state for the volumetric flow
    P_std: float = 1.0e5              # Pa

    @property
    def w_N2_in(self) -> float:
        return max(0.0, 1.0 - float(np.sum(self.w_in)))


# --------------------------------------------------------------------------
# numerical settings
# --------------------------------------------------------------------------
@dataclass
class Numerics:
    n_inlet: int = 30                 # cells in the inert inlet
    n_cat: int = 120                  # cells in the catalytic zone (refined)
    n_outlet: int = 30                # cells in the inert outlet
    t_end: float = 30.0               # s
    n_save: int = 240                 # stored time snapshots
    tau_wc: float = 1.0e-4            # washcoat pseudo-transient relaxation time [s]
                                     # (analog of the reference AlgPseudoTransientTau;
                                     #  regularises the algebraic surface balance ->
                                     #  stiff ODE; physics is tau_wc-independent)

    @property
    def n_nodes(self) -> int:
        return self.n_inlet + self.n_cat + self.n_outlet


@dataclass
class Config:
    cat: Catalyst = field(default_factory=Catalyst)
    geo: Geometry = field(default_factory=Geometry)
    feed: Feed = field(default_factory=Feed)
    num: Numerics = field(default_factory=Numerics)

    def mixture_MW_in(self) -> float:
        w6 = self.feed.w_in
        inv = np.sum(w6 / MW[:N_SP]) + self.feed.w_N2_in / MW[IDX["N2"]]
        return 1.0 / inv

    def rho_std(self) -> float:
        """feed density at standard conditions [kg/m^3]"""
        return self.feed.P_std * self.mixture_MW_in() / (R_GAS * self.feed.T_std)

    def mass_flux_G(self) -> float:
        """superficial mass flux G [kg/m^2/s] referred to the open channel area."""
        q_std = self.feed.volume_flow_NLmin * 1e-3 / 60.0      # m^3/s at std
        mdot = q_std * self.rho_std()                          # kg/s
        area_total = self.geo.channel_area * self.geo.n_channels
        return mdot / area_total


DEFAULT = Config()
