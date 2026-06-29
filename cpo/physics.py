"""
Constitutive closures for the CPO reactor model:

  * thermodynamics      -> cp_i(T), H_i(T), S_i(T), dH_reaction(T), K_eq(T)
  * gas mixture state   -> mole fractions, density, partial pressures
  * transport           -> Chung viscosity, Eucken conductivity, Fuller
                           diffusion, Wilke mixing rules
  * transfer coeffs     -> Re, Pr, Sc, Sherwood/Nusselt (Graetz) -> beta, alpha
  * kinetics            -> 5 net LHHW reaction rates
  * effectiveness       -> slab Thiele modulus with Knudsen diffusion

Everything is vectorised over the axial nodes.  Temperatures in K, pressures in
Pa internally; reaction kinetics use partial pressures in atm (as the rate
constants were fitted).
"""
from __future__ import annotations
import numpy as np
from . import params as P
from .params import (R_GAS, P_ATM, MW, N_SP, IDX, NASA_LOW, NASA_HIGH, NASA_TMID,
                     LJ_SIGMA, T_CRIT, V_CRIT, DIPOLE, ACENTRIC, ASSOC, DIFFVOL,
                     K_873, E_A, K_ADS, H_ADS, STAB_FACTOR, RM)

# ==========================================================================
#  THERMODYNAMICS  (NASA-7)
# ==========================================================================
def _nasa_pick(T):
    """return coefficient array (n,7,7) selecting low/high range per node."""
    T = np.atleast_1d(T)
    low = T < NASA_TMID
    coeff = np.where(low[:, None, None], NASA_LOW[None], NASA_HIGH[None])
    return coeff  # (n, 7species, 7coeff)


def cp_species(T):
    """molar heat capacity [J/mol/K], shape (n,7)."""
    T = np.atleast_1d(T).astype(float)
    c = _nasa_pick(T)
    a1, a2, a3, a4, a5 = (c[..., k] for k in range(5))
    cp_R = a1 + a2 * T[:, None] + a3 * T[:, None]**2 + a4 * T[:, None]**3 + a5 * T[:, None]**4
    return cp_R * R_GAS


def enthalpy_species(T):
    """absolute molar enthalpy [J/mol], shape (n,7)."""
    T = np.atleast_1d(T).astype(float)
    c = _nasa_pick(T)
    a1, a2, a3, a4, a5, a6 = (c[..., k] for k in range(6))
    Tn = T[:, None]
    H_RT = a1 + a2/2*Tn + a3/3*Tn**2 + a4/4*Tn**3 + a5/5*Tn**4 + a6/Tn
    return H_RT * R_GAS * Tn


def entropy_species(T):
    """absolute molar entropy [J/mol/K], shape (n,7)."""
    T = np.atleast_1d(T).astype(float)
    c = _nasa_pick(T)
    a1, a2, a3, a4, a5, _, a7 = (c[..., k] for k in range(7))
    Tn = T[:, None]
    S_R = a1*np.log(Tn) + a2*Tn + a3/2*Tn**2 + a4/3*Tn**3 + a5/4*Tn**4 + a7
    return S_R * R_GAS


def dH_reactions(T):
    """reaction enthalpies [J/mol], shape (n,5)  (negative = exothermic)."""
    H = enthalpy_species(T)[:, :N_SP]          # (n,6)
    return H @ RM.T                            # (n,5)


def K_eq(T):
    """dimensionless equilibrium constants (std state 1 atm), shape (n,5)."""
    H = enthalpy_species(T)[:, :N_SP]
    S = entropy_species(T)[:, :N_SP]
    dG = (H @ RM.T) - T[:, None] * (S @ RM.T)  # (n,5)  J/mol
    return np.exp(-dG / (R_GAS * T[:, None]))


# ==========================================================================
#  GAS MIXTURE STATE
# ==========================================================================
def composition(w6, T, P_pa):
    """
    From reacting-species mass fractions w6 (n,6), temperature (n,), pressure (n,)
    return:  x (n,7) mole fractions incl N2, M_mix (n,), rho (n,) [kg/m3],
             p_atm (n,6) partial pressures [atm], C (n,6) [mol/m3].
    """
    w6 = np.clip(w6, 0.0, None)
    s = w6.sum(axis=1)
    # renormalise only where the 6 reacting fractions already exceed 1
    over = s > 1.0
    if np.any(over):
        w6 = w6.copy()
        w6[over] /= s[over][:, None]
    w_N2 = np.clip(1.0 - w6.sum(axis=1), 0.0, None)

    inv_M = (w6 / MW[None, :N_SP]).sum(axis=1) + w_N2 / MW[IDX["N2"]]
    M_mix = 1.0 / inv_M
    x6 = w6 * M_mix[:, None] / MW[None, :N_SP]
    x_N2 = w_N2 * M_mix / MW[IDX["N2"]]
    x = np.concatenate([x6, x_N2[:, None]], axis=1)

    rho = P_pa * M_mix / (R_GAS * T)
    p_atm = x6 * (P_pa / P_ATM)[:, None]
    C = x6 * (P_pa / (R_GAS * T))[:, None]
    return x, M_mix, rho, p_atm, C


# ==========================================================================
#  TRANSPORT PROPERTIES
# ==========================================================================
def _omega_visc(T, i):
    """Neufeld collision integral via reduced temperature (Chung)."""
    Tstar = 1.2593 / T_CRIT[i] * T
    return (1.16145 * Tstar**(-0.14874) + 0.52487*np.exp(-0.77320*Tstar)
            + 2.16178*np.exp(-2.43787*Tstar))


def viscosity_species(T):
    """pure-species viscosity [Pa s], shape (n,7), Chung method."""
    T = np.atleast_1d(T).astype(float)
    mu = np.empty((T.size, 7))
    M_g = MW * 1e3                              # g/mol for the Chung formula
    for i in range(7):
        Om = _omega_visc(T, i)
        mu_r = 131.3 * DIPOLE[i] / np.sqrt(T_CRIT[i] * V_CRIT[i])
        Fc = 1 - 0.2756*ACENTRIC[i] + 0.059035*mu_r**4 + ASSOC[i]
        mu[:, i] = 1e-7 * 40.785 * Fc * np.sqrt(M_g[i]*T) / (V_CRIT[i]**(2/3) * Om)
    return mu


def conductivity_species(T):
    """pure-species thermal conductivity [W/m/K], modified Eucken, shape (n,7)."""
    mu = viscosity_species(T)
    cp = np.empty((np.atleast_1d(T).size, 7))
    cp_all = cp_species(T)                      # (n,7) molar
    cp[:] = cp_all
    # modified Eucken: lambda = (cp + 1.25 R) * mu / M     (M in kg/mol)
    return (cp + 1.25 * R_GAS) * mu / MW[None, :]


def _wilke_mix(prop, x):
    """Wilke mixing rule for viscosity/conductivity. prop,x: (n,7) -> (n,)."""
    n = prop.shape[0]
    M = MW
    # Phi_ij = [1 + (mu_i/mu_j)^0.5 (M_j/M_i)^0.25]^2 / sqrt(8 (1 + M_i/M_j))
    ratio_mu = prop[:, :, None] / prop[:, None, :]            # (n,i,j)
    ratio_M = (M[None, :] / M[:, None])                       # (i,j) = M_j/M_i
    num = (1.0 + np.sqrt(ratio_mu) * (ratio_M[None])**0.25)**2
    den = np.sqrt(8.0 * (1.0 + (M[:, None] / M[None, :])[None]))
    Phi = num / den                                           # (n,i,j)
    denom = np.einsum('nj,nij->ni', x, Phi)                   # (n,i)
    denom = np.maximum(denom, 1e-30)
    return np.sum(x * prop / denom, axis=1)


def diffusion_in_N2(T, P_pa):
    """Fuller-Schettler-Giddings diffusivity of each reactant in N2 [m2/s] (n,6)."""
    T = np.atleast_1d(T).astype(float)
    P_bar = P_pa / 1e5
    M_g = MW * 1e3
    vN2 = DIFFVOL[IDX["N2"]]
    D = np.empty((T.size, N_SP))
    for i in range(N_SP):
        M_ab = 2.0 / (1.0/M_g[i] + 1.0/M_g[IDX["N2"]])
        Dcm2 = 0.00143 * T**1.75 / (P_bar * np.sqrt(M_ab) *
                                    (DIFFVOL[i]**(1/3) + vN2**(1/3))**2)
        D[:, i] = Dcm2 * 1e-4
    return D


def gas_properties(w6, T, P_pa):
    """bundle: mu_mix, lam_mix, D_i(n,6), cp_mass(n,), rho(n,), x(n,7)."""
    x, M_mix, rho, p_atm, C = composition(w6, T, P_pa)
    mu_i = viscosity_species(T)
    lam_i = conductivity_species(T)
    mu_mix = _wilke_mix(mu_i, x)
    lam_mix = _wilke_mix(lam_i, x)
    D_i = diffusion_in_N2(T, P_pa)
    cp_molar_mix = np.sum(x * cp_species(T), axis=1)           # J/mol/K
    cp_mass = cp_molar_mix / M_mix                             # J/kg/K
    return dict(mu=mu_mix, lam=lam_mix, D=D_i, cp_mass=cp_mass,
                rho=rho, x=x, M_mix=M_mix, p_atm=p_atm, C=C)


def transfer_coefficients(gp, geo, G, eps, z_local_active, active_mask):
    """
    Heat (alpha, n) and mass (beta, n x 6) transfer coefficients.
    Graetz developing-flow correlation inside the catalytic zone, fully
    developed laminar/Gnielinski outside.
    """
    mu, lam, D, cp_mass, rho = gp['mu'], gp['lam'], gp['D'], gp['cp_mass'], gp['rho']
    d_h = geo.d_h
    n = mu.size
    Re = d_h * G / (eps * mu)
    Pr = cp_mass * mu / lam

    Nu = np.empty(n)
    # outside catalyst: laminar const-flux (4.36) or Gnielinski if turbulent
    lam_flow = Re < 2300
    Nu_lam = np.full(n, 4.36)
    xi_f = (0.790 * np.log(np.maximum(Re, 1e3)) - 1.64) ** (-2)
    Nu_turb = (xi_f/8) * (Re - 1000.0) * Pr / (1 + 12.7*np.sqrt(xi_f/8) * (Pr**(2/3) - 1))
    Nu_out = np.where(lam_flow, Nu_lam, np.maximum(Nu_turb, 4.36))

    # inside catalyst: Graetz / Hawthorn developing correlation
    Zt = np.maximum(z_local_active / (d_h * Re * Pr), 1e-12)
    Nu_in = 2.977 + 8.827 * (1000.0 * Zt) ** (-0.545) * np.exp(-48.2 * Zt)

    Nu = np.where(active_mask, Nu_in, Nu_out)
    Nu = np.clip(Nu, 2.977, 50.0)          # bound the developing-flow singularity
    alpha = Nu * lam / d_h

    beta = np.zeros((n, N_SP))
    Sc = mu[:, None] / (rho[:, None] * D)
    Zs = np.maximum(z_local_active[:, None] / (d_h * Re[:, None] * Sc), 1e-12)
    Sh = 2.977 + 8.827 * (1000.0 * Zs) ** (-0.545) * np.exp(-48.2 * Zs)
    Sh = np.clip(Sh, 2.977, 50.0)          # bound the developing-flow singularity
    beta_active = Sh * D / d_h
    beta[active_mask] = beta_active[active_mask]
    return alpha, beta, Re, Pr


# ==========================================================================
#  KINETICS  (5 net LHHW reaction rates, per kg catalyst)
# ==========================================================================
def arrhenius(k873, Ea, T):
    return k873 * np.exp(-Ea / R_GAS * (1.0 / T - 1.0 / 873.0))


def reaction_rates(p_atm, T):
    """
    Net molar reaction rates [mol/(kg_cat s)], shape (n,5).
    p_atm: (n,6) partial pressures [atm]; T: (n,) catalyst temperature [K].
    """
    pCH4 = p_atm[:, IDX["CH4"]]
    pO2  = p_atm[:, IDX["O2"]]
    pCO2 = p_atm[:, IDX["CO2"]]
    pCO  = p_atm[:, IDX["CO"]]
    pH2O = p_atm[:, IDX["H2O"]]
    pH2  = p_atm[:, IDX["H2"]]

    # depletion / stabilisation factors
    def sig(p):
        return p / (p + STAB_FACTOR)

    k = np.empty((T.size, 6))
    for j in range(6):
        k[:, j] = arrhenius(K_873[j], E_A[j], T)
    kT, kSR, kW, kRW, kH, kCO = (k[:, j] for j in range(6))

    kads_H2O = arrhenius(K_ADS[IDX["H2O"]], H_ADS[IDX["H2O"]], T)
    kads_CO  = arrhenius(K_ADS[IDX["CO"]],  H_ADS[IDX["CO"]],  T)
    kads_O2  = arrhenius(K_ADS[IDX["O2"]],  H_ADS[IDX["O2"]],  T)

    Keq = K_eq(T)
    floor = 1e-30

    # 1) total oxidation (irreversible)
    r_TOX = sig(pO2) * kT * pCH4 / (1.0 + kads_H2O * pH2O)

    # 2) steam reforming (reversible)
    Q_SR = pH2**3 * pCO / np.maximum(pCH4 * pH2O, floor)
    r_SR = (sig(pH2O) * kSR * pCH4 * (1.0 - Q_SR / np.maximum(Keq[:, 1], floor))
            / (1.0 + kads_CO * pCO + kads_O2 * pO2))

    # 3) water-gas shift (reversible, smooth forward/reverse switch)
    Q_WGS = pCO2 * pH2 / np.maximum(pCO * pH2O, floor)
    A = np.log(np.maximum(Keq[:, 2], floor)) - np.log(np.maximum(Q_WGS, floor))
    driveF = np.maximum(np.tanh(0.25 * A), 0.0)
    driveB = np.maximum(-np.tanh(0.25 * A), 0.0)
    r_WGS = (sig(pCO) * kW * pH2O / (1.0 + kads_H2O * pH2O)**2 * driveF
             - sig(pH2) * kRW * pCO2 * driveB)

    # 4) hydrogen oxidation
    r_HOX = sig(pO2) * kH * pH2
    # 5) CO oxidation
    r_COOX = sig(pO2) * kCO * pCO

    return np.column_stack([r_TOX, r_SR, r_WGS, r_HOX, r_COOX])


# ==========================================================================
#  INTERNAL DIFFUSION (slab Thiele modulus, Knudsen regime)
# ==========================================================================
def _knudsen_Deff(T, M_i, cat):
    v = np.sqrt(8.0 * R_GAS * T / (np.pi * M_i))          # mean molecular speed
    D_macro = (2.0/3.0) * cat.d_macro * v
    D_micro = (2.0/3.0) * cat.d_micro * v
    return (cat.eps_macro**2 * D_macro
            + cat.eps_micro**2 * (1 + 3*cat.eps_macro) / (1 - cat.eps_macro) * D_micro)


def effectiveness(T_s, k_reac, cat):
    """
    First-order slab effectiveness factors for the O2/CH4 limited reactions.
    Returns dict with eta for TOX, SR, HOX, COOX (n,).
    """
    D_CH4 = _knudsen_Deff(T_s, MW[IDX["CH4"]], cat)
    D_O2 = _knudsen_Deff(T_s, MW[IDX["O2"]], cat)

    def eta(k_j, D):
        k_eff = k_j * cat.rho_cat * R_GAS * T_s / P_ATM        # 1/s
        phi = cat.L_wc * np.sqrt(np.maximum(k_eff, 1e-30) / D)
        phi = np.maximum(phi, 1e-9)
        return np.tanh(phi) / phi

    return dict(TOX=eta(k_reac[:, 0], D_O2),
                SR=eta(k_reac[:, 1], D_CH4),
                HOX=eta(k_reac[:, 4], D_O2),
                COOX=eta(k_reac[:, 5], D_O2))
