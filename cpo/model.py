"""
Method-of-lines spatial discretisation and time integration of the 1-D
heterogeneous CPO reactor model.

State per axial node j (8 unknowns):
    w[0..5]  reacting-species mass fractions in the bulk gas
    Tg       bulk gas temperature  [K]
    Ts       solid (catalyst) temperature [K]
Flattened as S.reshape(-1) with S of shape (n_nodes, 8).

The quasi-steady washcoat (surface) species balance is eliminated inside the
right-hand side by a damped, vectorised Newton iteration (warm-started across
calls).  Convection is first-order upwind; solid conduction is a 3-point
non-uniform central difference.  Integration uses SciPy's stiff BDF method with
a supplied Jacobian sparsity pattern.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse
from . import params as P
from . import physics as ph
from .params import N_SP, IDX, MW, RM


# ==========================================================================
#  GRID
# ==========================================================================
class Grid:
    def __init__(self, cfg: P.Config):
        g, n = cfg.geo, cfg.num
        # cell-centred nodes in each zone
        def centres(L0, L1, ncell):
            edges = np.linspace(L0, L1, ncell + 1)
            return 0.5 * (edges[:-1] + edges[1:])
        z_in = centres(0.0, g.L_inlet, n.n_inlet)
        z_ct = centres(g.L_inlet, g.z_cat_end, n.n_cat)
        z_ou = centres(g.z_cat_end, g.L_total, n.n_outlet)
        self.z = np.concatenate([z_in, z_ct, z_ou])
        self.n = self.z.size
        self.active = (self.z >= g.z_cat_start) & (self.z <= g.z_cat_end)
        self.act_idx = np.where(self.active)[0]
        # axial distance from catalyst entrance (for the Graetz correlation)
        self.z_local = np.maximum(self.z - g.z_cat_start, 1e-9)
        # upstream / downstream spacings
        self.dz_w = np.empty(self.n)         # j - (j-1)
        self.dz_w[1:] = self.z[1:] - self.z[:-1]
        self.dz_w[0] = self.z[1] - self.z[0]
        self.dz_e = np.empty(self.n)         # (j+1) - j
        self.dz_e[:-1] = self.z[1:] - self.z[:-1]
        self.dz_e[-1] = self.z[-1] - self.z[-2]


# ==========================================================================
#  RHS
# ==========================================================================
class CPOModel:
    def __init__(self, cfg: P.Config):
        self.cfg = cfg
        self.grid = Grid(cfg)
        self.G = cfg.mass_flux_G()
        self.eps = cfg.cat.eps_bed
        self.P_pa = np.full(self.grid.n, cfg.feed.P_in)
        self.w_in = cfg.feed.w_in.copy()
        self.Tg_in = cfg.feed.T_gas_in
        self.Ts_in = cfg.feed.T_solid_in
        self.n = self.grid.n

    # ---- state packing -------------------------------------------------
    def unpack(self, Y):
        S = Y.reshape(self.n, 8)
        return S[:, :6], S[:, 6], S[:, 7]

    def initial_state(self, ignite=True, T_peak=1100.0, width=0.004):
        """
        Build a consistent initial state.

        ignite=True : smooth *ignited* guess - gas and solid share a smooth
        temperature profile that is cold in the inert inlet and hot over the
        catalyst (no spatial discontinuity, gas==solid so the sub-millisecond
        gas/solid thermal layer is pre-equilibrated).  Time-marching then
        relaxes this guess onto the true ignited steady state.

        ignite=False : cold, uniform feed state (for cold-start experiments).
        """
        S = np.zeros((self.n, 8))
        S[:, :6] = self.w_in[None, :]
        if ignite:
            g = self.cfg.geo
            z = self.grid.z
            ramp = 0.5 * (1.0 + np.tanh((z - g.z_cat_start) / width))
            Tprof = self.Tg_in + (T_peak - self.Tg_in) * ramp
            S[:, 6] = Tprof
            S[:, 7] = Tprof
        else:
            S[:, 6] = self.Tg_in
            S[:, 7] = self.Ts_in
        return S.reshape(-1)

    # ---- catalytic reaction source (effective rates at the wall) -------
    def _reaction(self, w6_a, Ts_a):
        """
        Effective net molar production Rnet (n_act,6) [mol/(kg_cat s)] and heat
        release Q (n_act,) [W/kg_cat] on the active nodes, evaluated at the gas
        composition with internal-diffusion effectiveness factors.

        Justification for using the bulk composition: at these conditions the
        external mass-transfer Damkoehler number is small (a_v*beta ~ 1e3-5e3 /s
        >> volumetric reaction rate ~1e2 /s), so the wall and bulk compositions
        nearly coincide.  The (small) wall depletion is reconstructed separately
        by `estimate_wall` for diagnostics.
        """
        cat = self.cfg.cat
        na = w6_a.shape[0]
        p_P = self.P_pa[self.grid.act_idx]
        kreac = np.column_stack([ph.arrhenius(P.K_873[j], P.E_A[j], Ts_a) for j in range(6)])
        eta = ph.effectiveness(Ts_a, kreac, cat)
        ev = np.ones((na, 5))
        ev[:, 0] = eta["TOX"]; ev[:, 1] = eta["SR"]; ev[:, 3] = eta["HOX"]; ev[:, 4] = eta["COOX"]
        _, _, _, p_atm, _ = ph.composition(w6_a, Ts_a, p_P)
        r_eff = ph.reaction_rates(p_atm, Ts_a) * ev          # (na,5)
        Rnet = r_eff @ RM                                    # (na,6)
        Q = np.sum(-ph.dH_reactions(Ts_a) * r_eff, axis=1)   # W/kg_cat
        return Rnet, Q

    def estimate_wall(self, w6_a, Ts_a, rho_g_a, beta_a):
        """One linearised film-transfer correction giving wall mass fractions
        for diagnostics (bulk - reaction/transfer).  Not used by the integrator."""
        cat = self.cfg.cat
        beta_safe = np.where(beta_a > 1e-12, beta_a, 1e-12)
        cfac = (MW[None, :N_SP] * cat.rho_cat_eff) / (cat.a_v * rho_g_a[:, None] * beta_safe)
        Rnet, _ = self._reaction(w6_a, Ts_a)
        return np.clip(w6_a + cfac * Rnet, 0.0, None)

    # ---- quasi-steady washcoat (surface) species balance (diagnostic) --
    def _solve_wall(self, w_gas_a, Ts_a, rho_g_a, beta_a, tol=1e-12, max_iter=12):
        """
        Solve the quasi-steady washcoat balance
            F(w) = w - w_gas - cfac * Rnet(w, Ts) = 0,
            cfac = MW * rho_cat_eff / (a_v * rho_g * beta),
        on the active nodes by a vectorised Newton iteration (6x6 block per
        node, finite-difference Jacobian).  Started from w_gas so that rhs(t,Y)
        is a *pure* function - essential for the implicit integrator's Jacobian.
        Shapes: (n_act, 6).
        """
        cat = self.cfg.cat
        a_v, rce = cat.a_v, cat.rho_cat_eff
        na = w_gas_a.shape[0]
        p_atm_P = self.P_pa[self.grid.act_idx]
        beta_safe = np.where(beta_a > 1e-12, beta_a, 1e-12)
        cfac = (MW[None, :N_SP] * rce) / (a_v * rho_g_a[:, None] * beta_safe)

        # effectiveness factors (depend on Ts only -> computed once)
        kreac = np.column_stack([ph.arrhenius(P.K_873[j], P.E_A[j], Ts_a) for j in range(6)])
        eta = ph.effectiveness(Ts_a, kreac, cat)
        eta_vec = np.ones((na, 5))
        eta_vec[:, 0] = eta["TOX"]; eta_vec[:, 1] = eta["SR"]
        eta_vec[:, 3] = eta["HOX"]; eta_vec[:, 4] = eta["COOX"]

        def Rnet(w_wall):
            _, _, _, p_atm, _ = ph.composition(w_wall, Ts_a, p_atm_P)
            return (ph.reaction_rates(p_atm, Ts_a) * eta_vec) @ RM    # (na,6)

        def resid(w_wall):
            return w_wall - w_gas_a - cfac * Rnet(w_wall)

        w = np.clip(w_gas_a.copy(), 1e-14, None)
        F = resid(w)
        I6 = np.eye(6)[None]
        for _ in range(max_iter):
            if np.max(np.abs(F)) < tol:
                break
            # finite-difference 6x6 Jacobian (perturb each species)
            J = np.empty((na, 6, 6))
            for k in range(6):
                h = 1e-9 + 1e-7 * w[:, k]
                wp = w.copy(); wp[:, k] += h
                J[:, :, k] = (resid(wp) - F) / h[:, None]
            step = np.linalg.solve(J, F[:, :, None])[:, :, 0]
            # damped, positivity-preserving update
            w_new = np.clip(w - step, 1e-14, None)
            F_new = resid(w_new)
            grew = np.max(np.abs(F_new), axis=1) > np.max(np.abs(F), axis=1)
            if np.any(grew):                      # half-step where residual grew
                w_half = np.clip(w - 0.5 * step, 1e-14, None)
                w_new[grew] = w_half[grew]
                F_new = resid(w_new)
            w, F = w_new, F_new
        return w, eta_vec

    # ---- full right-hand side -----------------------------------------
    def rhs(self, t, Y):
        n, eps, G = self.n, self.eps, self.G
        grid = self.grid
        cat = self.cfg.cat
        w6, Tg, Ts = self.unpack(Y)
        w6 = np.clip(w6, 0.0, None)
        Tg = np.clip(Tg, 250.0, 4000.0)
        Ts = np.clip(Ts, 250.0, 4000.0)

        gp = ph.gas_properties(w6, Tg, self.P_pa)
        rho_g, cp_mass = gp['rho'], gp['cp_mass']
        alpha, beta, Re, Pr = ph.transfer_coefficients(
            gp, self.cfg.geo, G, eps, grid.z_local, grid.active)

        # --- catalytic reaction on active nodes (bulk composition) ---
        ai = grid.act_idx
        Rnet, Q = self._reaction(w6[ai], Ts[ai])                    # (na,6),(na,)

        dS = np.zeros((n, 8))

        # ---------- gas species (upwind convection + reaction source) ----------
        w_up = np.vstack([self.w_in[None, :], w6[:-1, :]])          # upstream values
        conv = -(G / (rho_g[:, None] * eps)) * (w6 - w_up) / grid.dz_w[:, None]
        dS[:, :6] = conv
        # heterogeneous reaction adds/removes species from the gas
        dS[ai, :6] += (MW[None, :N_SP] * cat.rho_cat_eff * Rnet) / (eps * rho_g[ai, None])

        # ---------- gas energy ----------
        Tg_up = np.concatenate([[self.Tg_in], Tg[:-1]])
        convT = -(G / (rho_g * eps)) * (Tg - Tg_up) / grid.dz_w
        htg = -(cat.a_v / eps) * alpha / (rho_g * cp_mass) * (Tg - Ts)
        dS[:, 6] = convT + htg

        # ---------- solid energy (heat exchange + conduction + reaction) ----------
        # effective axial conductivity (solid + radiative)
        k_rad = (16.0/3.0) * 1.12 * P.SIGMA_SB * (self.cfg.geo.d_h/2.0) * Ts**3
        k_ax = (1.0 - cat.xi) * cat.k_s + k_rad
        # 3-point non-uniform second derivative with BCs
        Tw = np.concatenate([[self.Ts_in], Ts[:-1]])               # west neighbour (inlet Dirichlet)
        Te = np.concatenate([Ts[1:], [Ts[-2]]])                    # east neighbour (outlet Neumann: reflect)
        hw, he = grid.dz_w, grid.dz_e
        d2T = 2.0 * ((Te - Ts) / he - (Ts - Tw) / hw) / (he + hw)
        Cs = cat.rho_bed * cat.cp_s                                 # volumetric heat capacity
        dTs = cat.a_v * alpha * (Tg - Ts) / Cs + k_ax * d2T / Cs
        dTs[ai] += cat.rho_cat_eff * Q / Cs
        dS[:, 7] = dTs

        return dS.reshape(-1)

    # ---- Jacobian sparsity pattern ------------------------------------
    def jac_sparsity(self):
        n = self.n
        S = sparse.lil_matrix((8 * n, 8 * n), dtype=bool)
        def idx(j, v):
            return j * 8 + v
        for j in range(n):
            for v in range(8):
                # intra-node coupling (reaction, transfer, properties)
                for vv in range(8):
                    S[idx(j, v), idx(j, vv)] = True
                # upwind convection -> depends on j-1
                if j > 0:
                    for vv in range(8):
                        S[idx(j, v), idx(j-1, vv)] = True
                # conduction (Ts) -> depends on j-1, j+1
                if j < n - 1:
                    S[idx(j, 7), idx(j+1, 7)] = True
        return sparse.csr_matrix(S)


# ==========================================================================
#  TIME INTEGRATION
# ==========================================================================
def run(cfg: P.Config | None = None, ignite=True, t_end=None, verbose=True):
    cfg = cfg or P.DEFAULT
    model = CPOModel(cfg)
    Y0 = model.initial_state(ignite=ignite)
    tend = t_end if t_end is not None else cfg.num.t_end
    t_eval = np.linspace(0.0, tend, cfg.num.n_save)

    atol = np.zeros((model.n, 8))
    atol[:, :6] = 1e-9
    atol[:, 6:] = 1e-3
    atol = atol.reshape(-1)

    if verbose:
        print(f"nodes={model.n}  active={model.grid.act_idx.size}  "
              f"G={model.G:.4f} kg/m2/s  t_end={tend}s")

    sol = solve_ivp(model.rhs, (0.0, tend), Y0, method="BDF",
                    t_eval=t_eval, jac_sparsity=model.jac_sparsity(),
                    rtol=1e-5, atol=atol, first_step=1e-8, max_step=tend/20.0)
    if verbose:
        print(f"  success={sol.success}  nfev={sol.nfev}  msg={sol.message}")
    return model, sol


def steady_state(model: CPOModel, Y_guess, tol=1e-6, max_iter=60, verbose=True):
    """
    Polish a transient snapshot onto the true steady state by a damped Newton
    iteration on rhs(0, Y) = 0, using the sparse finite-difference Jacobian
    (built from the model's sparsity pattern).  Variables are scaled
    (mass fractions O(1), temperatures /1000) for good conditioning.
    """
    from scipy.optimize._numdiff import approx_derivative
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu
    n = model.n
    Tsc = 1000.0
    scale = np.ones((n, 8)); scale[:, 6:] = Tsc; scale = scale.reshape(-1)
    pattern = model.jac_sparsity().astype(float)

    def res(Ys):
        return model.rhs(0.0, Ys * scale) / scale       # scaled residual

    x = Y_guess / scale
    F = res(x)
    nF = np.linalg.norm(F)
    for it in range(max_iter):
        if nF < tol:
            break
        J = approx_derivative(res, x, method="2-point", sparsity=pattern)
        try:
            lu = splu(csc_matrix(J))
            dx = lu.solve(-F)
        except Exception:
            break
        # backtracking line search
        lam = 1.0
        for _ in range(25):
            xn = x + lam * dx
            Fn = res(xn); nFn = np.linalg.norm(Fn)
            if nFn < nF or lam < 1e-4:
                break
            lam *= 0.5
        x, F, nF = xn, Fn, nFn
    Y = x * scale
    if verbose:
        print(f"  steady-state Newton: iters={it} |res|={nF:.2e} "
              f"({'converged' if nF < tol*50 else 'partial'})")
    return Y, nF < tol * 50


def postprocess(model: CPOModel, sol):
    """unpack the solution into convenient arrays + key engineering metrics."""
    n = model.n
    nt = sol.t.size
    W = np.empty((nt, n, 6)); Tg = np.empty((nt, n)); Ts = np.empty((nt, n))
    for it in range(nt):
        S = sol.y[:, it].reshape(n, 8)
        W[it] = S[:, :6]; Tg[it] = S[:, 6]; Ts[it] = S[:, 7]
    # convert bulk mass fractions to dry mole fractions for reporting
    out = dict(t=sol.t, z=model.grid.z, active=model.grid.active,
               W=W, Tg=Tg, Ts=Ts)
    # mole fractions over time
    X = np.empty_like(W)
    for it in range(nt):
        x, *_ = ph.composition(W[it], Tg[it], model.P_pa)
        X[it] = x[:, :6]
    out['X'] = X
    # CH4 / O2 conversion at the outlet over time (mass-fraction based)
    w_in = model.w_in
    out['conv_CH4'] = 1.0 - W[:, -1, IDX['CH4']] / w_in[IDX['CH4']]
    out['conv_O2'] = 1.0 - W[:, -1, IDX['O2']] / w_in[IDX['O2']]
    # H2/CO dry molar ratio at outlet (final time)
    xf = X[-1, -1]
    out['H2_CO_ratio'] = xf[IDX['H2']] / max(xf[IDX['CO']], 1e-12)
    return out
