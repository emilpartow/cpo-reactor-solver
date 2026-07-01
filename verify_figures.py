"""
Figures that demonstrate the ALGEBRAIC CONSTRAINT of the coupled DAE is
satisfied, i.e. the catalyst-surface species balance

    0 = a_v * rho_g * K_mat,i * (w_i - w_i,wall)          (film transfer)
        + ( sum_j nu_ij r_j^eff ) * MW_i * rho_cat * xi   (reaction)

is met at the solution.  Produces figures/fig8_constraint_verification.png:

  (a) transfer term vs. (-reaction term) along the reactor  -> they overlie
  (b) dimensionless constraint residual per species along the reactor (~1e-7)
  (c) tau_wc -> 0 : residual falls first-order while the physics stays fixed
  (d) constraint residual vs. time (stays ~1e-7 after start-up)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import visualize as viz            # reuse house style + colours
from visualize import SP, SP_LABEL, SP_COLOR, _shade_catalyst, FIGS
from cpo import params as P
from cpo import model as M
from cpo import physics as ph
from cpo.params import MW, N_SP, IDX, RM
import verify as V


def terms(model, W, Tg, Ts, Wwall):
    """per active node, per species: transfer term, reaction term, residual, scale."""
    ai = model.grid.act_idx
    gp = ph.gas_properties(W, Tg, model.P_pa)
    _, beta, _, _ = ph.transfer_coefficients(
        gp, model.cfg.geo, model.G, model.eps, model.grid.z_local, model.grid.active)
    rho_a = gp['rho'][ai]; beta_a = np.maximum(beta[ai], 1e-12)
    transfer = model.cfg.cat.a_v * rho_a[:, None] * beta_a * (W[ai] - Wwall)
    Rnet, _ = model._reaction_wall(Wwall, Ts[ai])
    reaction = MW[None, :N_SP] * model.cfg.cat.rho_cat_eff * Rnet
    resid = transfer + reaction
    scale = model.cfg.cat.a_v * rho_a[:, None] * beta_a
    return transfer, reaction, resid, scale


def main():
    print("running developed-state case ...", flush=True)
    model, sol, out = V.run_case(tau_wc=5e-4, t_end=150.0, n=(16, 60, 16))
    W, Tg, Ts, Ww = out['W'], out['Tg'], out['Ts'], out['Wwall']
    geo = {"z_cat_start": model.cfg.geo.z_cat_start, "z_cat_end": model.cfg.geo.z_cat_end}
    za = model.grid.z[model.grid.act_idx] * 1e3
    transfer, reaction, resid, scale = terms(model, W[-1], Tg[-1], Ts[-1], Ww[-1])

    # residual vs time
    tres = np.array([np.max(np.abs(V.constraint_residual(model, W[i], Tg[i], Ts[i], Ww[i])[0]))
                     for i in range(len(sol.t))])

    # tau_wc sweep
    print("running tau_wc sweep ...", flush=True)
    taus = np.array([2e-3, 5e-4, 1e-4, 2e-5])
    rr, pk, cv = [], [], []
    for tau in taus:
        m2, s2, o2 = V.run_case(tau_wc=tau, t_end=90.0, n=(10, 40, 10))
        r, *_ = V.constraint_residual(m2, o2['W'][-1], o2['Tg'][-1], o2['Ts'][-1], o2['Wwall'][-1])
        rr.append(r); pk.append(o2['Ts'][-1].max()); cv.append(o2['conv_CH4'][-1]*100)
    rr, pk, cv = map(np.array, (rr, pk, cv))

    # ---------------- plot ----------------
    fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.0))

    # (a) term balance for O2 and CH4
    ax = axs[0, 0]
    _shade_catalyst(ax, geo, loc="top")
    for s in ("O2", "CH4"):
        i = IDX[s]
        ax.plot(za, transfer[:, i], color=SP_COLOR[s], lw=2.3,
                label=f"{SP_LABEL[s]}: film transfer")
        ax.plot(za, -reaction[:, i], color=SP_COLOR[s], lw=0, marker="o", ms=4,
                mfc="white", mec=SP_COLOR[s], label=f"{SP_LABEL[s]}: $-$reaction")
    ax.set_xlabel("z [mm]"); ax.set_ylabel(r"rate  [kg m$^{-3}$ s$^{-1}$]")
    ax.set_title("(a) Constraint terms balance:  transfer $= -$reaction",
                 weight="bold", fontsize=11)
    ax.legend(fontsize=8.5, loc="best")

    # (b) residual per species along reactor
    ax = axs[0, 1]
    _shade_catalyst(ax, geo, loc="top")
    for s in SP:
        ax.semilogy(za, np.abs(resid[:, IDX[s]]) / scale[:, IDX[s]] + 1e-20,
                    color=SP_COLOR[s], label=SP_LABEL[s], lw=1.6)
    ax.set_xlabel("z [mm]"); ax.set_ylabel(r"$|g_i|\,/\,(a_v\rho_g\beta_i)$")
    ax.set_title("(b) Dimensionless constraint residual along reactor",
                 weight="bold", fontsize=11)
    ax.set_ylim(1e-12, 1e-2); ax.legend(ncol=3, fontsize=8, loc="upper right")

    # (c) tau_wc -> 0
    ax = axs[1, 0]
    ax.loglog(taus, rr, "o-", color="#b2182b", lw=2, ms=7, label="constraint residual")
    ax.loglog(taus, rr[-1]*taus/taus[-1], "--", color="#888888", lw=1.4,
              label=r"$\propto \tau_{wc}$ (first order)")
    ax.set_xlabel(r"$\tau_{wc}$  [s]"); ax.set_ylabel("constraint residual")
    ax.set_title(r"(c) Pseudo-transient limit  $\tau_{wc}\!\to\!0$  = true DAE",
                 weight="bold", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    axb = ax.twinx(); axb.grid(False)
    axb.semilogx(taus, pk, "s:", color="#1f77b4", ms=6, label="peak $T_s$")
    axb.set_ylabel("peak $T_s$ [K]", color="#1f77b4")
    axb.tick_params(axis="y", colors="#1f77b4")
    axb.set_ylim(pk.mean()-60, pk.mean()+60)
    axb.text(0.97, 0.10, "physics $\\tau_{wc}$-independent\n(peak $T_s$, conv. flat)",
             transform=axb.transAxes, ha="right", fontsize=8.5, color="#1f77b4")

    # (d) residual vs time
    ax = axs[1, 1]
    ax.semilogy(sol.t, tres, color="#b2182b", lw=2)
    ax.set_xlabel("time  t  [s]"); ax.set_ylabel("max constraint residual")
    ax.set_title("(d) Constraint residual during the transient",
                 weight="bold", fontsize=11)
    ax.set_ylim(1e-9, 1e-2)
    ax.text(0.5, 0.9, "initial guess off-manifold,\nthen held at $\\sim10^{-7}$",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    fig.suptitle("Verification of the algebraic surface constraint  "
                 r"$0 = a_v\rho_g K_{mat,i}(\omega_i-\omega_{i,wall}) + "
                 r"(\sum_j \nu_{ij} r_j^{eff})\,MW_i\,\rho_{cat}\xi$",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.30, wspace=0.24)
    fig.savefig(os.path.join(FIGS, "fig8_constraint_verification.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/fig8_constraint_verification.png")


if __name__ == "__main__":
    main()
