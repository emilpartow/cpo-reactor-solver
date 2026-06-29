"""
Production run of the 1-D heterogeneous CPO reactor model.

Integrates the full transient from a smooth ignited initial guess to a
developed (near-steady) state, then writes the solution and a set of
engineering metrics to results/cpo_solution.npz for the plotting script.

Usage:
    python run.py
"""
import os
import time
import numpy as np

from cpo import params as P
from cpo import model as M
from cpo import physics as ph
from cpo.params import IDX

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def main(t_end=200.0, n_inlet=16, n_cat=64, n_outlet=16, n_save=80):
    cfg = P.DEFAULT
    cfg.num.n_inlet = n_inlet
    cfg.num.n_cat = n_cat
    cfg.num.n_outlet = n_outlet
    cfg.num.n_save = n_save
    cfg.num.t_end = t_end

    t0 = time.time()
    model, sol = M.run(cfg, ignite=True, t_end=t_end)
    out = M.postprocess(model, sol)
    wall = time.time() - t0

    # developed-state wall (surface) mole fractions for the bulk-vs-wall figure
    W_dev = out["W"][-1]
    Tg_dev = out["Tg"][-1]
    Ts_dev = out["Ts"][-1]
    gp = ph.gas_properties(W_dev, Tg_dev, model.P_pa)
    alpha, beta, Re, Pr = ph.transfer_coefficients(
        gp, cfg.geo, model.G, model.eps, model.grid.z_local, model.grid.active)
    ai = model.grid.act_idx
    w_wall = model.estimate_wall(W_dev[ai], Ts_dev[ai], gp["rho"][ai], beta[ai])
    x_wall_full = np.full_like(W_dev, np.nan)
    xw, *_ = ph.composition(w_wall, Ts_dev[ai], model.P_pa[ai])
    x_wall_full[ai] = xw[:, :6]

    meta = dict(
        G=model.G, eps=model.eps,
        z_cat_start=cfg.geo.z_cat_start, z_cat_end=cfg.geo.z_cat_end,
        L_total=cfg.geo.L_total, T_feed=cfg.feed.T_gas_in,
        species=np.array(P.SPECIES[:6]),
        peak_Ts=float(Ts_dev.max()),
        peak_Ts_z=float(model.grid.z[np.argmax(Ts_dev)]),
        conv_CH4_dev=float(out["conv_CH4"][-1]),
        conv_O2_dev=float(out["conv_O2"][-1]),
        H2_CO=float(out["H2_CO_ratio"]),
        wall_time=wall,
    )

    np.savez(
        os.path.join(RESULTS, "cpo_solution.npz"),
        t=out["t"], z=model.grid.z, active=model.grid.active,
        W=out["W"], Tg=out["Tg"], Ts=out["Ts"], X=out["X"],
        conv_CH4=out["conv_CH4"], conv_O2=out["conv_O2"],
        Re=Re, Pr=Pr, alpha=alpha, x_wall=x_wall_full,
        **meta,
    )

    print("=" * 64)
    print(f" Production run finished in {wall:.1f} s  (success={sol.success})")
    print(f"   grid: {model.n} nodes ({ai.size} catalytic)   t_end={t_end:.0f}s")
    print(f"   developed state:")
    print(f"     peak solid T  = {meta['peak_Ts']:.0f} K  at z = {meta['peak_Ts_z']*1e3:.1f} mm")
    print(f"     outlet gas T  = {Tg_dev[-1]:.0f} K")
    print(f"     CH4 conversion = {meta['conv_CH4_dev']*100:.1f} %")
    print(f"     O2  conversion = {meta['conv_O2_dev']*100:.1f} %")
    print(f"     H2/CO ratio    = {meta['H2_CO']:.2f}")
    print(f"   saved -> results/cpo_solution.npz")
    print("=" * 64)


if __name__ == "__main__":
    main()
