"""
Publication-style visualisations of the CPO reactor solution.

Reads results/cpo_solution.npz (written by run.py) and produces a set of
figures in figures/:

    fig1_temperature_axial.png     gas & solid axial T (developed + evolution)
    fig2_species_axial.png         bulk mole fractions along the reactor
    fig3_bulk_vs_wall.png          bulk vs catalyst-surface composition
    fig4_ignition_spacetime.png    solid-temperature space-time map (light-off)
    fig5_lightoff_curves.png       transient hot-spot T and conversions
    fig6_conversion_selectivity.png axial conversion, selectivity, H2/CO
    fig7_dashboard.png             combined multi-panel summary

Usage:
    python visualize.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from cpo.params import MW, IDX

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# ---- house style ---------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 190,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
})

SP = ["CH4", "O2", "CO2", "CO", "H2O", "H2"]
SP_LABEL = {"CH4": r"CH$_4$", "O2": r"O$_2$", "CO2": r"CO$_2$",
            "CO": "CO", "H2O": r"H$_2$O", "H2": r"H$_2$"}
SP_COLOR = {"CH4": "#222222", "O2": "#1f77b4", "CO2": "#7f7f7f",
            "CO": "#d62728", "H2O": "#17becf", "H2": "#2ca02c"}

GAS_C = "#e8801a"      # gas temperature
SOL_C = "#b2182b"      # solid temperature
CAT_FILL = "#cfe8cf"   # catalyst zone shading


def load():
    d = dict(np.load(os.path.join(RESULTS, "cpo_solution.npz"), allow_pickle=True))
    return d


def _shade_catalyst(ax, d, label=True):
    z0, z1 = float(d["z_cat_start"]) * 1e3, float(d["z_cat_end"]) * 1e3
    ax.axvspan(z0, z1, color=CAT_FILL, alpha=0.7, lw=0, zorder=0)
    if label:
        ax.text(0.5 * (z0 + z1), 0.97, "catalyst", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9,
                color="#2e7d32", weight="bold")


# ==========================================================================
def fig_temperature(d):
    z = d["z"] * 1e3
    t = d["t"]; Tg = d["Tg"]; Ts = d["Ts"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    _shade_catalyst(ax, d)
    # evolution (faded)
    idxs = np.linspace(0, len(t) - 1, 7).astype(int)[1:-1]
    for k in idxs:
        a = 0.18 + 0.5 * k / len(t)
        ax.plot(z, Ts[k], color=SOL_C, alpha=a * 0.5, lw=1.0)
    # developed state
    ax.plot(z, Ts[-1], color=SOL_C, lw=2.6, label=f"solid  $T_s$  (t={t[-1]:.0f} s)")
    ax.plot(z, Tg[-1], color=GAS_C, lw=2.6, label=f"gas    $T_g$  (t={t[-1]:.0f} s)")
    ax.plot(z, Ts[0], color="#888888", lw=1.3, ls="--", label="initial guess")
    zpk = float(d["peak_Ts_z"]) * 1e3
    ax.annotate(f"hot spot\n{float(d['peak_Ts']):.0f} K",
                xy=(zpk, float(d["peak_Ts"])), xytext=(zpk + 6, float(d["peak_Ts"]) - 30),
                fontsize=9, color=SOL_C,
                arrowprops=dict(arrowstyle="->", color=SOL_C, lw=1.2))
    ax.set_xlabel("axial position  z  [mm]")
    ax.set_ylabel("temperature  [K]")
    ax.set_title("Axial temperature profiles — catalytic partial oxidation of methane",
                 fontsize=12, weight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(z.min(), z.max())
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_temperature_axial.png"))
    plt.close(fig)


def fig_species(d):
    z = d["z"] * 1e3
    X = d["X"][-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    _shade_catalyst(ax, d)
    for s in SP:
        ax.plot(z, X[:, IDX[s]], color=SP_COLOR[s], label=SP_LABEL[s])
    ax.set_xlabel("axial position  z  [mm]")
    ax.set_ylabel("mole fraction  (dry, reacting species)")
    ax.set_title("Axial bulk-gas composition at the developed state",
                 fontsize=12, weight="bold")
    ax.legend(ncol=3, loc="upper right")
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_species_axial.png"))
    plt.close(fig)


def fig_bulk_wall(d):
    z = d["z"] * 1e3
    X = d["X"][-1]; Xw = d["x_wall"]
    active = d["active"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    _shade_catalyst(ax, d)
    for s in ["CH4", "O2"]:
        ax.plot(z, X[:, IDX[s]], color=SP_COLOR[s], lw=2.4,
                label=f"{SP_LABEL[s]} bulk")
        zz = z[active]; yw = Xw[active, IDX[s]]
        ax.plot(zz, yw, color=SP_COLOR[s], lw=1.8, ls="--", marker="o",
                ms=3, mfc="white", label=f"{SP_LABEL[s]} surface")
    ax.set_xlabel("axial position  z  [mm]")
    ax.set_ylabel("mole fraction")
    ax.set_title("Bulk vs. catalyst-surface composition (external mass transfer)",
                 fontsize=12, weight="bold")
    ax.legend(ncol=2, loc="upper right")
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_bulk_vs_wall.png"))
    plt.close(fig)


def fig_spacetime(d):
    z = d["z"] * 1e3; t = d["t"]; Ts = d["Ts"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    Zg, Tg_ = np.meshgrid(z, t)
    pcm = ax.pcolormesh(Zg, Tg_, Ts, shading="gouraud", cmap="inferno",
                        norm=Normalize(vmin=Ts.min(), vmax=Ts.max()))
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("solid temperature  $T_s$  [K]")
    z0, z1 = float(d["z_cat_start"]) * 1e3, float(d["z_cat_end"]) * 1e3
    ax.axvline(z0, color="white", lw=1.0, ls=":", alpha=0.8)
    ax.axvline(z1, color="white", lw=1.0, ls=":", alpha=0.8)
    ax.set_xlabel("axial position  z  [mm]")
    ax.set_ylabel("time  t  [s]")
    ax.set_title("Light-off transient: solid-temperature space-time map",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_ignition_spacetime.png"))
    plt.close(fig)


def fig_lightoff(d):
    t = d["t"]; Ts = d["Ts"]
    maxTs = Ts.max(axis=1)
    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    l1, = ax1.plot(t, maxTs, color=SOL_C, label="peak solid T")
    ax1.set_xlabel("time  t  [s]")
    ax1.set_ylabel("peak solid temperature  [K]", color=SOL_C)
    ax1.tick_params(axis="y", colors=SOL_C)
    ax2 = ax1.twinx()
    ax2.grid(False)
    l2, = ax2.plot(t, d["conv_CH4"] * 100, color="#222222", label="CH$_4$ conversion")
    l3, = ax2.plot(t, d["conv_O2"] * 100, color="#1f77b4", ls="--", label="O$_2$ conversion")
    ax2.set_ylabel("conversion  [%]")
    ax2.set_ylim(0, 105)
    ax1.set_title("Ignition / light-off dynamics", fontsize=12, weight="bold")
    ax1.legend(handles=[l1, l2, l3], loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_lightoff_curves.png"))
    plt.close(fig)


def _flux(W):
    """molar flux of each species (proportional to w_i / M_i)."""
    return W / MW[:6][None, :]


def fig_conv_sel(d):
    z = d["z"] * 1e3
    W = d["W"][-1]
    F = _flux(W)
    Fin = (d["W"][-1][0] / MW[:6])  # inlet (first node ~ feed)
    # use the prescribed feed as reference (first inert node equals feed)
    w_in = d["W"][-1][0]
    conv_CH4 = 1.0 - W[:, IDX["CH4"]] / w_in[IDX["CH4"]]
    # carbon-based CO selectivity, H-based H2 selectivity
    f = _flux(W)
    sel_CO = f[:, IDX["CO"]] / np.maximum(f[:, IDX["CO"]] + f[:, IDX["CO2"]], 1e-12)
    sel_H2 = f[:, IDX["H2"]] / np.maximum(f[:, IDX["H2"]] + f[:, IDX["H2O"]], 1e-12)
    ratio = f[:, IDX["H2"]] / np.maximum(f[:, IDX["CO"]], 1e-12)
    # selectivity/ratio are meaningless before products form -> mask the inert inlet
    prod = f[:, IDX["CO"]] + f[:, IDX["CO2"]]
    mask = prod < 1e-3 * f[:, IDX["CH4"]].max()
    sel_CO = sel_CO.copy(); sel_H2 = sel_H2.copy(); ratio = ratio.copy()
    sel_CO[mask] = np.nan; sel_H2[mask] = np.nan; ratio[mask] = np.nan

    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))
    _shade_catalyst(ax1, d)
    ax1.plot(z, conv_CH4 * 100, color="#222222", label="CH$_4$ conversion")
    ax1.plot(z, sel_CO * 100, color=SP_COLOR["CO"], label="CO selectivity (C)")
    ax1.plot(z, sel_H2 * 100, color=SP_COLOR["H2"], label="H$_2$ selectivity (H)")
    ax1.set_xlabel("axial position  z  [mm]")
    ax1.set_ylabel("conversion / selectivity  [%]")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(z, ratio, color="#9467bd", ls="--", lw=1.8, label="H$_2$/CO")
    ax2.set_ylabel("H$_2$/CO molar ratio", color="#9467bd")
    ax2.tick_params(axis="y", colors="#9467bd")
    ax2.set_ylim(0, max(4.0, np.nanmax(ratio[np.isfinite(ratio)]) * 1.1))
    ax1.set_title("Conversion, selectivity and syngas ratio along the reactor",
                  fontsize=12, weight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_xlim(z.min(), z.max())
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_conversion_selectivity.png"))
    plt.close(fig)


def fig_dashboard(d):
    z = d["z"] * 1e3; t = d["t"]
    Tg = d["Tg"]; Ts = d["Ts"]; X = d["X"]
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 8.4))

    ax = axs[0, 0]
    _shade_catalyst(ax, d)
    ax.plot(z, Ts[-1], color=SOL_C, label="$T_s$ solid")
    ax.plot(z, Tg[-1], color=GAS_C, label="$T_g$ gas")
    ax.set_ylabel("temperature [K]"); ax.set_xlabel("z [mm]")
    ax.set_title("(a) Developed temperature profile", weight="bold", fontsize=11)
    ax.legend(loc="lower right"); ax.set_xlim(z.min(), z.max())

    ax = axs[0, 1]
    _shade_catalyst(ax, d)
    for s in SP:
        ax.plot(z, X[-1][:, IDX[s]], color=SP_COLOR[s], label=SP_LABEL[s])
    ax.set_ylabel("mole fraction"); ax.set_xlabel("z [mm]")
    ax.set_title("(b) Developed composition", weight="bold", fontsize=11)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_xlim(z.min(), z.max()); ax.set_ylim(0, None)

    ax = axs[1, 0]
    Zg, Tt = np.meshgrid(z, t)
    pcm = ax.pcolormesh(Zg, Tt, Ts, shading="gouraud", cmap="inferno")
    fig.colorbar(pcm, ax=ax, pad=0.02, label="$T_s$ [K]")
    ax.set_ylabel("time [s]"); ax.set_xlabel("z [mm]")
    ax.set_title("(c) Light-off (solid T space-time)", weight="bold", fontsize=11)

    ax = axs[1, 1]
    ax.plot(t, Ts.max(axis=1), color=SOL_C, label="peak $T_s$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("peak $T_s$ [K]", color=SOL_C)
    ax.tick_params(axis="y", colors=SOL_C)
    axb = ax.twinx(); axb.grid(False)
    axb.plot(t, d["conv_CH4"] * 100, color="#222222", label="CH$_4$ conv")
    axb.plot(t, d["conv_O2"] * 100, color="#1f77b4", ls="--", label="O$_2$ conv")
    axb.set_ylabel("conversion [%]"); axb.set_ylim(0, 105)
    ax.set_title("(d) Ignition dynamics", weight="bold", fontsize=11)
    ha, la = ax.get_legend_handles_labels()
    hb, lb = axb.get_legend_handles_labels()
    ax.legend(ha + hb, la + lb, loc="center right", fontsize=8)

    fig.suptitle("1-D heterogeneous CPO reactor — solution overview",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIGS, "fig7_dashboard.png"))
    plt.close(fig)


def main():
    d = load()
    fig_temperature(d)
    fig_species(d)
    fig_bulk_wall(d)
    fig_spacetime(d)
    fig_lightoff(d)
    fig_conv_sel(d)
    fig_dashboard(d)
    print("figures written to", FIGS)
    for f in sorted(os.listdir(FIGS)):
        print("  ", f)


if __name__ == "__main__":
    main()
