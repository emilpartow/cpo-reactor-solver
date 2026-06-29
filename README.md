# CPO Reactor Solver — 1-D Heterogeneous Transient Model

A clean, working Python solver for the **catalytic partial oxidation (CPO) of
methane** in a 1-D, two-phase (gas + solid) monolith reactor, reproducing the
model of *Beretta et al. (2009)*. This repository was written to replace the
earlier solver attempts that failed to converge.

It integrates the full transient PDE system from a smooth ignited initial
guess to a developed (near-steady) state and produces a set of
publication-style figures of the solution.

```
python run.py          # solve, save results/cpo_solution.npz   (~30 s)
python visualize.py    # render the 7 figures into figures/
```

---

## 1. The problem solved

State fields along the axial coordinate `z ∈ [0, L]` and time `t`: six gas
mass fractions `ω_i` (CH₄, O₂, CO₂, CO, H₂O, H₂), the gas temperature `T_g`,
and the solid/catalyst temperature `T_s`.

**Gas phase (bulk).** Advection + interphase exchange:

```
∂ω_i/∂t = −(G/ρ_g ε)·∂ω_i/∂z + (1/ε ρ_g)·MW_i·ρ_cat·Σ_j ν_ij r_j      (i = 1..6)
∂T_g/∂t = −(G/ρ_g ε)·∂T_g/∂z − (a_v/ε)·(h/ρ_g ĉ_g)·(T_g − T_s)
```

**Solid phase (catalyst).** Interphase heat + axial conduction + reaction heat:

```
∂T_s/∂t = a_v·h·(T_g−T_s)/(ρ_s ĉ_s) + k_ax,eff·∂²T_s/∂z²/(ρ_s ĉ_s)
                                     + ρ_cat·Σ_j (−ΔH_j) r_j /(ρ_s ĉ_s)
k_ax,eff = (1−ξ)·k_s + (16/3)(1.12)·σ·(d_h/2)·T_s³     (conduction + radiation)
```

**Reaction scheme (5 net LHHW rates).**

```
TOX :  CH₄ + 2 O₂ → CO₂ + 2 H₂O
SR  :  CH₄ + H₂O ⇌ CO + 3 H₂
WGS :  CO + H₂O ⇌ CO₂ + H₂
HOX :  H₂ + ½ O₂ → H₂O
COOX:  CO + ½ O₂ → CO₂
```

**Boundary conditions.** Dirichlet feed at the inlet (`ω_i,0`, `T_g,0`, `T_s,0`);
zero conductive flux (`∂T_s/∂z = 0`) at the outlet.

Closures: NASA-7 thermodynamics (`cp, H, S, ΔH_R, K_eq`), Chung viscosity,
Eucken conductivity, Fuller diffusion, Wilke mixing, Graetz developing-flow
Sherwood/Nusselt correlations, and slab Thiele effectiveness factors with
Knudsen diffusion. All parameters are taken from `Reactor_Properties.xlsx`.

---

## 2. Numerical method

* **Method of lines.** First-order **upwind** for the advective `∂/∂z`,
  3-point non-uniform central differences for the conductive `∂²/∂z²`, on a
  graded mesh refined over the catalyst.
* **Time integration.** SciPy `solve_ivp` with the stiff **BDF** method and a
  supplied **Jacobian sparsity pattern**; per-variable absolute tolerances.
* **Steady-state polish.** Optional damped sparse-Newton on `rhs = 0`
  (`model.steady_state`).

### Why the previous attempts failed — and what fixes it

1. **Non-deterministic right-hand side.** A warm-start cache in the inner
   washcoat solver made `rhs(t, Y)` history-dependent, which corrupts the
   implicit integrator's finite-difference Jacobian and collapses the step
   size. The RHS here is a **pure function**.
2. **Stiff algebraic wall balance.** The gas↔surface film balance is a stiff
   nonlinear algebraic system. Here the external-transfer Damköhler number is
   small (`a_v·β ≈ 10³–10⁴ s⁻¹ ≫` reaction `≈ 10² s⁻¹`), so reactions are
   evaluated at the bulk composition and the (small) surface depletion is
   reconstructed only for plotting (`fig3`). This removes the stiff inner solve.
3. **Start-up transient.** A smooth ignited initial profile with `T_g = T_s`
   pre-equilibrates the sub-millisecond gas/solid thermal layer; the
   developing-flow transfer correlations are bounded to remove the
   catalyst-entrance singularity.

---

## 3. Results (developed state, default case)

| quantity | value |
|---|---|
| peak solid temperature | ≈ 1706 K, at the catalyst inlet (z ≈ 16 mm) |
| O₂ conversion | 100 % |
| CH₄ conversion | ≈ 51 % |
| outlet H₂/CO | ≈ 2.6 |

The solution reproduces the canonical CPO signature: a **sharp oxidation
hot-spot at the catalyst inlet** (O₂ fully consumed, CO₂/H₂O formed),
followed by an endothermic **reforming zone** producing syngas (CO + H₂).
Element (C/H/O) balances close to machine precision.

Figures (`figures/`):

| file | content |
|---|---|
| `fig1_temperature_axial.png` | gas & solid axial temperature + light-off evolution |
| `fig2_species_axial.png` | bulk-gas composition along the reactor |
| `fig3_bulk_vs_wall.png` | bulk vs. catalyst-surface composition |
| `fig4_ignition_spacetime.png` | solid-temperature space-time map (light-off) |
| `fig5_lightoff_curves.png` | transient hot-spot temperature and conversions |
| `fig6_conversion_selectivity.png` | axial conversion, selectivity, H₂/CO ratio |
| `fig7_dashboard.png` | combined four-panel overview |

---

## 4. Repository layout

```
cpo_solver/
├── cpo/
│   ├── params.py     parameters from the spreadsheets (SI units)
│   ├── physics.py    thermodynamics, transport, kinetics, effectiveness
│   └── model.py      grid, method-of-lines RHS, integrator, steady-state
├── run.py            production run -> results/cpo_solution.npz
├── visualize.py      figures -> figures/
├── results/          saved solution (.npz)
├── figures/          generated PNGs
└── requirements.txt
```

## 5. Notes and modelling choices

* Temperatures in K, pressures in Pa internally; partial pressures converted to
  atm for the kinetics (as the rate constants were fitted).
* N₂ molar mass corrected to 28.013 g/mol (the spreadsheet stored 14.007).
* The default feed is mass fractions CH₄ 0.174 / O₂ 0.19 / N₂ balance
  (molar O/C ≈ 1.1), 630 K, 1 bar, 10 NL/min over 236 channels — giving
  `G ≈ 0.732 kg/m²/s`.
* Geometry: 15 mm inert inlet, 20 mm catalyst, 15 mm inert outlet.
* The reported peak solid temperature reflects the sharp, adiabatic inlet
  oxidation zone of this slightly O-rich feed; change the feed/geometry in
  `cpo/params.py` to explore other operating points.
