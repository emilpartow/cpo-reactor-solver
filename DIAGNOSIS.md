# Why the original MATLAB solver failed — and what was wrong with the data

This note documents the problems found in the previous MATLAB implementation
(`BzzDae-like/`) and in the spreadsheet data, and explains how the Python
solver in this repository avoids each of them.

The findings were obtained by reading the MATLAB sources and the
`Reactor_Properties.xlsx` / `Modelling_Properties.xlsx` files. Line numbers
refer to the versions in the project as of this analysis. A few items are
"strong suspicions" rather than certainties (noted as such) because some unit
groupings could in principle hide a compensating convention.

---

## TL;DR

| # | Where | Problem | Severity |
|---|-------|---------|----------|
| A1 | model structure | Quasi-steady washcoat ⇒ stiff index-1 **DAE**; needed `ode15i` + a **4 197-line** restart/homotopy wrapper that kept failing | structural |
| A2 | `odefun_physicalIt.m:4` | **Non-pure RHS**: `persistent w_Surf_cache/w_Surf_history` warm-start corrupts the implicit Jacobian and is non-reproducible | critical |
| A3 | inner wall solve | Fixed-point iteration (convergence judged on CH₄ only, 80-iter cap, prints "no convergence"); later an exp-log algebraic variable with bound "stoppers" | high |
| B1 | `odefun_physicalIt.m:268-283` | **Dimensional error** in the gas-energy balance: `M_mix*c_PGas/1000` is not a mass-specific cp; the term is mis-scaled and the outlet node is *inconsistent* with the others | high |
| B2 | `odefun_physicalIt.m:150` | Effectiveness rate constants use **T_gas**, while the reaction rates use **T_solid** | medium |
| B3 | several | g_cat vs kg_cat, Knudsen molar-mass units, simplified Wilke rule | medium |
| **D1** | `Species` sheet | **N₂ molar mass = 14.0067 g/mol** (atomic N!). Should be 28.0134. N₂ is ~60 % of the gas, so the whole mixture state is wrong | critical |
| D2 | the two `.xlsx` | Same parameter defined twice with **different values** (stability factor 1e-3 vs 1e-6; MinConc 1e-8 vs 1e-12) | medium |

---

## A. The integration was the core struggle

### A1 — It is a stiff DAE, attacked with very heavy machinery

The model assumes the catalyst washcoat is in pseudo-steady state, so the
surface-species balance has **no time derivative**. The full system is therefore
a differential-algebraic system (DAE), not an ODE. The project went through two
formulations:

* **`odefun_physicalIt.m`** — `ode15s` with the algebraic surface balance solved
  by an *inner* fixed-point iteration on every call.
* **`odefun_ode15i.m`** — a fully implicit residual for **`ode15i`**, with the
  wall mass fractions promoted to algebraic variables `q_Surf`
  (`w_Surf = WallFloor·exp(QScale·q)`), solved simultaneously with the
  differential variables.

The second formulation is the more correct one, but making `ode15i` converge
required `ode15i_restart_wrapper.m` — **4 197 lines** of backward-Euler rescue,
homotopy ramps, "front mode", consistent-IC projection and step-size
babysitting. When a solver needs that much scaffolding, the underlying setup is
fighting the integrator. The cause is the combination below (A2, A3, B1).

### A2 — The right-hand side was not a pure function (critical)

`odefun_physicalIt.m` line 4:

```matlab
persistent w_Surf_cache w_Surf_history iter_counter;
```

The previous surface solution is cached and reused as the warm start, and
`w_Surf_history{iter_counter}` grows on **every** call. Implicit integrators
(`ode15s`, `ode15i`) build their Jacobian by finite-differencing the RHS:
they call it many times at perturbed states `Y ± δ`. With a mutating cache,
each of those probe calls changes the hidden state, so:

* the numerical Jacobian is computed from an **inconsistent** function → Newton
  iterations of the integrator do not converge → the step size collapses
  ("required step size below floating-point spacing");
* results are **not reproducible** (depend on call history);
* `w_Surf_history` grows without bound (memory).

This is not speculation — the authors rediscovered it themselves. In
`odefun_ode15i.m` the molar masses carry the comment *"Do not keep this as
persistent state. ode15i may call this residual many times … persistent …
would silently corrupt later runs"* (≈ line 327) and *"This produced a
persistent residual oscillation (not converging …)"* (≈ line 372).

> **This solver:** `rhs(t, Y)` is a strictly pure function. This single change
> is what turned a never-finishing integration into one that completes in ~30 s.

### A3 — The inner wall solve was fragile

The fixed-point loop (`odefun_physicalIt.m` ≈ 133–227) uses adaptive
relaxation, **judges convergence on the CH₄ surface concentration only**
(≈ 209–211), caps at 80 iterations and prints *"WARNING: no convergence"*. A
non-converged inner solve makes the outer RHS noisy, which again defeats the
implicit integrator. The `ode15i` rewrite replaced it with bounded exp-log
algebraic variables — more robust, but it is what forced the elaborate restart
logic.

> **This solver:** at these conditions the external mass-transfer Damköhler
> number is small (`a_v·β ≈ 10³–10⁴ s⁻¹` ≫ reaction `≈ 10² s⁻¹`), so the wall
> and bulk compositions nearly coincide. Reactions are evaluated at the bulk
> composition (conservative, smooth, no inner nonlinear solve); the small wall
> depletion is reconstructed afterwards only for plotting. The stiff algebraic
> sub-problem disappears.

---

## B. Concrete coding errors

### B1 — Dimensional error in the gas energy balance (high)

`UpdateHeatCapacities.m` returns a **molar** heat capacity
(`Cp = 4.184·1.987·(NASA poly) = R·(c_p/R)`, units J·mol⁻¹·K⁻¹), and
`c_PGas = Σ xᵢ Cpᵢ` is therefore the molar mixture cp.

The gas energy balance (`odefun_physicalIt.m` lines 268/273/278) divides the
heat-transfer term by

```matlab
Rho_G * (M_mix * c_PGas / 1000)
```

To convert a molar cp to a **mass-specific** cp you must *divide* by the molar
mass: `ĉ_p = c_PGas / (M_mix·10⁻³)`. The code instead *multiplies* by
`M_mix·10⁻³`. The volumetric heat capacity `ρ_g·ĉ_p` is thus mis-scaled by a
factor of order `(1000/M_mix)² ≈ 1.6·10³`, which makes the gas equilibrate with
the solid far too fast — extra stiffness on top of A2/A3.

Worse, the **outlet** node (line 283) divides by `c_PGas(j)` alone — dropping
the `M_mix/1000` factor that the other three branches keep. So even internally
the four gas-energy branches are not consistent with each other.

> **This solver:** `cp_mass = cp_molar / M_mix` is computed once in
> `physics.gas_properties`, with `M_mix` in kg/mol, and used everywhere.

### B2 — Reaction temperature inconsistency (medium)

For the effectiveness factor, the rate constants are evaluated at the **gas**
temperature (`odefun_physicalIt.m:150`, `k_Reac … exp(-E_A/R(1/T_Gas - 1/873))`),
while the actual reaction rates (`CalculateReactionRates`) are evaluated at the
**solid/surface** temperature. A catalytic rate should use the catalyst
temperature consistently.

> **This solver:** all catalytic kinetics (rates *and* effectiveness) use the
> solid temperature `T_s`.

### B3 — Unit ambiguities (medium)

* **g_cat vs kg_cat.** Comments call the reaction rate `mol/(g_cat·s)`, but the
  surface balance and heat-release terms are only dimensionally consistent if it
  is `mol/(kg_cat·s)`. A stray factor of 1000 is easy to introduce here.
* **Knudsen diffusion** (`CatalystEffectivity_ThieleMod`): the mean molecular
  speed `√(8RT/πM)` is evaluated with `M` in g/mol rather than kg/mol, making
  the diffusivity ~30× too small (Thiele modulus too large). The Thiele
  expression also multiplies by `a_v_cat`, which leaves it dimensionally
  inhomogeneous.
* **Mixing rule.** Viscosity/conductivity mixing uses the simplified
  `φ = √(Mⱼ/Mᵢ)` weight rather than the full Wilke factor.

> **This solver:** SI throughout (`MW` in kg/mol); a correct first-order slab
> Thiele modulus `φ = L_wc·√(k_v/D_eff)`; full Wilke mixing rule.

---

## C. Data problems (the spreadsheets)

### D1 — N₂ molar mass is wrong, and it dominates the mixture (critical)

In `Reactor_Properties.xlsx`, sheet **Species**, N₂ has

```
Molar Mass = 14.0067     (this is atomic nitrogen; N₂ is 28.0134 g/mol)
```

This is used directly in `TransformWeightedFractions.m`
(`M_helpN2 = 1./Parameter_Reactor.Species.MolarMass(7)`) to form the mixture
molar mass, mole fractions, partial pressures and density. Because the feed is
~60 % N₂ by mass, the error propagates everywhere:

| quantity | with N₂ = 14.0067 | with N₂ = 28.0134 (correct) |
|---|---|---|
| mixture molar mass M_mix | ≈ 17.5 g/mol | ≈ 25.3 g/mol |

A ~45 % error in `M_mix` means a ~45 % error in gas density, in every partial
pressure fed to the kinetics, and in the convective velocity / residence time.
To make matters worse, `UpdateGasProperties.m` *hard-codes* 28.0134 for the
Fuller diffusion estimate — so two parts of the same code disagree about how
heavy nitrogen is.

> **This solver:** N₂ = 28.0134 g/mol, used consistently
> (`cpo/params.py`, with a comment flagging the spreadsheet error).

### D2 — The same parameter defined twice, with different values (medium)

Several quantities live in **both** `Reactor_Properties.xlsx` and
`Modelling_Properties.xlsx` with different numbers:

| parameter | Reactor_Properties | Modelling_Properties |
|---|---|---|
| ReactantStabilityFactor | 1e-3 | 1e-6 |
| MinConc | 1e-8 | 1e-12 |
| NoLines | 400 | 400 |

The stabilisation factor σ̃ enters every rate as `pᵢ/(pᵢ+σ̃)`; a factor-1000
difference changes where rates are throttled near depletion. Whichever sheet
happens to be read for a given field silently wins, which is a reproducibility
hazard.

> **This solver:** one source of truth (`cpo/params.py`); the stabilisation
> factor and concentration floors are set explicitly in one place.

### Not a bug (to avoid false alarms)

The adsorption constants `k_ads`/`H_ads` are populated **only** for O₂, CO and
H₂O and left blank for CH₄/CO₂/H₂/N₂. That is intentional and correct — those
are the inhibiting species in the LHHW rate laws — not a missing-data error.

---

## D. How this repository sidesteps all of the above

1. **Pure RHS** — no hidden/persistent state, so the implicit Jacobian is
   correct and the integration is reproducible (fixes A2).
2. **No stiff inner solve** — reactions evaluated at bulk composition with a
   justified small-Damköhler argument; the DAE collapses to a smooth, stiff ODE
   that `solve_ivp(BDF)` handles directly with a sparsity pattern (fixes A1/A3).
3. **One consistent unit system** (SI; molar→mass cp by division; SI molar
   masses; correct Thiele/Knudsen) (fixes B1–B3).
4. **One parameter source** in `cpo/params.py`, with N₂ corrected to 28.0134 and
   all duplicated/ambiguous settings pinned (fixes D1/D2).
5. **Smooth ignited start** (`T_g = T_s`) plus bounded developing-flow transfer
   correlations remove the start-up transient that the original setup could not
   get past.

The result is validated by **C/H/O element balances closing to machine
precision** and by reaction enthalpies / equilibrium constants matching textbook
values (ΔH: TOX −802, SR +206, WGS −41 kJ/mol; K_eq,WGS(1000 K) ≈ 1.44).
```
