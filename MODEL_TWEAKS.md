# Model tweaks & corrections

Every deviation of this solver from a literal transcription of the spreadsheet
+ MATLAB reference, why it was made, and its effect.  None of these change the
model *architecture* (governing equations, closures, coupling); they are
parameter/unit corrections, documentation-faithful choices, and numerical
regularisations.  Kinetic constants, feed, geometry and reaction scheme are
**unchanged** from your data.

## A. Data corrections (spreadsheet values that are inconsistent/typos)

| # | quantity | reference value | used here | why | effect |
|---|----------|-----------------|-----------|-----|--------|
| A1 | N₂ molar mass | 14.0067 g/mol | **28.0134** | 14.0067 is atomic N; N₂ is ~60 % of the mixture, so this shifts every mole fraction / partial pressure / density | large (M_mix 17.5 → 25.3 g/mol) |
| A2 | washcoat thickness `L_wc` | 1 µm (`CatLayerThickness`) | **10 µm** | the Model Description states 7–15 µm; 1 µm is inconsistent with it | negligible (peak T ±5 K, CH₄ ±1 %) |

## B. Unit / consistency corrections in the closures (MATLAB)

| # | term | reference | corrected to | why |
|---|------|-----------|--------------|-----|
| B1 | gas heat capacity in the gas-energy balance | `M_mix·c_PGas/1000` | `c_PGas/(M_mix·1e-3)` | `c_PGas` is **molar** (`=R·(cp/R)`); mass-specific cp is molar cp *divided* by molar mass, not multiplied |
| B2 | solid axial conduction | `k_ax·∂²T/∂z² / (ρ_s c_p)` | `… / (ρ_s c_p (1−ε))` | interphase transfer and reaction already carry `(1−ε)`; the reference conduction term omitted it → now all three solid terms are `(1−ε)`-consistent (energy-consistent). Lowered the hot spot 1828 → 1701 K |
| B3 | Knudsen diffusivity `√(8RT/πM)` | `M` in g/mol | `M` in kg/mol | otherwise the mean molecular speed (hence D_eff and the Thiele modulus) is off by √1000 |
| B4 | mixture viscosity/conductivity | `φ = √(Mⱼ/Mᵢ)` | full **Wilke** rule | standard, better-conditioned mixing rule |

## C. Documentation-faithful modelling choices

| # | choice | why |
|---|--------|-----|
| C1 | effectiveness factor applied **only to the O₂-consuming reactions** (TOX, HOX, COOX); SR and WGS keep η = 1 | Model Description: *"oxygen diffusional resistance magnitudes higher than other species ⇒ only oxygen-educt reactions with ThieleMod"*. (No numerical effect here — SR is limited by kinetics/CO-adsorption, not pore diffusion — but it matches the spec.) |
| C2 | reaction rates, ΔH_R, K_eq, effectiveness evaluated at **T_surface**; gas cp at **T_gas** | matches the reference "separate temperature treatment" (Zusammenfassung §5) |
| C3 | reaction heat scaled by `ρ_cat/ρ_s` with the `(1−ε)` solid fraction (`ρ_s = 3800` = α‑Al₂O₃, intrinsic) | reference `UseCatalystDensityInHeatSource = true` |

## D. Numerical regularisations (do not change the physics)

| # | item | note |
|---|------|------|
| D1 | washcoat species carried as states with relaxation time `τ_wc` | analog of the reference `AlgPseudoTransientTau`; the algebraic surface constraint is recovered as `τ_wc → 0` (verified: residual ∝ τ_wc, physics τ_wc-independent — see `verify.py`) |
| D2 | developing-flow Sherwood/Nusselt bounded to `[2.977, 50]` | removes the `Z*→0` entrance singularity of the Graetz correlation |
| D3 | smooth ignited initial profile (`T_g = T_s`) | pre-equilibrates the sub-ms gas/solid thermal layer so the integrator can start |

## E. Deliberately NOT changed (would be tuning, not a defensible tweak)

- **Kinetic constants** `k_873`, `E_A`, adsorption `k_ads`/`H_ads`, stabilisation
  factor — your calibrated data. These (not any tweakable parameter) set the
  ~50 % CH₄ conversion and hence the ~1700 K hot spot: the endothermic steam
  reforming that would cool a real CPO reactor is kinetically/adsorption
  limited here. Lowering the temperature to typical experimental values
  (~1100–1250 °C) would require **re-calibrating the reforming kinetics against
  data** (a *validation* step), not a small parameter tweak.
- **Feed** (mass fractions CH₄ 0.174 / O₂ 0.19 → molar O/C ≈ 1.09) and geometry —
  taken as given. The slight O-excess (O/C > 1) itself raises the temperature.
- **Adiabatic** operation — as in the reference; a real reactor loses some heat.

## Net effect of the tweaks on the developed state

| | peak T_s | outlet T_g | CH₄ conv | H₂/CO |
|---|---|---|---|---|
| after all corrections | ≈ 1701 K | ≈ 1531 K | ≈ 51 % | ≈ 2.3 |

The temperature is **robust** to every diffusion/geometry tweak tried; the
remaining gap to experimental CPO is a kinetics (validation) matter, documented
in section E.
