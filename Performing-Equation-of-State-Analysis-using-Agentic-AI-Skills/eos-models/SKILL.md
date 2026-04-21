---
name: eos-models
description: Fit EOS models from E(V), P(V), or enthalpy workflows with optimized coefficients, MSE ranking, and graphical/text reporting. 
---

# EOS Models

Use this skill to fit equation-of-state (EOS) data and generate curves/reports.

## Supported Inputs

- `EvsV.txt`: two columns -> volume (Angstrom^3), energy (eV)
- `PvsV.dat`: two columns -> volume (Angstrom^3), pressure (GPa)

## Required Outputs

For `EvsV.txt`:

1. `EOS_fit_results.txt`
2. `EvsV.png`
3. `PvsV.png`
4. terminal summary (MSE table + best model)

For `PvsV.dat`:

1. `EOS_fit_results.txt`
2. `PvsV.png`
3. `EvsV.png` (derived by integrating pressure)
4. terminal summary (model + MSE)

## Dependency and Runtime Rules

1. Prefer `python3`; if packages are missing, retry with `python`.
2. Required packages: `numpy`, `scipy`, `matplotlib`.
3. Use `matplotlib.use("Agg")` before importing `pyplot`.
4. Read input using `numpy.loadtxt`.
5. Validate data shape before fitting.

Suggested dependency check:

```bash
python3 -c "import numpy, scipy, matplotlib; print('ok')" || python -c "import numpy, scipy, matplotlib; print('ok')"
```

## Reference Script (Recommended)

Use the bundled implementation instead of regenerating code each time:

- `.opencode/skills/eos-models/scripts/run_eos_fit.py`

Example commands:

```bash
python .opencode/skills/eos-models/scripts/run_eos_fit.py --infile EvsV.txt --eostype energy
python .opencode/skills/eos-models/scripts/run_eos_fit.py --infile PvsV.dat --eostype pressure
```

This script preserves method parity (`plot_eos`, `fitting`, `_get_peos`, `_get_eeos`) and writes `EOS_fit_results.txt` plus PNG figures.

## Do-Not-Remove Functionality

If working in class/API mode, keep these methods and behaviors:

- `plot_eos(...)`
- `fitting(...)`
- `_get_peos(...)`
- `_get_eeos(...)`
- energy models: `eos_vinet`, `eos_birch`, `eos_murnaghan`, `eos_birch_murnaghan`
- pressure model: `eos_birch_murnaghan_pressure`

Why this matters:

- removing `fitting` breaks optimized coefficients + MSE ranking
- removing `_get_peos` breaks `EvsV.txt -> PvsV.png`
- removing `_get_eeos` breaks `PvsV.dat -> EvsV.png`

## Required Option Parity (plot_eos)

Keep this API (same option names):

```python
plot_eos(
    infile=None,
    eostype="energy",         # energy | pressure | enthalpy
    natoms=1,
    au=False,
    vlim=None,
    model=None,                # Vinet | Birch | Murnaghan | Birch-Murnaghan
    raw_data=True,
    export=True,
    savefig=True,
)
```

Behavior to preserve:

- `natoms`: divide plotted intensive quantities by atom count
- `au=True`: convert energy/volume labels and values to atomic units
- `vlim`: custom interpolation range for smooth curves
- `model`: plot only chosen model; `None` plots all supported models
- `raw_data`: include raw points
- `savefig`: save PNG figures
- `export`: retained for API parity; only `EOS_fit_results.txt` is written

## Streamlined Execution Flow

For `eostype="energy"`:

1. load `V, E`
2. quadratic initial guess (`polyfit`) -> `[E0, B0, Bp, V0]`
3. `fitting()` for 4 energy models
4. compute MSE for each model + select best
5. `_get_peos()` by central difference to get pressure curves
6. plot `EvsV.png` and `PvsV.png`
7. write `EOS_fit_results.txt`

For `eostype="pressure"`:

1. load `V, P`
2. quadratic initial guess (`polyfit`) -> `[P0, K0, Kp, V0]`
3. `fitting()` for pressure EOS (Birch-Murnaghan pressure form)
4. compute MSE
5. `_get_eeos()` by integrating pressure to relative energy
6. plot `PvsV.png` and derived `EvsV.png`
7. write `EOS_fit_results.txt`

For `eostype="enthalpy"`:

- keep original per-phase best-model selection behavior in `fitting()`
- keep `_get_peos()` enthalpy branch that computes phase pressure arrays

## Reference Implementation Snippets

Use these formulas exactly to preserve numerical behavior.

```python
def eos_murnaghan(coeffs, vol):
    E0, B0, Bp, V0 = coeffs
    return E0 + B0 / Bp * vol * ((V0 / vol) ** Bp / (Bp - 1.0) + 1.0) - V0 * B0 / (Bp - 1.0)


def eos_birch_murnaghan(coeffs, vol):
    E0, B0, Bp, V0 = coeffs
    eta = (vol / V0) ** (1.0 / 3.0)
    return E0 + 9.0 * B0 * V0 / 16.0 * (eta ** 2 - 1.0) ** 2 * (6.0 + Bp * (eta ** 2 - 1.0) - 4.0 * eta ** 2)


def eos_birch(coeffs, vol):
    E0, B0, Bp, V0 = coeffs
    t = (V0 / vol) ** (2.0 / 3.0) - 1.0
    return E0 + 9.0 / 8.0 * B0 * V0 * t ** 2 + 9.0 / 16.0 * B0 * V0 * (Bp - 4.0) * t ** 3


def eos_vinet(coeffs, vol):
    E0, B0, Bp, V0 = coeffs
    eta = (vol / V0) ** (1.0 / 3.0)
    return E0 + 2.0 * B0 * V0 / (Bp - 1.0) ** 2 * (
        2.0 - (5.0 + 3.0 * Bp * (eta - 1.0) - 3.0 * eta) * np.exp(-3.0 * (Bp - 1.0) * (eta - 1.0) / 2.0)
    )


def eos_birch_murnaghan_pressure(coeffs, vol):
    _P0, K0, Kp, V0 = coeffs
    eta = (V0 / vol) ** (1.0 / 3.0)
    return (3.0 / 2.0) * K0 * (eta ** 7 - eta ** 5) * (1.0 + 0.75 * (Kp - 4.0) * (eta ** 2 - 1.0))
```

```python
def _get_peos(self, dv=1e-3):
    # P = -dE/dV, then convert eV/Angstrom^3 -> GPa with 1.60217e02
    def p_from_e(model_fn, coeffs, vol):
        e1 = model_fn(coeffs, vol - dv)
        e2 = model_fn(coeffs, vol + dv)
        return -1.0 * (e2 - e1) / (2.0 * dv) * 1.60217e02
```

```python
def _get_eeos(self):
    # relative energy from pressure integration
    # convert GPa*Angstrom^3 -> eV with 6.241509e-3
    from scipy.integrate import cumulative_trapezoid
    p = self.eos_birch_murnaghan_pressure(self.eos_birch_murnaghan_pressure_fitted.x, self.volume)
    self.E_birch_murnaghan = -cumulative_trapezoid(p, self.volume, initial=0.0) * 6.241509e-3
```

## Minimal Reporting Contract

Always include:

- fitted coefficients per model
- MSE per model
- best model name (when multiple models are fit)
- generated file paths
- Tabulated MSE summary for all models 

After the required reporting, add a brief interpretation section (3-6 bullets):

- coefficient meaning: `E0`/`P0` baseline, `V0` equilibrium volume, `B0`/`K0` bulk modulus, `Bp`/`Kp` pressure derivative
- fit quality: lower MSE means better agreement with input data
- stiffness context: larger `B0` (`K0`) means a stiffer material near equilibrium
- unit hint: convert `B0` from `eV/Angstrom^3` to `GPa` using `1 eV/Angstrom^3 = 160.21766208 GPa`
- plausibility check: flag clearly non-physical coefficients (for example `B0 <= 0`, extreme `Bp`, or unrealistic `V0`)

Keep interpretation concise and practical; prefer one short paragraph or bullet list.

Use real newline characters in text output (do not write escaped `\\n` literals).

## Common Failure Fixes

1. `ModuleNotFoundError` on `python3` -> retry with `python`.
2. Empty/invalid data -> confirm two numeric columns, no header.
3. Plotting fails in non-GUI environment -> set `matplotlib.use("Agg")` before `pyplot` import.
4. Unrealistic fit -> verify initial guess from quadratic fit and data ordering/range.
