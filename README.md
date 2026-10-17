# 46750 - Assignment 1: Demand-Side Flexibility in Active Distribution Grids

Group 8's implementation of DTU course **46750 – Optimization in Modern Power
Systems**, Assignment 1. It implements Q1, Q2(b/c/e), Q3 and the battery model,
with reproducible experiments and numerical validation. Input JSON files remain
as supplied; sensitivity scenarios are derived in Python.


## 1. Setup

### 1.1 Python environment

Use one of the two options (both install the same packages).

**Option A - pip and venv**
```bash
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate.bat         # Windows cmd
# venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

**Option B - conda**
```bash
conda env create -f environment.yaml
conda activate 46750-a1
```

### 1.2 Gurobi licence

The code uses [Gurobi](https://www.gurobi.com) through the `gurobipy` package. The package ships with a
restricted licence that was sufficient for all tested assignment models, so
`python main.py` works out of the box. For unrestricted use, request a free
[academic licence](https://www.gurobi.com/academia/academic-program-and-licenses/) with your DTU e-mail
and activate it with `grbgetkey <your-key>` (on the DTU network or VPN).

### 1.3 Check that everything works
```bash
python main.py
```
This solves Q1 case A and saves its schedule, cost/utility components and duals.
Run `python main.py --validate` for the analytical and physical checks. The
implementation was verified with Python 3.12.14, gurobipy 13.0.3, NumPy 2.3.5,
pandas 2.2.3 and Matplotlib 3.10.8. The conda environment uses Python 3.11;
`requirements.txt` records the supported minimum dependency versions.

## 2. Repository structure

```
main.py                  Entry point: load data -> build model -> solve -> analyse -> save results and figures
src/
  data_loader.py         Load and validate input data for each assignment case
  model.py               Gurobi optimisation models for Q1, Q2 and Q3
  scenarios.py           Helpers to create modified input scenarios for sensitivity analyses
  analysis.py            Post-processing, summary metrics and comparison of solved optimisation cases
  plotting.py            Figures for input data, optimal schedules, duals and comparisons
data/
  appliance_params.json  SHARED catalogue: every PV system, flexible load and battery - see Section 4
  bus_params.json        SHARED grid connection: prices (both days) and tariffs
  params_Q1_caseA.json   One small file per case: composes a consumer from the catalogue
  params_Q1_caseB.json
  params_Q2_linear.json
  params_Q2_quadratic.json
  params_Q3.json
  params_Q3_battery.json
results/                 Written by main.py (git-ignored)
requirements.txt, environment.yaml, LICENSE, .gitignore
```

The modules separate data, model construction/solving, scenario transformations,
analysis and figures. Experiments stay in `main.py`. Question-specific classes
are thin configurations of a shared parameterized LP/QP builder.

## 3. How to use the code

Q1 solutions return daily procurement cost, gross consumption utility and net
surplus in `results.metrics`, in DKK. The minimized objective is procurement
minus utility; surplus is its negative. Components are saved in the JSON
summary and checked against the solver objective.

The Q2 sensitivity analyses distinguish between binding absolute-deviation
constraints and physical load bounds.

`deviation_bound_hours` counts the hours in which at least one of the
absolute-deviation inequalities is binding. Since the auxiliary deviation
variable represents the absolute difference between optimized and reference
consumption, at least one inequality is binding in each hour for the linear
disutility model with a positive coefficient.

The physical consumption limits are evaluated separately:
`load_bound_hours` counts hours at either the minimum or maximum load,
while `lower_bound_hours` and `upper_bound_hours` report the respective
limits individually. `deviating_bound_hours` counts hours where consumption
both deviates from the reference and reaches a physical load bound.
`curtailed_bound_hours` identifies hours where consumption is reduced below
the reference and reaches the minimum load.

All binding conditions are evaluated using an absolute tolerance of
`1e-6 kWh`.

Run these commands from the repository root:

| Report experiment | Command | Output directory |
|---|---|---|
| Q1 case A, including price/tariff/PV controls | `python main.py --question Q1_caseA --scenarios` | `results/Q1_caseA/` |
| Q1 case B, including controls | `python main.py --question Q1_caseB --scenarios` | `results/Q1_caseB/` |
| Q2(b) base and linear-coefficient sweep | `python main.py --question Q2_linear --scenarios` | `results/Q2_linear/` |
| Q2(c) base and quadratic-coefficient sweep | `python main.py --question Q2_quadratic --scenarios` | `results/Q2_quadratic/` |
| Q2(d) selected-scenario comparison | `python main.py --q2-comparison` | `results/Q2_comparison/` |
| Q2(e) base and price-order experiment | `python main.py --q2-bonus` | `results/Q2_bonus/` |
| Q3 base, comparison and two sensitivities | `python main.py --question Q3 --scenarios` | `results/Q3/` |
| Q3(g) battery base, comparison and sensitivities | `python main.py --question Q3_battery --scenarios` | `results/Q3_battery/` |
| Sanity checks, ties and dual sensitivity | `python main.py --validate` | `results/validation/` |

Omit `--scenarios` for a base case. Add `--show` to an experiment command to
open all figures after computation; otherwise figures are saved without a GUI.
Figures are saved as PNG files without plot or subplot titles.
All figures include axis labels, units, and legends where applicable.

`--q2-comparison` runs the existing Q2 sweeps once and reuses their results for
the base case and three selected scenarios: weak preferences `(cL=0.5, cQ=0.1)`,
transition `(1.41, 0.5)`, and strong preferences `(2.5, 20)`.
These pairings demonstrate regimes in each model; they do not equate coefficients
with different units or assert equal comfort valuation. At the linear tie
`cL=cPV=1.41`, a returned schedule is one possible optimal schedule.
The comparison CSV and per-scenario schedules/figures are in
`results/Q2_comparison/`; all physical inputs are held equal across models.

Q3 energy-sweep threshold annotations use the solved unconstrained quadratic
daily load, rather than a hardcoded value or the first positive sampled dual.
`energy_active` uses primal slack (`1e-6 kWh` tolerance); `energy_binding` also
requires a positive dual above `1e-6 DKK/kWh`. The threshold sweep tightens
Gurobi's `BarConvTol` to `1e-12` to resolve small slacks and marginal values.
The existing PV-utilisation marker is computed by bisection, with a 0.001 kWh
energy interval tolerance and a 0.00001 kWh unused-PV tolerance. The threshold
regression test derives its expectation from the independent analytical hourly
solution, rather than fixed previously observed values.

Every model solve validates balance/bounds, reconstructed economic components,
relevant dual identities and complementary slackness. Separable Q1/Q2 models
are checked against economic breakpoints and quadratic stationary points;
Q3 additionally uses a scalar daily-energy multiplier search. Linear ties are
checked by optimal objective, allowing any optimal schedule. Battery checks
include all 25 SoC states, cyclic closure and charge/discharge overlap. QP
solves use `BarConvTol=1e-12`. `--validate` also checks regime extremes, a price
permutation, the uncoupled temporal limit and a finite-difference Q3 dual.
Its outputs are in `results/validation/`. Regression checks can be run with
`python -m unittest discover -s tests`.

Generated outputs are kept compact:

| Experiment | Saved outputs |
|---|---|
| Base cases and selected comparisons | Hourly CSV, JSON summary and PNG figures |
| Coefficient, energy and battery sensitivity sweeps | Combined metrics CSV and PNG figures |
| Numerical validation | `validation.csv` and `checks.json` |
| Saved battery solutions | An additional `*_battery_states.csv` with all 25 boundary states |

JSON summaries retain the actual scenario parameters, validation diagnostics and
available hourly analytical comparisons, including Q1 balance-dual intervals.
Separate TXT summaries and analytical CSVs are omitted because they duplicate
these records. Sensitivity sweeps retain aggregate tables rather than exporting
every individual solve. Q3/battery comparisons retain their paired schedules and
`comparison.csv`; battery sensitivity tables include both objectives and overlap
counts. Files under `results/` are generated and need not be committed.

Existing files are not deleted automatically. To avoid mixing old and new outputs,
rename the existing `results/` directory before regenerating the experiments,
then restore the tracked placeholder with `git restore -- results/.gitkeep`.

**Use it from a notebook or your own script:**

```python
from src.data_loader import load_question
from src.model import FlexibleConsumerModel
from src.plotting import plot_schedule, plot_duals
from src.scenarios import scale_prices

data = load_question("Q1_caseA")
print(data.summary())

results = FlexibleConsumerModel(data).build().solve()
print(results)                    # objective, daily totals, scalar duals
results.hourly                    # DataFrame: one row per hour with variables, prices and hourly duals
plot_schedule(results, data)

high_spread = scale_prices(data, factor=2.0, keep_mean=True)
results_hs = FlexibleConsumerModel(high_spread).build().solve()
```

### Implemented optimization models

The optimisation models are implemented in `src/model.py`.

- `FlexibleConsumerModel` implements Question 1 with price-elastic hourly consumption.
- `LinearDisutilityModel` implements Question 2(b) with absolute-deviation disutility.
- `QuadraticDisutilityModel` implements Question 2(c) with quadratic disutility.
- `TemporalDisutilityModel` configures Q2(e) with adjacent-hour discomfort.
- `DailyEnergyModel` configures Q3 with quadratic discomfort and a daily minimum.
- `BatteryModel` configures Q3(g) with losses, power/capacity limits and cyclic terminal SoC.

All question classes now configure one parameterized `FlexibleConsumerModel`:
`preference` selects utility/linear/quadratic/temporal; `daily_energy` and
`battery` enable coupling. Physical balances, procurement, absolute deviations
and battery dynamics each have one implementation. The public class names are
compatible wrappers. `MODEL_CLASSES`, `SWEEPS`, `Q1_SCENARIOS`,
`Q2_COMPARISON_SELECTIONS` and `PRICE_PROFILES` in `main.py` declare experiments;
the shared sweep runner returns results in memory and reuses existing base-case
solutions; experiment runners save aggregate tables and figures.


### Primal and Dual Value Extraction

The optimization models store decision variables in `self.var` and
constraints in `self.con`. The `solve()` method automatically extracts
the corresponding primal and dual values from Gurobi.

Hourly decision variables, such as `import`, `export`, `load`, and `pv`,
are stored in `results.hourly`. The dual values of hourly constraints
are included in the same DataFrame as `dual_<name>`, while dual values
of individual constraints are stored in `results.duals`.

Physical bounds for which shadow prices are required are implemented
as explicit constraints. This allows their dual values to be extracted
directly through Gurobi's `Pi` attribute.

All implemented optimization models are continuous linear or quadratic
programs (LP/QP). Quadratic disutility is represented in the objective
function, while the constraints remain linear.

The nodal power balance is formulated as:

`PV + import (+ battery discharge) = load + export (+ battery charge)`

The extracted dual values represent the marginal change in the
objective function with respect to the right-hand side of the
corresponding constraints, following Gurobi's sign convention.

## 4. Input Data

All optimization models use the input data provided for Assignment 1,
stored in the `data/` directory. The original JSON files remain unchanged
to ensure reproducibility and consistency across the different questions.

### 4.1 Data Structure

The input data is organized into two shared parameter files and
individual configuration files for each optimization case.

| File | Description |
|---|---|
| `appliance_params.json` | Technical and economic parameters of the PV systems, flexible loads, and battery storage. |
| `bus_params.json` | Electricity prices, import/export tariffs, and grid connection parameters. |
| `params_Q1_caseA.json` | Configuration for Question 1, Case A. |
| `params_Q1_caseB.json` | Configuration for Question 1, Case B. |
| `params_Q2_linear.json` | Configuration for the linear disutility model. |
| `params_Q2_quadratic.json` | Configuration for the quadratic disutility model. |
| `params_Q3.json` | Configuration for the daily minimum energy requirement. |
| `params_Q3_battery.json` | Configuration for the battery extension. |

Each case file specifies which appliances and parameters are used in
the corresponding optimization problem.

### 4.2 Data Processing

Input data is loaded and processed through `src/data_loader.py`.

The `load_question()` function reads the corresponding JSON files,
extracts the relevant parameters, and converts them into the physical
quantities required by the optimization models.

The main input parameters include:

- Hourly electricity prices and import/export tariffs (DKK/kWh).
- Available PV generation (kWh per hour).
- Minimum and maximum hourly electricity consumption (kWh per hour).
- Reference consumption profiles (kWh per hour).
- Consumption utility and disutility coefficients.
- Minimum daily energy requirements (kWh).
- Battery energy capacity (kWh), charging/discharging power limits
  (kW), efficiencies, and initial state of charge.

All optimization cases use a 24-hour horizon with hourly time steps.

PV availability and reference consumption are calculated from the
corresponding maximum capacities and hourly profile ratios.
Battery initial state of charge is calculated from the storage
capacity and the specified initial SoC ratio.

The supplied data also includes next-day electricity prices, PV
availability, and reference consumption profiles. These are loaded
by the data loader but are not used in the implemented single-day
optimization models.

### 4.3 Sensitivity Analyses

Sensitivity analyses are implemented in `src/scenarios.py`.

The scenarios modify selected model parameters, such as electricity
prices, PV availability, disutility coefficients, minimum daily
energy requirements, and battery capacity.

All modifications are performed programmatically without changing
the original JSON input files. This ensures that the base cases
remain reproducible and that the impact of individual parameters
can be evaluated consistently.

### 4.4 Model Assumptions

The optimization models use the following assumptions:

- The optimization horizon consists of 24 one-hour periods.
- Electricity prices and PV availability are assumed to be known
  for the optimization horizon.
- The original reference consumption profiles remain unchanged
  during sensitivity analyses unless explicitly modified.
- The battery model includes charging and discharging efficiencies,
  power limits, and energy capacity constraints.
- A cyclic terminal state-of-charge condition is imposed for the
  battery, meaning that the final SoC equals the initial SoC.

Additional assumptions specific to individual questions are
described in the corresponding sections of the report.


## 5. Question 2(e): temporal disutility bonus

Reproduce the implemented Q2(e) model with:

```bash
python main.py --q2-bonus
python main.py --q2-bonus --show
```

This follows the existing structure:

- `src/model.py`: the new temporal preference model reuses the physical model.
- `main.py`: `run_q2_bonus_base` and `run_q2_price_structure_experiment` run
  the experiments, in the same pattern as the existing Q2 sweeps and Q3 comparisons.
- `src/analysis.py`: `compare_q2_disutility_models` reuses
  `analyze_linear_sensitivity` and `analyze_quadratic_sensitivity` for Q2(b/c).
  Procurement cost continues to use `calculate_procurement_cost`. Only the
  temporal disutility calculation and cross-model comparison are new.
- `src/scenarios.py`: `set_q2_price_structure` reuses `scale_pv` to remove PV
  availability and overrides only the prices and reference load.
- `src/plotting.py`: two new figures show the cross-model base comparison
  and the controlled price-order experiment.

There is no separate bonus-runner module. Models are solved once per case,
their Results objects are shared by analysis and plotting, and hourly results
are written with the existing `Results.save`. Neither coefficient sweep is
rerun by the bonus command. Metrics retain the existing Q2 names:
`daily_load`, `absolute_deviation`, `procurement_cost` and `disutility`.

Tables and figures are saved under `results/Q2_bonus/base/` and
`results/Q2_bonus/price_structure/`.

With `a_t = |load_t - reference_t|`, temporal discomfort is
`D = c0 * sum(a_t^2) + cT * sum((a_(t-1) + a_t)^2)`, including zero padding
outside the day. The coefficients are `c0 = 0.5` and `cT = 0.25`
DKK/kWh^2. Thus `c0 + 2*cT = 1`, matching the supplied quadratic model's
penalty for isolated deviations; adjacent deviations incur an extra cost.
The auxiliary-variable formulation is a convex QP.

Base cases retain the supplied Q2 inputs. The controlled experiment uses
twelve prices of 0.85 and twelve of 2.50 DKK/kWh, either alternating or in
two twelve-hour blocks. Every model receives no PV availability and a constant
reference equal to the original daily reference energy divided by 24.
Tariffs, bounds and each model's coefficients are fixed. Only price ordering
changes between the two profiles.

Daily surplus is minus the Q2 objective under the zero-at-reference discomfort
normalization. The surplus level depends on the preference representation.
Therefore, the analysis focuses on the effects of price ordering
within each model rather than interpreting differences in surplus
levels across models as a common welfare ranking. The runner verifies all objective cost
components and the separable models' invariance to price ordering.

## 6. Scenario assumptions and input checks

Input construction and model initialization reject non-finite values, inconsistent
PV availability, invalid load bounds, non-positive discomfort coefficients,
infeasible daily requirements, incomplete battery data, invalid efficiencies and
out-of-capacity initial SoC. The supplied reference remains fixed when load bounds
are varied; it may be outside new physical bounds, representing unavoidable
comfort loss. Periods remain one hour. Grid limits are absent in the supplied data
and are rejected rather than silently ignored.

Mean-preserving price scaling rejects factors that would produce negative prices;
prices are not clipped because clipping changes the mean. The factor-2 battery
experiment has a slightly negative effective export price in one hour
(`-0.019167 DKK/kWh`), even though wholesale prices remain non-negative.
Saved metadata records whether all effective export prices are positive. This
experiment must be interpreted without assuming they all are. All tested battery
runs still satisfy the non-overlap check.

PV scaling also rejects factors exceeding the installed peak rather than clipping
the profile. Battery-capacity scenarios preserve the supplied initial SoC fraction;
for the base data this is 50%. The implemented end condition is cyclic, not a free
terminal state or salvage-value objective.
