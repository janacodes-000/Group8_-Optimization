# 46750 - Assignment 1: Demand-Side Flexibility in Active Distribution Grids

Group 8's implementation of DTU course **46750 – Optimization in Modern Power
Systems**, Assignment 1. It implements Q1, Q2(b/c/e), Q3 and the battery model,
with reproducible experiments and numerical validation. Input JSON files remain
as supplied; sensitivity scenarios are derived in Python.

**Submitting your code.** Choose one of the two, and say in your report which one you chose:
attach your complete project code as a single `.zip` file to your submission in DTU Learn, or give
the link to your group's GitHub repository on the front page of your report. A linked repository must
be accessible to the graders (public, or private with the teaching team invited) and must not be
modified after the deadline - the last commit before the deadline is what is graded.

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

The Q2 sweep tables interpret the requested **binding deviation bound** as the
physical endpoints `load_min-reference <= load-reference <= load_max-reference`.
No extra deviation constraint is introduced. `deviation_bound_hours` counts
hours at either endpoint once, using `1e-6 kWh` absolute tolerance.
`lower_bound_hours`, `upper_bound_hours`, `deviating_bound_hours` and
`curtailed_bound_hours` separate actual bound-limited deviations from hours
whose zero reference already equals the lower bound. The absolute-value
auxiliary inequalities are not counted: at least one is always tight at an
optimum with positive linear discomfort cost.

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
Figures are saved as PNG only, without plot or subplot titles. Describe each
figure in its report caption; axis labels, units and legends remain on the plots.

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

Every variable family is stored in `self.var` and every constraint for which dual values are required is stored in `self.con`. The existing `solve()` method then extracts primal and dual results automatically.

**Conventions that make the primal and dual values come out for free**

* Store every variable family in `self.var[<name>]`. `solve()` returns the hourly values of each family
  as a column of `results.hourly`. gurobipy indexes names automatically - `m.addVars(T, name="p_import")`
  creates `p_import[0]` ... `p_import[23]`, and the same holds for `m.addConstrs(..., name=...)` - so
  per-hour names come for free. Families named `import`, `export`, `load`, `pv` make the standard plots
  of `src/plotting.py` work out of the box. All implemented models are continuous
  LPs/QPs, and required duals are extracted and validated.

* Store every constraint family in `self.con[<name>]`. `solve()` returns the dual value (Gurobi attribute
  `Pi`) of every hourly constraint as a column `dual_<name>` of `results.hourly`, and of every single
  constraint in `results.duals`.
* Write the bounds you want a dual for as explicit constraints (`m.addConstr(...)`), not as variable bounds
  (`lb=`, `ub=`). Gurobi reports the sensitivity of a variable bound in the reduced cost (`RC`), not in `Pi`.
* Quadratic disutility is in the objective; all constraints are linear.
  Gurobi's `Pi` gives objective sensitivity to the right-hand side: non-positive
  for `<=` constraints and non-negative for `>=` constraints in these minimization
  models. The balance is `PV + import (+ discharge) = load + export (+ charge)`.
  Report the matching sign convention when interpreting its marginal value.

## 4. Input data

All input data lives in `data/`: two **shared** files plus one small **case** file per
question. The shared files form a catalogue of every appliance used anywhere in the assignment;
each `params_<case>.json` composes the consumer of one case by picking appliances from the
catalogue by ID. All time series have 24 hourly values (hour 0 to 23).

**Units.** Every field name carries its unit. Fields ending in `_kW` are power ratings (kW);
fields ending in `_kWh_per_hour` (or `_KWh_per_hour` - same unit despite the capital K) are hourly
energy bounds (kWh consumed within one hour); fields ending in `_kWh` are energy amounts;
`_DKK_per_kWh` are prices/costs and `_DKK_per_kWh2` is the quadratic disutility coefficient.
**Capacity factors and ratios** (`hourly_profile_ratio`, `reference_load_capacity_factor`,
`initial_state_of_charge_ratio`, and the `_next_day` profiles) are **dimensionless fractions in
[0, 1] of the corresponding maximum** - multiply by it to get physical values. With 1-hour periods,
a power rating in kW and an energy amount in kWh/h are numerically interchangeable; the loader
does every conversion for you and exposes physical quantities:
`pv_available = max_power_kW x hourly_profile_ratio` (kWh/h),
`reference_load = max_load_KWh_per_hour x reference_load_capacity_factor` (kWh/h), and
`battery_initial_soc_kWh = storage_capacity_kWh x initial_state_of_charge_ratio` (kWh).

| Question in the assignment | Case | Appliances (from the catalogue) |
|---|---|---|
| Question 1 - hourly consumption decision (utility u, PV cost c_PV), case A: c_PV < u | `Q1_caseA` | `PV_01_A` + `FL_01` (a load with `consumption_utility_DKK_per_kWh`; no reference profile, no daily requirement) |
| Question 1, case B: a more expensive PV, c_PV > u | `Q1_caseB` | `PV_01_B` + `FL_01` (`PV_01_B` differs from `PV_01_A` only in `marginal_cost_DKK_per_kWh`) |
| Question 2 - linear disutility of deviating from a reference profile | `Q2_linear` | `PV_01_A` + `FL_02_L` (reference profile, `linear_disutility_DKK_per_kWh`; same hourly bounds as Question 1; no utility) |
| Question 2 - quadratic disutility | `Q2_quadratic` | `PV_01_A` + `FL_02_Q` (as `FL_02_L` with `quadratic_disutility_DKK_per_kWh2` instead) |
| Question 3 - minimum daily energy requirement | `Q3` | `PV_01_A` + `FL_03` (as `FL_02_Q` plus `min_total_energy_per_day_kWh`, the E_min of the daily requirement) |
| Question 3.(g) - the same consumer with a small battery | `Q3_battery` | `PV_01_A` + `FL_03` + `BAT_01` |

No data is shipped for the bonus question 2.(e) on purpose: choosing your own cost parameters (and
justifying them) is part of that question - start from the `Q2_quadratic` case and override in code.

`params_<case>.json` - the case file
: `consumers[]`: `consumer_id`, `connection_bus` (a `bus_id` of `bus_params.json`), `list_appliances`
  (IDs of the consumer's DERs, loads and storages in `appliance_params.json`). `hyperparameters` is
  unused in Assignment 1 (`null`).

`appliance_params.json` - the shared catalogue, one list per appliance type (`null` if none)
: **DERs**: `DER_id`, `DER_type` (`"PV"`), `max_power_kW` (peak power, kW), `marginal_cost_DKK_per_kWh` (cost of every kWh produced), `hourly_profile_ratio` (dimensionless fraction of `max_power_kW` available each hour; available energy in kWh/h = `max_power_kW` x ratio), `hourly_profile_ratio_next_day` (see *Next-day forecasts* below)
: **loads**: `load_id`, `load_type`, `max_load_KWh_per_hour` and `min_load_KWh_per_hour` (hourly consumption bounds, kWh/h), `consumption_utility_DKK_per_kWh` (value of every kWh consumed, Question 1), `linear_disutility_DKK_per_kWh` and `quadratic_disutility_DKK_per_kWh2` (Questions 2-3), `min_total_energy_per_day_kWh` (the E_min of Question 3), `reference_load_capacity_factor` (dimensionless fraction of `max_load_KWh_per_hour`; preferred consumption in kWh/h = max load x factor), `reference_hourly_profile_ratio_next_day` (see *Next-day forecasts*). Fields a load does not use are `null`. The hourly bounds are the same for every load: deviations from the reference are limited only by them.
: **storages**: `storage_id`, `storage_type` (`"battery"`), `storage_capacity_kWh`, `max_charging_power_kW`, `max_discharging_power_kW`, `charging_efficiency` and `discharging_efficiency` (fraction of the energy transferred that survives each conversion - a full round trip keeps η_ch·η_dis), `initial_state_of_charge_ratio` (fraction of `storage_capacity_kWh` in the battery at the start of the day). **No final state of charge is given on purpose**: how to treat the battery's state of charge at the end of the day (constraint or objective) is your modeling choice in Question 3.(g).
: **heat_pumps**: not used in Assignment 1 (`null`)

`bus_params.json` - grid and market conditions at the connection bus
: `bus_id`, `import_tariff_DKK_per_kWh`, `export_tariff_DKK_per_kWh`, `energy_price_DKK_per_kWh` (24 values), `energy_price_next_day_DKK_per_kWh` (see *Next-day forecasts*), `max_import_kW`, `max_export_kW` (`null` = no grid limit in Assignment 1)

**Next-day forecasts.** Three fields carry the forecast for the *following* day:
`energy_price_next_day_DKK_per_kWh` (in `bus_params.json`), `hourly_profile_ratio_next_day` (in the
DER entries) and `reference_hourly_profile_ratio_next_day` (in the load entries). They are **not
needed for the base models**: they are provided in case you want to explore the end-of-horizon
choice of Question 3.(g) with a look-ahead (for instance a rolling-horizon or a two-day experiment).
`load_question()` exposes them as `energy_price_next_day`, `pv_available_next_day` and
`reference_load_next_day`.

`load_question()` in `src/data_loader.py` shows exactly how each attribute is derived from these files.

**Modifying or adding data.** For sensitivity analyses, prefer deriving scenarios in code
(`src/scenarios.py`) over editing the JSON files - it keeps the base case intact and the experiment
reproducible. If you do add appliances or case files, follow the same structure (a new entry in the
catalogue, a new `params_<case>.json`) and document them here.

## 5. What is expected of your code

Your code is part of the submission (as a repository link or a `.zip`, see the top of this README).
The graders should be able to open it, follow this README, and reproduce every number and figure in
your report. In practice:

* keep the separation between data loading, model building, solving and plotting;
* document every function you add (a short docstring stating inputs, outputs and units is enough), and
  describe any new module or data file in this README;
* make each experiment of the report runnable with a single command (e.g. `python main.py --question ...`
  or one notebook cell), and save its outputs under `results/`;
* commit regularly and with meaningful messages - the git history is also a record of everyone's contribution.

## 6. Question 2(e): temporal disutility bonus

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
normalization. Its level depends on the preference representation; compare
ordering effects within each model rather than treating cross-model surplus
levels as a common welfare ranking. The runner verifies all objective cost
components and the separable models' invariance to price ordering.

## 7. Scenario assumptions and input checks

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
