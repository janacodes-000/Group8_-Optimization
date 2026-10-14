"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace
import json
#from turtle import pd
import pandas as pd
import numpy as np

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, DailyEnergyModel, BatteryModel, Results, TemporalDisutilityModel
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_linear_disutility_sensitivity,plot_q3_comparison, plot_quadratic_disutility_sensitivity, plot_q3_energy_sensitivity, plot_q3_energy_supply_sensitivity, plot_q3_cq_sensitivity, plot_q3_battery_comparison, plot_battery_value_sensitivity, plot_q2_bonus_base, plot_q2_price_structure
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_linear_disutility, set_quadratic_disutility, set_load_preferences, set_battery_capacity, set_q2_price_structure
from src.analysis import compare_q2_q3, analyze_linear_sensitivity, analyze_quadratic_sensitivity, analyze_q3_energy_sensitivity, analyze_q3_cq_sensitivity, compare_q3_battery, compare_q2_disutility_models

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    if question == "Q2_linear":
        model = LinearDisutilityModel(data).build()
    elif question == "Q2_quadratic":
        model = QuadraticDisutilityModel(data).build()
    elif question == "Q3":
        model = DailyEnergyModel(data).build()
    elif question == "Q3_battery":
        model = BatteryModel(data).build()
    else:
        model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    if results.metrics:
        print(pd.Series(results.metrics).to_string(), "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results

def run_linear_disutility_sweep(out: Path):
    """Run Q2 linear for different values of the linear disutility coefficient."""

    data = load_question("Q2_linear")

    cL_values = [
        0.5, 1.0, 1.2, 1.3, 1.4,
        1.409, 1.41, 1.411, 1.43,
        1.5, 1.75, 2.0, 2.5, 3.0
    ]

    results = []

    for cL in cL_values:

        # Create scenario
        scenario_data = set_linear_disutility(data, cL)

        # Solve model
        result = LinearDisutilityModel(scenario_data).build().solve()

        results.append((cL, result))

    # Analyze all results
    metrics = analyze_linear_sensitivity(data, results)

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        out / "linear_disutility_sensitivity.csv",
        index=False,
    )

    # Print summary
    for m in metrics:
        print(
            f"cL={m['cL']:.3f} | "
            f"load={m['daily_load']:.1f} kWh | "
            f"deviation={m['absolute_deviation']:.1f} kWh | "
            f"procurement={m['procurement_cost']:.2f} DKK | "
            f"disutility={m['disutility']:.2f} DKK"
        )

    plot_linear_disutility_sensitivity(
        results,
        data,
        save_to=out / "linear_disutility_sensitivity.png",
    )

    return results

def run_quadratic_disutility_sweep(out: Path):
    """Run Q2 quadratic for different values of the quadratic disutility coefficient."""

    data = load_question("Q2_quadratic")

    cQ_values = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    ]

    results = []

    for cQ in cQ_values:

        # Create scenario
        scenario_data = set_quadratic_disutility(data, cQ)

        # Solve model
        result = QuadraticDisutilityModel(
            scenario_data
        ).build().solve()

        results.append((cQ, result))

    # Analyze results
    metrics = analyze_quadratic_sensitivity(
        data,
        results,
    )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        out / "quadratic_disutility_sensitivity.csv",
        index=False,
    )

    # Print summary
    for m in metrics:
        print(
            f"cQ={m['cQ']:.3f} | "
            f"load={m['daily_load']:.1f} kWh | "
            f"deviation={m['absolute_deviation']:.1f} kWh | "
            f"procurement={m['procurement_cost']:.2f} DKK | "
            f"disutility={m['disutility']:.2f} DKK"
        )

    plot_quadratic_disutility_sensitivity(
        results,
        data,
        save_to=out / "quadratic_disutility_sensitivity.png",
    )

    return results


def _run_q2_disutility_comparison(
    data_linear,
    data_quadratic,
    out: Path,
    tag: str,
):
    """Solve each Q2 model once; reuse Results.save and Q2 analysis helpers."""

    out.mkdir(parents=True, exist_ok=True)

    models = {
        "Linear": (data_linear, LinearDisutilityModel(data_linear)),
        "Quadratic": (data_quadratic, QuadraticDisutilityModel(data_quadratic)),
        "Temporal": (
            data_quadratic,
            TemporalDisutilityModel(data_quadratic, c0=0.5, cT=0.25),
        ),
    }

    runs = {}

    for label, (data, model) in models.items():
        result = model.build().solve()
        result.save(out, tag=f"{tag.lower()}_{label.lower()}")
        runs[label] = (data, result)

    temporal_model = models["Temporal"][1]
    metrics = compare_q2_disutility_models(
        runs, c0=temporal_model.c0, cT=temporal_model.cT
    )

    return {label: result for label, (_, result) in runs.items()}, metrics


def run_q2_model_comparison(out: Path):
    """Q2(d): reuse coefficient sweeps for base and three selected regimes.

    Coefficients are selected separately for their respective units. Equal
    scenario labels describe response regimes, not equal comfort preferences.
    """
    out.mkdir(parents=True, exist_ok=True)
    data_linear = load_question("Q2_linear")
    data_quadratic = load_question("Q2_quadratic")
    for field in ["energy_price", "reference_load", "pv_available", "pv_marginal_cost",
                  "load_min_kWh", "load_max_kWh", "import_tariff", "export_tariff"]:
        if not np.allclose(getattr(data_linear, field), getattr(data_quadratic, field)):
            raise ValueError(f"Q2 comparison physical inputs differ: {field}")
    for folder in [out / "linear_sweep", out / "quadratic_sweep"]:
        folder.mkdir(parents=True, exist_ok=True)
    linear = dict(run_linear_disutility_sweep(out / "linear_sweep"))
    quadratic = dict(run_quadratic_disutility_sweep(out / "quadratic_sweep"))
    selections = {
        "Base": (data_linear.linear_disutility, data_quadratic.quadratic_disutility),
        "Weak preferences": (0.5, 0.1),
        "Transition": (1.41, 0.5),
        "Strong preferences": (2.5, 20.0),
    }
    metrics = []
    for scenario, (cL, cQ) in selections.items():
        runs = {
            "Linear": (set_linear_disutility(data_linear, cL), linear[cL]),
            "Quadratic": (set_quadratic_disutility(data_quadratic, cQ), quadratic[cQ]),
        }
        folder = out / scenario.lower().replace(" ", "_")
        for label, (_, result) in runs.items():
            result.save(folder, tag=label.lower())
        for row in compare_q2_disutility_models(runs, c0=0.5, cT=0.25):
            metrics.append({"scenario": scenario, "cL": cL, "cQ": cQ, **row})
        plot_q2_bonus_base(
            {label: result for label, (_, result) in runs.items()}, data_quadratic,
            save_to=folder / "comparison.png",
            title=f"Q2(d): {scenario} — cL={cL:g} DKK/kWh, cQ={cQ:g} DKK/kWh²",
        )
    table = pd.DataFrame(metrics)
    table.to_csv(out / "comparison.csv", index=False)
    print("\nQ2(d) selected coefficient-sweep comparison")
    print(table.round(4).to_string(index=False))
    return table


def run_q2_bonus_base(out: Path):
    """Run Q2(e)'s supplied-data comparison against Q2(b) and Q2(c)."""

    data_linear = load_question("Q2_linear")
    data_quadratic = load_question("Q2_quadratic")

    # Confirm only the preference representation differs between base cases.
    for field in [
        "energy_price", "reference_load", "pv_available", "pv_marginal_cost",
        "load_min_kWh", "load_max_kWh", "import_tariff", "export_tariff",
    ]:
        if not np.allclose(
            getattr(data_linear, field), getattr(data_quadratic, field)
        ):
            raise ValueError(f"Q2 base physical inputs differ: {field}")

    if not np.isclose(data_quadratic.quadratic_disutility, 1.0):
        raise ValueError("The selected temporal parameters are calibrated to cQ = 1.")

    runs, metrics = _run_q2_disutility_comparison(
        data_linear, data_quadratic, out, tag="Base"
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "comparison.csv", index=False)

    plot_q2_bonus_base(
        runs, data_quadratic, save_to=out / "comparison.png"
    )

    print("\nQ2(e) base comparison")
    print(metrics_df.round(4).to_string(index=False))

    return metrics_df


def run_q2_price_structure_experiment(out: Path):
    """Compare alternating and clustered prices across the three Q2 models."""

    data_linear = load_question("Q2_linear")
    data_quadratic = load_question("Q2_quadratic")

    if data_quadratic.n_hours != 24:
        raise ValueError("The selected experiment uses 24 one-hour periods.")

    profiles = {
        "Alternating": np.tile([0.85, 2.50], 12),
        "Clustered": np.repeat([0.85, 2.50], 12),
    }

    if not np.array_equal(
        np.sort(profiles["Alternating"]), np.sort(profiles["Clustered"])
    ):
        raise ValueError("Price distributions must match exactly.")

    reference_kWh = float(data_quadratic.reference_load.mean())
    runs, metrics = {}, []

    for profile_name, prices in profiles.items():
        scenario_linear = set_q2_price_structure(
            data_linear, prices, reference_kWh
        )
        scenario_quadratic = set_q2_price_structure(
            data_quadratic, prices, reference_kWh
        )

        runs[profile_name], profile_metrics = _run_q2_disutility_comparison(
            scenario_linear, scenario_quadratic, out, tag=profile_name
        )

        for values in profile_metrics:
            metrics.append({"profile": profile_name, **values})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "comparison.csv", index=False)

    # Independent hourly models must be invariant to this price permutation.
    for model_name in ["Linear", "Quadratic"]:
        subset = metrics_df[metrics_df["model"] == model_name].set_index("profile")
        for metric in ["surplus", "absolute_deviation"]:
            if not np.isclose(
                subset.loc["Alternating", metric],
                subset.loc["Clustered", metric],
                atol=1e-5,
                rtol=1e-6,
            ):
                raise ValueError(f"Unexpected price-order effect: {model_name}, {metric}")

    plot_q2_price_structure(
        profiles,
        runs,
        np.full(24, reference_kWh),
        save_to=out / "comparison.png",
    )

    print("\nQ2(e) price-structure comparison")
    print(metrics_df.round(4).to_string(index=False))
    print("\nObjective-component and separable-model checks passed.")

    return metrics_df

def run_q3_comparison(out: Path):
    """Run and compare the Q2 quadratic and Q3 base cases."""

    data_q2 = load_question("Q2_quadratic")
    data_q3 = load_question("Q3")

    result_q2 = QuadraticDisutilityModel(data_q2).build().solve()
    result_q3 = DailyEnergyModel(data_q3).build().solve()

    comp_out = out / "q3_comparison"
    comp_out.mkdir(parents=True, exist_ok=True)

    result_q2.save(comp_out, tag="Q2_quadratic")
    result_q3.save(comp_out, tag="Q3")

    metrics = compare_q2_q3(
        data_q2,
        result_q2,
        data_q3,
        result_q3,
    )

    print("\nQ3 comparison")
    print(f"Q2 quadratic load      : {metrics['load_q2']:.2f} kWh")
    print(f"Q3 load                : {metrics['load_q3']:.2f} kWh")
    print(f"Q2 procurement cost    : {metrics['procurement_q2']:.2f} DKK")
    print(f"Q3 procurement cost    : {metrics['procurement_q3']:.2f} DKK")
    print(f"Q2 disutility          : {metrics['disutility_q2']:.2f} DKK")
    print(f"Q3 disutility          : {metrics['disutility_q3']:.2f} DKK")
    print(f"Q2 hours above ref     : {metrics['above_ref_q2']}")
    print(f"Q3 hours above ref     : {metrics['above_ref_q3']}")
    print(f"Q3 energy dual         : {metrics['energy_dual_q3']:.4f}")

    plot_q3_comparison(
        result_q2,
        result_q3,
        data_q3,
        save_to=comp_out / "q2_q3_load_comparison.png",
    )

    return metrics

def run_q3_battery_comparison(out: Path):
    """Compare Q3 without and with battery."""

    data_q3 = load_question("Q3")
    data_bat = load_question("Q3_battery")

    result_q3 = DailyEnergyModel(data_q3).build().solve()
    result_bat = BatteryModel(data_bat).build().solve()

    metrics = compare_q3_battery(
        data_q3,
        result_q3,
        data_bat,
        result_bat,
    )

    plot_q3_battery_comparison(
        result_q3,
        result_bat,
        data_bat,
        save_to=out / "q3_battery_comparison.png",
    )

    print("\nQ3 battery comparison")
    for key, value in metrics.items():
        print(f"{key:25s}: {value}")

    return metrics
def run_q3_energy_sweep(out: Path):
    """Run Q3 for different minimum daily energy requirements."""

    data = load_question("Q3")
    unconstrained_model = QuadraticDisutilityModel(data).build()
    unconstrained_model.m.Params.BarConvTol = 1e-12
    unconstrained_energy = float(unconstrained_model.solve().hourly["load"].sum())

    Emin_values = [
        15.0,
        20.0,
        22.0,
        22.76,
        22.77,
        25.0,
        30.0,

        # Reference-profile energy
        32.34,

        35.0,
        40.0,
        45.0,
        50.0,
    ]

    results = []

    for Emin in Emin_values:

        # Create Q3 scenario with modified minimum daily energy
        scenario_data = set_load_preferences(
            data,
            min_daily_energy_kWh=Emin,
        )

        # Resolve small energy slacks and duals near the activation threshold.
        model = DailyEnergyModel(scenario_data).build()
        model.m.Params.BarConvTol = 1e-12
        result = model.solve()

        results.append((Emin, result))

    # Analyse all sensitivity runs
    metrics = analyze_q3_energy_sensitivity(
        data,
        results,
    )

    pv_saturation_Emin = find_pv_saturation_Emin(
        data,
    )

    print(
        f"\nFull PV utilization reached at approximately "
        f"Emin={pv_saturation_Emin:.3f} kWh"
    )

    # Save numerical results
    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        out / "q3_energy_sensitivity.csv",
        index=False,
    )

    # Print summary
    print("\nQ3 minimum-energy sensitivity")

    for m in metrics:
        print(
            f"Emin={m['E_min']:5.2f} kWh | "
            f"load={m['daily_load']:5.2f} kWh | "
            f"objective={m['objective']:6.2f} DKK | "
            f"dual={m['energy_dual']:6.3f} | "
            f"import={m['import']:5.2f} kWh | "
            f"PV={m['pv']:5.2f} kWh"
        )

    # Plot sensitivity
    plot_q3_energy_sensitivity(
        metrics,
        data,
        save_to=out / "q3_energy_sensitivity.png",
        unconstrained_energy=unconstrained_energy,
    )

    plot_q3_energy_supply_sensitivity(
        metrics,
        data,
        pv_saturation_Emin=pv_saturation_Emin,
        save_to=out / "q3_energy_supply_sensitivity.png",
        unconstrained_energy=unconstrained_energy,
    )
    return results

def find_pv_saturation_Emin(
    data,
    lower: float = 35.0,
    upper: float = 40.0,
    energy_tol: float = 1e-3,
    pv_tol: float = 1e-5,
) -> float:
    """Find the minimum daily energy at which all available PV is used."""

    max_pv = float(data.pv_available.sum())

    while upper - lower > energy_tol:

        Emin = 0.5 * (lower + upper)

        scenario_data = set_load_preferences(
            data,
            min_daily_energy_kWh=Emin,
        )

        result = DailyEnergyModel(
            scenario_data
        ).build().solve()

        pv_used = float(result.hourly["pv"].sum())

        if max_pv - pv_used <= pv_tol:
            upper = Emin
        else:
            lower = Emin

    return upper

def run_q3_cq_sweep(out: Path):
    """Run Q3 for different quadratic disutility coefficients."""

    data = load_question("Q3")

    cQ_values = [
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
    ]

    results = []

    for cQ in cQ_values:

        scenario_data = set_quadratic_disutility(
            data,
            cQ,
        )

        result = DailyEnergyModel(
            scenario_data
        ).build().solve()

        results.append((cQ, result))

    metrics = analyze_q3_cq_sensitivity(
        data,
        results,
    )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        out / "q3_cq_sensitivity.csv",
        index=False,
    )

    print("\nQ3 quadratic-disutility sensitivity")

    for m in metrics:
        print(
            f"cQ={m['cQ']:6.3f} | "
            f"load={m['daily_load']:5.2f} kWh | "
            f"objective={m['objective']:7.2f} DKK | "
            f"deviation={m['absolute_deviation']:5.2f} kWh | "
            f"procurement={m['procurement_cost']:6.2f} DKK | "
            f"dual={m['energy_dual']:6.3f}"
        )

    plot_q3_cq_sensitivity(
        metrics,
        save_to=out / "q3_cq_sensitivity.png",
    )

    return results



def run_q3_battery_sensitivity(out: Path):
    """Sensitivity of battery value to price variation and capacity."""

    base_no_bat = load_question("Q3")
    base_bat = load_question("Q3_battery")

    # ---------------------------------------------------------
    # 1. Price variation: scale spread while keeping mean fixed
    # ---------------------------------------------------------
    price_factors = [0.0, 0.5, 1.0, 1.5, 2.0]
    price_results = []

    for factor in price_factors:
        data_no_bat = scale_prices(base_no_bat, factor, keep_mean=True)
        data_bat = scale_prices(base_bat, factor, keep_mean=True)

        result_no_bat = DailyEnergyModel(data_no_bat).build().solve()
        result_bat = BatteryModel(data_bat).build().solve()

        price_results.append({
            "factor": factor,
            "battery_value": result_no_bat.objective - result_bat.objective,
        })

    # ---------------------------------------------------------
    # 2. Battery capacity
    # ---------------------------------------------------------
    capacities = [1.0, 2.0, 4.0, 6.0, 8.0]
    capacity_results = []

    # No-battery case is identical for all capacities
    result_no_bat = DailyEnergyModel(base_no_bat).build().solve()

    for capacity in capacities:
        data_bat = set_battery_capacity(base_bat, capacity)
        result_bat = BatteryModel(data_bat).build().solve()

        capacity_results.append({
            "capacity": capacity,
            "battery_value": result_no_bat.objective - result_bat.objective,
        })

    # Save results
    pd.DataFrame(price_results).to_csv(
        out / "battery_price_sensitivity.csv", index=False
    )
    pd.DataFrame(capacity_results).to_csv(
        out / "battery_capacity_sensitivity.csv", index=False
    )

    print("\nBattery value - price variation")
    for r in price_results:
        print(
            f"spread={r['factor']:4.1f} | "
            f"value={r['battery_value']:6.3f} DKK/day"
        )

    print("\nBattery value - capacity")
    for r in capacity_results:
        print(
            f"capacity={r['capacity']:4.1f} kWh | "
            f"value={r['battery_value']:6.3f} DKK/day"
        )

    plot_battery_value_sensitivity(
    price_results,
    "factor",
    "Price variation factor",
    base_x=1.0,
    save_to=out / "battery_price_sensitivity.png",
)

    plot_battery_value_sensitivity(
    capacity_results,
    "capacity",
    "Battery capacity [kWh]",
    base_x=4.0,
    save_to=out / "battery_capacity_sensitivity.png",
    )

    return price_results, capacity_results

def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: net cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_validation(out: Path):
    """Validate base cases, regime extremes, ties, temporal controls and Q3 duals."""
    out.mkdir(parents=True, exist_ok=True)
    cases = [(q, load_question(q), cls, {}) for q, cls in [
        ("Q1_caseA", FlexibleConsumerModel), ("Q1_caseB", FlexibleConsumerModel),
        ("Q2_linear", LinearDisutilityModel), ("Q2_quadratic", QuadraticDisutilityModel),
        ("Q3", DailyEnergyModel), ("Q3_battery", BatteryModel)]]
    for c in [0.5, 1.41, 3.0]:
        cases.append((f"linear_{c}", set_linear_disutility(load_question("Q2_linear"), c), LinearDisutilityModel, {}))
    for c in [0.01, 20.0]:
        cases.append((f"quadratic_{c}", set_quadratic_disutility(load_question("Q2_quadratic"), c), QuadraticDisutilityModel, {}))
    for energy in [20.0, 22.76, 22.77, 50.0]:
        cases.append((f"energy_{energy}", set_load_preferences(load_question("Q3"), min_daily_energy_kWh=energy), DailyEnergyModel, {}))
    for c in [0.01, 20.0]:
        cases.append((f"daily_quadratic_{c}", set_quadratic_disutility(load_question("Q3"), c), DailyEnergyModel, {}))
    for factor in [0.0, 2.0]:
        cases.append((f"battery_spread_{factor}", scale_prices(load_question("Q3_battery"), factor, True), BatteryModel, {}))
    for capacity in [1.0, 8.0]:
        cases.append((f"battery_capacity_{capacity}", set_battery_capacity(load_question("Q3_battery"), capacity), BatteryModel, {}))
    cases.append(("temporal_base", load_question("Q2_quadratic"), TemporalDisutilityModel, {}))
    cases.append(("temporal_uncoupled", load_question("Q2_quadratic"), TemporalDisutilityModel, {"c0": 1.0, "cT": 0.0}))
    d = load_question("Q1_caseA")
    tied = scale_pv(replace(d, energy_price=np.full(d.n_hours, d.consumption_utility - d.import_tariff)), 0)
    cases.append(("Q1_import_tie", tied, FlexibleConsumerModel, {}))
    for label, lower, upper in [("PV_below_min", 2.0, 6.0), ("PV_above_max", 0.0, 1.0)]:
        cases.append((label, set_load_preferences(d, load_min_kWh=lower, load_max_kWh=upper), FlexibleConsumerModel, {}))
    profiles = {"alternating": np.tile([0.85, 2.5], 12), "clustered": np.repeat([0.85, 2.5], 12)}
    for profile, prices in profiles.items():
        for label, question, cls in [("linear", "Q2_linear", LinearDisutilityModel),
                                     ("quadratic", "Q2_quadratic", QuadraticDisutilityModel),
                                     ("temporal", "Q2_quadratic", TemporalDisutilityModel)]:
            source = load_question(question)
            cases.append((f"{profile}_{label}", set_q2_price_structure(source, prices, float(source.reference_load.mean())), cls, {}))
    d = load_question("Q3")
    step = 1e-3
    for label, change in [("dual_plus", step), ("dual_minus", -step)]:
        cases.append((label, set_load_preferences(d, min_daily_energy_kWh=d.min_daily_energy_kWh + change), DailyEnergyModel, {}))
    solved, rows = {}, []
    for label, data, cls, kwargs in cases:
        result = cls(data, **kwargs).build().solve()
        result.save(out / label)
        solved[label] = result
        rows.append({"scenario": label, "objective": result.objective, **result.meta["validation"]})
    finite_difference = (solved["dual_plus"].objective - solved["dual_minus"].objective) / (2 * step)
    if not np.isclose(finite_difference, solved["Q3"].duals["min_daily_energy"], atol=1e-4, rtol=0):
        raise ValueError("Daily-energy dual does not match objective sensitivity.")
    for label in ["linear", "quadratic"]:
        a, b = solved[f"alternating_{label}"], solved[f"clustered_{label}"]
        if not np.isclose(a.objective, b.objective, atol=1e-5, rtol=0) or not np.isclose(
            abs(a.hourly.load - a.hourly.reference_load).sum(),
            abs(b.hourly.load - b.hourly.reference_load).sum(), atol=1e-5, rtol=0):
            raise ValueError("Separable models changed under a controlled price permutation.")
    if not np.isclose(solved["temporal_uncoupled"].objective, solved["Q2_quadratic"].objective, atol=1e-5, rtol=0):
        raise ValueError("Uncoupled temporal model does not recover quadratic disutility.")
    if not any(row["nonunique_load"] for row in solved["Q1_import_tie"].meta["analytical_hourly"]):
        raise ValueError("Analytical validator failed to identify the import-price tie.")
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation.csv", index=False)
    (out / "checks.json").write_text(json.dumps({"solves": len(cases), "energy_dual_finite_difference": finite_difference,
                                               "price_permutation": "passed", "uncoupled_temporal": "passed", "linear_tie": "passed"}, indent=2) + "\n")
    print(f"Validation passed for {len(cases)} solves; analytical, physical and dual checks passed.")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    parser.add_argument("--q2-bonus", action="store_true", help="run Q2(e) base and price-structure comparisons")
    parser.add_argument("--q2-comparison", action="store_true", help="run Q2(d) selected coefficient-sweep comparisons")
    parser.add_argument("--validate", action="store_true", help="run analytical, physical and sensitivity sanity checks")
    args = parser.parse_args()


    if args.validate:
        run_validation(RESULTS_DIR / "validation")
        return

    if args.q2_comparison:
        if not args.show:
            matplotlib.use("Agg")
        out = RESULTS_DIR / "Q2_comparison"
        run_q2_model_comparison(out)
        if args.show:
            matplotlib.pyplot.show()
        print(f"\nOutputs written to {out}")
        return

    if args.q2_bonus:
        if not args.show:
            matplotlib.use("Agg")
        out = RESULTS_DIR / "Q2_bonus"
        run_q2_bonus_base(out / "base")
        run_q2_price_structure_experiment(out / "price_structure")
        if args.show:
            matplotlib.pyplot.show()
        print(f"\nOutputs written to {out}")
        return

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)

    # Q3 requires comparison with the unconstrained Q2(c) case
    if args.question == "Q3" and base is not None:
        run_q3_comparison(out)
    if args.question == "Q3_battery" and base is not None:
        run_q3_battery_comparison(out)

    if args.scenarios and base is not None:

        if args.question == "Q2_linear":
            run_linear_disutility_sweep(out)

        elif args.question == "Q2_quadratic":
            run_quadratic_disutility_sweep(out)

        elif args.question == "Q3":
            run_q3_energy_sweep(out)
            run_q3_cq_sweep(out)

        elif args.question == "Q3_battery":
            run_q3_battery_sensitivity(out)

        else:
            run_scenarios(args.question, out)

    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
