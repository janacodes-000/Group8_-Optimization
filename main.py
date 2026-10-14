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
import pandas as pd
import numpy as np

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import (
    FlexibleConsumerModel,
    LinearDisutilityModel,
    QuadraticDisutilityModel,
    DailyEnergyModel,
    BatteryModel,
    Results,
    TemporalDisutilityModel,
)
from src.plotting import (
    plot_duals,
    plot_inputs,
    plot_scenario_comparison,
    plot_schedule,
    plot_linear_disutility_sensitivity,
    plot_q3_comparison,
    plot_quadratic_disutility_sensitivity,
    plot_q3_energy_sensitivity,
    plot_q3_energy_supply_sensitivity,
    plot_q3_cq_sensitivity,
    plot_q3_battery_comparison,
    plot_battery_value_sensitivity,
    plot_q2_bonus_base,
    plot_q2_price_structure,
)
from src.scenarios import (
    scale_prices,
    scale_pv,
    set_tariffs,
    set_linear_disutility,
    set_quadratic_disutility,
    set_load_preferences,
    set_battery_capacity,
    set_q2_price_structure,
)
from src.analysis import (
    compare_q2_q3,
    analyze_linear_sensitivity,
    analyze_quadratic_sensitivity,
    analyze_q3_energy_sensitivity,
    analyze_q3_cq_sensitivity,
    compare_q3_battery,
    compare_q2_disutility_models,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Report experiments are declared once; runners own solving/analysis/exports.
MODEL_CLASSES = {
    "Q1_caseA": FlexibleConsumerModel,
    "Q1_caseB": FlexibleConsumerModel,
    "Q2_linear": LinearDisutilityModel,
    "Q2_quadratic": QuadraticDisutilityModel,
    "Q3": DailyEnergyModel,
    "Q3_battery": BatteryModel,
}
SWEEPS = {
    "linear": [
        0.5,
        1.0,
        1.2,
        1.3,
        1.4,
        1.409,
        1.41,
        1.411,
        1.43,
        1.5,
        1.75,
        2.0,
        2.5,
        3.0,
    ],
    "quadratic": [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "energy": [
        15.0,
        20.0,
        22.0,
        22.76,
        22.77,
        25.0,
        30.0,
        32.34,
        35.0,
        40.0,
        45.0,
        50.0,
    ],
    "daily_quadratic": [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "battery_spread": [0.0, 0.5, 1.0, 1.5, 2.0],
    "battery_capacity": [1.0, 2.0, 4.0, 6.0, 8.0],
}
Q2_COMPARISON_SELECTIONS = {
    "Weak preferences": (0.5, 0.1),
    "Transition": (1.41, 0.5),
    "Strong preferences": (2.5, 20.0),
}
PRICE_PROFILES = {
    "Alternating": np.tile([0.85, 2.50], 12),
    "Clustered": np.repeat([0.85, 2.50], 12),
}
Q1_SCENARIOS = {
    "base": (None, {}),
    "flat_prices": (scale_prices, {"factor": 0.0, "keep_mean": True}),
    "double_spread": (scale_prices, {"factor": 2.0, "keep_mean": True}),
    "no_tariffs": (set_tariffs, {"import_tariff": 0.0, "export_tariff": 0.0}),
    "no_pv": (scale_pv, {"factor": 0.0}),
}


def _solve_sweep(data, model_class, values, transform, out, tag, existing=None):
    """Apply a declared sweep and save each run, reusing already solved base cases."""
    existing = existing or {}
    runs = []
    for value in values:
        result = existing.get(value)
        if result is None:
            result = model_class(transform(data, value)).build().solve()
        result.save(out, tag=f"{tag}_{value:g}")
        runs.append((value, result))
    return runs


def run_base_case(question: str, out: Path, show: bool = False) -> Results:
    """Solve and save one supplied case; figure display is handled after experiments."""
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = MODEL_CLASSES[question](data).build()
    results = model.solve()

    print(results, "\n")
    if results.metrics:
        print(pd.Series(results.metrics).to_string(), "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    return results


def run_linear_disutility_sweep(out: Path, base_result: Results | None = None):
    """Run Q2 linear for different values of the linear disutility coefficient."""

    data = load_question("Q2_linear")

    cL_values = SWEEPS["linear"]

    results = _solve_sweep(
        data,
        LinearDisutilityModel,
        cL_values,
        set_linear_disutility,
        out / "runs",
        "cL",
        {data.linear_disutility: base_result} if base_result is not None else {},
    )

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


def run_quadratic_disutility_sweep(out: Path, base_result: Results | None = None):
    """Run Q2 quadratic for different values of the quadratic disutility coefficient."""

    data = load_question("Q2_quadratic")

    cQ_values = SWEEPS["quadratic"]

    results = _solve_sweep(
        data,
        QuadraticDisutilityModel,
        cQ_values,
        set_quadratic_disutility,
        out / "runs",
        "cQ",
        {data.quadratic_disutility: base_result} if base_result is not None else {},
    )

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
    for field in [
        "energy_price",
        "reference_load",
        "pv_available",
        "pv_marginal_cost",
        "load_min_kWh",
        "load_max_kWh",
        "import_tariff",
        "export_tariff",
    ]:
        if not np.allclose(getattr(data_linear, field), getattr(data_quadratic, field)):
            raise ValueError(f"Q2 comparison physical inputs differ: {field}")
    for folder in [out / "linear_sweep", out / "quadratic_sweep"]:
        folder.mkdir(parents=True, exist_ok=True)
    linear = dict(run_linear_disutility_sweep(out / "linear_sweep"))
    quadratic = dict(run_quadratic_disutility_sweep(out / "quadratic_sweep"))
    selections = {
        "Base": (data_linear.linear_disutility, data_quadratic.quadratic_disutility),
        **Q2_COMPARISON_SELECTIONS,
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
            {label: result for label, (_, result) in runs.items()},
            data_quadratic,
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
        "energy_price",
        "reference_load",
        "pv_available",
        "pv_marginal_cost",
        "load_min_kWh",
        "load_max_kWh",
        "import_tariff",
        "export_tariff",
    ]:
        if not np.allclose(getattr(data_linear, field), getattr(data_quadratic, field)):
            raise ValueError(f"Q2 base physical inputs differ: {field}")

    if not np.isclose(data_quadratic.quadratic_disutility, 1.0):
        raise ValueError("The selected temporal parameters are calibrated to cQ = 1.")

    runs, metrics = _run_q2_disutility_comparison(
        data_linear, data_quadratic, out, tag="Base"
    )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "comparison.csv", index=False)

    plot_q2_bonus_base(runs, data_quadratic, save_to=out / "comparison.png")

    print("\nQ2(e) base comparison")
    print(metrics_df.round(4).to_string(index=False))

    return metrics_df


def run_q2_price_structure_experiment(out: Path):
    """Compare alternating and clustered prices across the three Q2 models."""

    data_linear = load_question("Q2_linear")
    data_quadratic = load_question("Q2_quadratic")

    if data_quadratic.n_hours != 24:
        raise ValueError("The selected experiment uses 24 one-hour periods.")

    profiles = PRICE_PROFILES

    if not np.array_equal(
        np.sort(profiles["Alternating"]), np.sort(profiles["Clustered"])
    ):
        raise ValueError("Price distributions must match exactly.")

    reference_kWh = float(data_quadratic.reference_load.mean())
    runs, metrics = {}, []

    for profile_name, prices in profiles.items():
        scenario_linear = set_q2_price_structure(data_linear, prices, reference_kWh)
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
                raise ValueError(
                    f"Unexpected price-order effect: {model_name}, {metric}"
                )

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


def run_q3_comparison(out: Path, result_q3=None, result_q2=None):
    """Run and compare the Q2 quadratic and Q3 base cases."""

    data_q2 = load_question("Q2_quadratic")
    data_q3 = load_question("Q3")

    if result_q2 is None:
        result_q2 = QuadraticDisutilityModel(data_q2).build().solve()
    if result_q3 is None:
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
    pd.DataFrame([metrics]).to_csv(comp_out / "comparison.csv", index=False)

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


def run_q3_battery_comparison(out: Path, result_bat=None, result_q3=None):
    """Compare Q3 without and with battery."""

    data_q3 = load_question("Q3")
    data_bat = load_question("Q3_battery")

    if result_q3 is None:
        result_q3 = DailyEnergyModel(data_q3).build().solve()
    if result_bat is None:
        result_bat = BatteryModel(data_bat).build().solve()

    metrics = compare_q3_battery(
        data_q3,
        result_q3,
        data_bat,
        result_bat,
    )
    comp_out = out / "battery_comparison"
    result_q3.save(comp_out, tag="without_battery")
    result_bat.save(comp_out, tag="with_battery")
    pd.DataFrame([metrics]).to_csv(comp_out / "comparison.csv", index=False)

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


def run_q3_energy_sweep(
    out: Path,
    base_result: Results | None = None,
    unconstrained_result: Results | None = None,
):
    """Run Q3 for different minimum daily energy requirements."""

    data = load_question("Q3")
    if unconstrained_result is None:
        unconstrained_result = QuadraticDisutilityModel(data).build().solve()
    unconstrained_energy = float(unconstrained_result.hourly["load"].sum())

    Emin_values = SWEEPS["energy"]

    results = _solve_sweep(
        data,
        DailyEnergyModel,
        Emin_values,
        lambda d, value: set_load_preferences(d, min_daily_energy_kWh=value),
        out / "energy_runs",
        "Emin",
        {data.min_daily_energy_kWh: base_result} if base_result is not None else {},
    )

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

        result = DailyEnergyModel(scenario_data).build().solve()

        pv_used = float(result.hourly["pv"].sum())

        if max_pv - pv_used <= pv_tol:
            upper = Emin
        else:
            lower = Emin

    return upper


def run_q3_cq_sweep(out: Path, base_result: Results | None = None):
    """Run Q3 for different quadratic disutility coefficients."""

    data = load_question("Q3")

    cQ_values = SWEEPS["daily_quadratic"]

    results = _solve_sweep(
        data,
        DailyEnergyModel,
        cQ_values,
        set_quadratic_disutility,
        out / "cq_runs",
        "cQ",
        {data.quadratic_disutility: base_result} if base_result is not None else {},
    )

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


def run_q3_battery_sensitivity(
    out: Path, base_no_bat_result=None, base_bat_result=None
):
    """Sensitivity of battery value to price variation and capacity."""

    base_no_bat = load_question("Q3")
    base_bat = load_question("Q3_battery")

    # ---------------------------------------------------------
    # 1. Price variation: scale spread while keeping mean fixed
    # ---------------------------------------------------------
    price_factors = SWEEPS["battery_spread"]
    price_results = []

    for factor in price_factors:
        data_no_bat = scale_prices(base_no_bat, factor, keep_mean=True)
        data_bat = scale_prices(base_bat, factor, keep_mean=True)

        result_no_bat = base_no_bat_result if factor == 1.0 else None
        result_bat = base_bat_result if factor == 1.0 else None
        if result_no_bat is None:
            result_no_bat = DailyEnergyModel(data_no_bat).build().solve()
        if result_bat is None:
            result_bat = BatteryModel(data_bat).build().solve()

        folder = out / "price_runs" / f"spread_{factor:g}"
        result_no_bat.save(folder, tag="without_battery")
        result_bat.save(folder, tag="with_battery")

        price_results.append(
            {
                "factor": factor,
                "objective_without_battery": result_no_bat.objective,
                "objective_with_battery": result_bat.objective,
                "battery_value": result_no_bat.objective - result_bat.objective,
                "load_without_battery": float(result_no_bat.hourly.load.sum()),
                "load_with_battery": float(result_bat.hourly.load.sum()),
                "simultaneous_hours": result_bat.meta["validation"][
                    "simultaneous_battery_hours"
                ],
            }
        )

    # ---------------------------------------------------------
    # 2. Battery capacity
    # ---------------------------------------------------------
    capacities = SWEEPS["battery_capacity"]
    capacity_results = []

    # No-battery case is identical for all capacities
    result_no_bat = base_no_bat_result
    if result_no_bat is None:
        result_no_bat = DailyEnergyModel(base_no_bat).build().solve()
    result_no_bat.save(out / "capacity_runs", tag="without_battery")

    for capacity in capacities:
        data_bat = set_battery_capacity(base_bat, capacity)
        result_bat = (
            base_bat_result if capacity == base_bat.battery_capacity_kWh else None
        )
        if result_bat is None:
            result_bat = BatteryModel(data_bat).build().solve()
        result_bat.save(
            out / "capacity_runs" / f"capacity_{capacity:g}", tag="with_battery"
        )

        capacity_results.append(
            {
                "capacity": capacity,
                "objective_without_battery": result_no_bat.objective,
                "objective_with_battery": result_bat.objective,
                "battery_value": result_no_bat.objective - result_bat.objective,
                "load_without_battery": float(result_no_bat.hourly.load.sum()),
                "load_with_battery": float(result_bat.hourly.load.sum()),
                "simultaneous_hours": result_bat.meta["validation"][
                    "simultaneous_battery_hours"
                ],
            }
        )

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
            f"spread={r['factor']:4.1f} | " f"value={r['battery_value']:6.3f} DKK/day"
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


def run_scenarios(
    question: str, out: Path, base_result: Results | None = None
) -> dict[str, Results]:
    """Run declared Q1 price/tariff/PV controls and reuse the base solution."""
    base = load_question(question)
    scenarios = {
        name: base if transform is None else transform(base, **kwargs)
        for name, (transform, kwargs) in Q1_SCENARIOS.items()
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = base_result if name == "base" else None
        if results is None:
            results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(
            f"{name:>14}: net cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
            f" | export {results.hourly['export'].sum():5.1f} kWh"
        )
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_validation(out: Path):
    """Validate base cases, regime extremes, ties, temporal controls and Q3 duals."""
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        (q, load_question(q), cls, {})
        for q, cls in [
            ("Q1_caseA", FlexibleConsumerModel),
            ("Q1_caseB", FlexibleConsumerModel),
            ("Q2_linear", LinearDisutilityModel),
            ("Q2_quadratic", QuadraticDisutilityModel),
            ("Q3", DailyEnergyModel),
            ("Q3_battery", BatteryModel),
        ]
    ]
    for c in [0.5, 1.41, 3.0]:
        cases.append(
            (
                f"linear_{c}",
                set_linear_disutility(load_question("Q2_linear"), c),
                LinearDisutilityModel,
                {},
            )
        )
    for c in [0.01, 20.0]:
        cases.append(
            (
                f"quadratic_{c}",
                set_quadratic_disutility(load_question("Q2_quadratic"), c),
                QuadraticDisutilityModel,
                {},
            )
        )
    for energy in [20.0, 22.76, 22.77, 50.0]:
        cases.append(
            (
                f"energy_{energy}",
                set_load_preferences(load_question("Q3"), min_daily_energy_kWh=energy),
                DailyEnergyModel,
                {},
            )
        )
    for c in [0.01, 20.0]:
        cases.append(
            (
                f"daily_quadratic_{c}",
                set_quadratic_disutility(load_question("Q3"), c),
                DailyEnergyModel,
                {},
            )
        )
    for factor in [0.0, 2.0]:
        cases.append(
            (
                f"battery_spread_{factor}",
                scale_prices(load_question("Q3_battery"), factor, True),
                BatteryModel,
                {},
            )
        )
    for capacity in [1.0, 8.0]:
        cases.append(
            (
                f"battery_capacity_{capacity}",
                set_battery_capacity(load_question("Q3_battery"), capacity),
                BatteryModel,
                {},
            )
        )
    cases.append(
        ("temporal_base", load_question("Q2_quadratic"), TemporalDisutilityModel, {})
    )
    cases.append(
        (
            "temporal_uncoupled",
            load_question("Q2_quadratic"),
            TemporalDisutilityModel,
            {"c0": 1.0, "cT": 0.0},
        )
    )
    d = load_question("Q1_caseA")
    tied = scale_pv(
        replace(
            d, energy_price=np.full(d.n_hours, d.consumption_utility - d.import_tariff)
        ),
        0,
    )
    cases.append(("Q1_import_tie", tied, FlexibleConsumerModel, {}))
    for label, lower, upper in [("PV_below_min", 2.0, 6.0), ("PV_above_max", 0.0, 1.0)]:
        cases.append(
            (
                label,
                set_load_preferences(d, load_min_kWh=lower, load_max_kWh=upper),
                FlexibleConsumerModel,
                {},
            )
        )
    profiles = {name.lower(): prices for name, prices in PRICE_PROFILES.items()}
    for profile, prices in profiles.items():
        for label, question, cls in [
            ("linear", "Q2_linear", LinearDisutilityModel),
            ("quadratic", "Q2_quadratic", QuadraticDisutilityModel),
            ("temporal", "Q2_quadratic", TemporalDisutilityModel),
        ]:
            source = load_question(question)
            cases.append(
                (
                    f"{profile}_{label}",
                    set_q2_price_structure(
                        source, prices, float(source.reference_load.mean())
                    ),
                    cls,
                    {},
                )
            )
    d = load_question("Q3")
    step = 1e-3
    for label, change in [("dual_plus", step), ("dual_minus", -step)]:
        cases.append(
            (
                label,
                set_load_preferences(
                    d, min_daily_energy_kWh=d.min_daily_energy_kWh + change
                ),
                DailyEnergyModel,
                {},
            )
        )
    solved, rows = {}, []
    for label, data, cls, kwargs in cases:
        result = cls(data, **kwargs).build().solve()
        result.save(out / label)
        solved[label] = result
        rows.append(
            {
                "scenario": label,
                "objective": result.objective,
                **result.meta["validation"],
            }
        )
    finite_difference = (
        solved["dual_plus"].objective - solved["dual_minus"].objective
    ) / (2 * step)
    if not np.isclose(
        finite_difference, solved["Q3"].duals["min_daily_energy"], atol=1e-4, rtol=0
    ):
        raise ValueError("Daily-energy dual does not match objective sensitivity.")
    for label in ["linear", "quadratic"]:
        a, b = solved[f"alternating_{label}"], solved[f"clustered_{label}"]
        if not np.isclose(
            a.objective, b.objective, atol=1e-5, rtol=0
        ) or not np.isclose(
            abs(a.hourly.load - a.hourly.reference_load).sum(),
            abs(b.hourly.load - b.hourly.reference_load).sum(),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError(
                "Separable models changed under a controlled price permutation."
            )
    if not np.isclose(
        solved["temporal_uncoupled"].objective,
        solved["Q2_quadratic"].objective,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError(
            "Uncoupled temporal model does not recover quadratic disutility."
        )
    if not any(
        row["nonunique_load"]
        for row in solved["Q1_import_tie"].meta["analytical_hourly"]
    ):
        raise ValueError(
            "Analytical validator failed to identify the import-price tie."
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation.csv", index=False)
    (out / "checks.json").write_text(
        json.dumps(
            {
                "solves": len(cases),
                "energy_dual_finite_difference": finite_difference,
                "price_permutation": "passed",
                "uncoupled_temporal": "passed",
                "linear_tie": "passed",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Validation passed for {len(cases)} solves; analytical, physical and dual checks passed."
    )
    return table


def main() -> None:
    """Run a documented report experiment or the numerical validation suite."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--question",
        default="Q1_caseA",
        choices=list_questions(),
        help="data case to use",
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="also run the example sensitivity scenarios",
    )
    parser.add_argument(
        "--show", action="store_true", help="open the figures in a window"
    )
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument(
        "--q2-bonus",
        action="store_true",
        help="run Q2(e) base and price-structure comparisons",
    )
    experiments.add_argument(
        "--q2-comparison",
        action="store_true",
        help="run Q2(d) selected coefficient-sweep comparisons",
    )
    experiments.add_argument(
        "--validate",
        action="store_true",
        help="run analytical, physical and sensitivity sanity checks",
    )
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

    unconstrained_result = no_battery_result = None
    # Q3 requires comparison with the unconstrained Q2(c) case
    if args.question == "Q3":
        unconstrained_result = (
            QuadraticDisutilityModel(load_question("Q2_quadratic")).build().solve()
        )
        run_q3_comparison(out, result_q3=base, result_q2=unconstrained_result)
    if args.question == "Q3_battery":
        no_battery_result = DailyEnergyModel(load_question("Q3")).build().solve()
        run_q3_battery_comparison(out, result_bat=base, result_q3=no_battery_result)

    if args.scenarios:

        if args.question == "Q2_linear":
            run_linear_disutility_sweep(out, base_result=base)

        elif args.question == "Q2_quadratic":
            run_quadratic_disutility_sweep(out, base_result=base)

        elif args.question == "Q3":
            run_q3_energy_sweep(
                out, base_result=base, unconstrained_result=unconstrained_result
            )
            run_q3_cq_sweep(out, base_result=base)

        elif args.question == "Q3_battery":
            run_q3_battery_sensitivity(
                out, base_no_bat_result=no_battery_result, base_bat_result=base
            )

        else:
            run_scenarios(args.question, out, base_result=base)

    if args.show:
        matplotlib.pyplot.show()
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
