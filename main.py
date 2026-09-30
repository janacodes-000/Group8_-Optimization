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
#from turtle import pd
import pandas as pd

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, DailyEnergyModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_linear_disutility_sensitivity,plot_q3_comparison, plot_quadratic_disutility_sensitivity, plot_q3_energy_sensitivity, plot_q3_energy_supply_sensitivity, plot_q3_cq_sensitivity 
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_linear_disutility, set_quadratic_disutility, set_load_preferences
from src.analysis import compare_q2_q3, analyze_linear_sensitivity, analyze_quadratic_sensitivity, analyze_q3_energy_sensitivity, analyze_q3_cq_sensitivity

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
    else:
        model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
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

def run_q3_energy_sweep(out: Path):
    """Run Q3 for different minimum daily energy requirements."""

    data = load_question("Q3")

    Emin_values = [
        15.0,
        20.0,
        22.0,

        # Fine resolution around unconstrained optimum
        22.70,
        22.72,
        22.74,
        22.76,
        22.77,
        22.78,
        22.80,
        22.82,
        22.85,
        22.90,

        23.0,
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

        # Solve Q3 model
        result = DailyEnergyModel(
            scenario_data
        ).build().solve()

        results.append((Emin, result))

    # Analyse all sensitivity runs
    metrics = analyze_q3_energy_sensitivity(
        data,
        results,
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
    )

    plot_q3_energy_supply_sensitivity(
        metrics,
        data,
        save_to=out / "q3_energy_supply_sensitivity.png",
    )
    return results

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
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)

    # Q3 requires comparison with the unconstrained Q2(c) case
    if args.question == "Q3" and base is not None:
        run_q3_comparison(out)

    if args.scenarios and base is not None:

        if args.question == "Q2_linear":
            run_linear_disutility_sweep(out)

        elif args.question == "Q2_quadratic":
            run_quadratic_disutility_sweep(out)

        elif args.question == "Q3":
            run_q3_energy_sweep(out)
            run_q3_cq_sweep(out)

        else:
            run_scenarios(args.question, out)

    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
