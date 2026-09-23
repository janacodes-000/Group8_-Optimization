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

import matplotlib

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel, DailyEnergyModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_linear_disutility_sensitivity,plot_q3_comparison
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_linear_disutility
from src.analysis import compare_q2_q3

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

    # Load the original Q2 linear data
    data = load_question("Q2_linear")

    # Create a separate folder for the sensitivity results
    sweep_out = out / "linear_disutility_sweep"
    sweep_out.mkdir(parents=True, exist_ok=True)

    # Values of cL to test
    cL_values = [0.5, 1.0, 1.2, 1.3, 1.4, 1.409, 1.41, 1.411, 1.43, 1.5, 1.75, 2.0, 2.5, 3.0]

    results = []

    for cL in cL_values:
        # Change only the linear disutility coefficient
        scenario_data = set_linear_disutility(data, cL)

        # Build and solve the Q2 linear model
        model = LinearDisutilityModel(scenario_data).build()
        result = model.solve()

        # Save the detailed results without overwriting the base case
        result.save(sweep_out, tag=f"cL_{cL:.2f}")

        results.append((cL, result))

        # Daily quantities for the report
        daily_load = result.hourly["load"].sum()

        absolute_deviation = abs(
            result.hourly["load"] - scenario_data.reference_load
        ).sum()

        total_disutility = cL * absolute_deviation

        procurement_cost = (
            scenario_data.pv_marginal_cost * result.hourly["pv"]
            + (scenario_data.energy_price + scenario_data.import_tariff)
            * result.hourly["import"]
            - (scenario_data.energy_price - scenario_data.export_tariff)
            * result.hourly["export"]
        ).sum()

        binding_hours = (
            abs(result.hourly["load"] - scenario_data.reference_load) > 1e-6
        ).sum()

        print(
            f"cL={cL:.3f} | "
            f"procurement={procurement_cost:.2f} DKK | "
            f"disutility={total_disutility:.2f} DKK | "
            f"load={daily_load:.1f} kWh | "
            f"deviation={absolute_deviation:.1f} kWh | "
            f"binding={binding_hours} h"
        )

    plot_linear_disutility_sensitivity(
    results,
    data,
    save_to=sweep_out / "linear_disutility_sensitivity.png",
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
    if args.scenarios and base is not None:
            if args.question == "Q2_linear":
                run_linear_disutility_sweep(out)
            else:
                run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
