"""Post-processing and comparison helpers for Assignment 1.

This module contains functions that analyse optimisation results after a model
has been solved. It should not build optimisation models or modify the input
data.

Typical tasks include:
- calculating procurement cost and disutility from a solved model;
- comparing the results of different questions or scenarios;
- extracting summary metrics used in the report;
- checking whether qualitative hypotheses are supported by the numerical results.

The functions here operate on ``InputData`` and ``Results`` objects and return
plain dictionaries or scalar values that can be printed, saved, or passed to
plotting functions.

Example
-------
>>> from src.analysis import compare_q2_q3
>>> metrics = compare_q2_q3(data_q2, result_q2, data_q3, result_q3)
>>> print(metrics["load_q3"])
40.0
"""

from __future__ import annotations

from .data_loader import InputData
from .model import Results

def calculate_procurement_cost(data: InputData, results: Results) -> float:
    """Return daily electricity procurement cost in DKK.

    Includes PV production cost and grid import cost, minus grid export revenue.
    """

    return float(
        (
            data.pv_marginal_cost * results.hourly["pv"]
            + (data.energy_price + data.import_tariff)
            * results.hourly["import"]
            - (data.energy_price - data.export_tariff)
            * results.hourly["export"]
        ).sum()
    )

def calculate_quadratic_disutility(
    data: InputData,
    results: Results,
) -> float:
    """Return total daily quadratic load disutility in DKK."""

    if data.reference_load is None:
        raise ValueError("Quadratic disutility requires a reference load.")

    if data.quadratic_disutility is None:
        raise ValueError("Quadratic disutility coefficient is missing.")

    deviation = results.hourly["load"] - data.reference_load

    return float(
        (
            data.quadratic_disutility
            * deviation**2
        ).sum()
    )

def analyze_linear_sensitivity(
    data: InputData,
    results: list[tuple[float, Results]],
) -> list[dict]:
    """Calculate summary metrics for a linear-disutility sensitivity sweep."""

    metrics = []

    for cL, result in results:

        # Difference between optimized and preferred load
        deviation = result.hourly["load"] - data.reference_load

        # Total absolute deviation from reference profile
        absolute_deviation = abs(deviation).sum()

        # Linear disutility
        disutility = cL * absolute_deviation

        # Electricity procurement cost
        procurement_cost = calculate_procurement_cost(data, result)

        # Total daily consumption
        daily_load = result.hourly["load"].sum()

        metrics.append({
            "cL": cL,
            "daily_load": daily_load,
            "absolute_deviation": absolute_deviation,
            "disutility": disutility,
            "procurement_cost": procurement_cost,
        })

    return metrics

def analyze_quadratic_sensitivity(
    data: InputData,
    results: list[tuple[float, Results]],
) -> list[dict]:
    """Calculate summary metrics for a quadratic-disutility sensitivity sweep."""

    metrics = []

    for cQ, result in results:

        # Difference between optimized and preferred load
        deviation = result.hourly["load"] - data.reference_load

        # Total absolute deviation for easy interpretation
        absolute_deviation = abs(deviation).sum()

        # Sum of squared deviations
        squared_deviation = (deviation**2).sum()

        # Quadratic disutility
        disutility = cQ * squared_deviation

        # Electricity procurement cost
        procurement_cost = calculate_procurement_cost(data, result)

        # Total daily consumption
        daily_load = result.hourly["load"].sum()

        metrics.append({
            "cQ": cQ,
            "daily_load": daily_load,
            "absolute_deviation": absolute_deviation,
            "squared_deviation": squared_deviation,
            "disutility": disutility,
            "procurement_cost": procurement_cost,
        })

    return metrics

def compare_q2_q3(
    data_q2: InputData,
    result_q2: Results,
    data_q3: InputData,
    result_q3: Results,
) -> dict[str, float | int | None]:
    """Compare Q2 quadratic with Q3 minimum daily energy requirement.

    Returns the main numerical metrics used in the Question 3 base-case
    comparison in the report.
    """

    if data_q2.reference_load is None or data_q3.reference_load is None:
        raise ValueError("Q2/Q3 comparison requires reference load profiles.")

    load_q2 = float(result_q2.hourly["load"].sum())
    load_q3 = float(result_q3.hourly["load"].sum())

    procurement_q2 = calculate_procurement_cost(data_q2, result_q2)
    procurement_q3 = calculate_procurement_cost(data_q3, result_q3)

    disutility_q2 = calculate_quadratic_disutility(data_q2, result_q2)
    disutility_q3 = calculate_quadratic_disutility(data_q3, result_q3)

    above_ref_q2 = int(
        (
            result_q2.hourly["load"]
            > data_q2.reference_load + 1e-6
        ).sum()
    )

    above_ref_q3 = int(
        (
            result_q3.hourly["load"]
            > data_q3.reference_load + 1e-6
        ).sum()
    )

    return {
        "load_q2": load_q2,
        "load_q3": load_q3,
        "procurement_q2": procurement_q2,
        "procurement_q3": procurement_q3,
        "disutility_q2": disutility_q2,
        "disutility_q3": disutility_q3,
        "objective_q2": float(result_q2.objective),
        "objective_q3": float(result_q3.objective),
        "import_q2": float(result_q2.hourly["import"].sum()),
        "import_q3": float(result_q3.hourly["import"].sum()),
        "export_q2": float(result_q2.hourly["export"].sum()),
        "export_q3": float(result_q3.hourly["export"].sum()),
        "pv_q2": float(result_q2.hourly["pv"].sum()),
        "pv_q3": float(result_q3.hourly["pv"].sum()),
        "above_ref_q2": above_ref_q2,
        "above_ref_q3": above_ref_q3,
        "energy_dual_q3": result_q3.duals.get("min_daily_energy"),
    }

