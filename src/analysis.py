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

import numpy as np

from .data_loader import InputData
from .model import Results

def count_deviation_bound_hours(
    data: InputData, results: Results, tolerance: float = 1e-6,
) -> dict[str, int]:
    """Count deviations limited by physical load bounds, using absolute kWh tolerance.

    There is no additional hard deviation limit in the assignment data. The
    signed deviation is bounded by load_min-reference and load_max-reference.
    Count active endpoints once per hour, including zero-reference hours, and
    separately report hours actually deviating and curtailed at the lower bound.
    This does not count the absolute-value epigraph inequalities.
    """
    if data.reference_load is None:
        raise ValueError("Deviation-bound counts require a reference profile.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive.")
    load = results.hourly["load"].to_numpy()
    lower = np.isclose(load, data.load_min_kWh, atol=tolerance, rtol=0)
    upper = np.isclose(load, data.load_max_kWh, atol=tolerance, rtol=0)
    at_bound = lower | upper
    deviating = np.abs(load - data.reference_load) > tolerance
    return {
        "deviation_bound_hours": int(at_bound.sum()),
        "lower_bound_hours": int(lower.sum()),
        "upper_bound_hours": int(upper.sum()),
        "deviating_bound_hours": int((at_bound & deviating).sum()),
        "curtailed_bound_hours": int((lower & (load < data.reference_load - tolerance)).sum()),
    }

def calculate_q1_components(data: InputData, results: Results) -> dict[str, float]:
    """Return Q1 procurement, gross utility and surplus in DKK/day.

    The solver minimizes procurement minus utility; surplus has the opposite
    sign. Reconstructing the objective catches cost/sign inconsistencies.
    """
    if data.consumption_utility is None:
        raise ValueError("Q1 components require a consumption utility.")
    procurement = calculate_procurement_cost(data, results)
    utility = float(data.consumption_utility * results.hourly["load"].sum())
    if abs(procurement - utility - results.objective) > 1e-5 + 1e-6 * abs(results.objective):
        raise ValueError("Q1 objective does not match procurement minus utility.")
    return {"procurement_cost": procurement, "utility": utility,
            "surplus": utility - procurement, "objective": float(results.objective)}

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
            **count_deviation_bound_hours(data, result),
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
            **count_deviation_bound_hours(data, result),
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


def compare_q3_battery(data_q3, result_q3, data_bat, result_bat):
    """Compare Q3 without and with battery storage."""

    load_q3 = result_q3.hourly["load"]
    load_bat = result_bat.hourly["load"]
    reference = result_bat.hourly["reference_load"]

    charge = result_bat.hourly["battery_charge"]
    discharge = result_bat.hourly["battery_discharge"]

    # Deviation from reference
    abs_dev_q3 = (load_q3 - reference).abs().sum()
    abs_dev_bat = (load_bat - reference).abs().sum()

    sq_dev_q3 = ((load_q3 - reference) ** 2).sum()
    sq_dev_bat = ((load_bat - reference) ** 2).sum()

    # Battery value = reduction in objective
    battery_value = result_q3.objective - result_bat.objective

    # Check simultaneous charging/discharging
    tol = 1e-6
    simultaneous = (
        (charge > tol) &
        (discharge > tol)
    )

    return {
        "objective_q3": result_q3.objective,
        "objective_battery": result_bat.objective,
        "battery_value": battery_value,

        "load_q3": load_q3.sum(),
        "load_battery": load_bat.sum(),

        "abs_deviation_q3": abs_dev_q3,
        "abs_deviation_battery": abs_dev_bat,

        "sq_deviation_q3": sq_dev_q3,
        "sq_deviation_battery": sq_dev_bat,

        "import_q3": result_q3.hourly["import"].sum(),
        "import_battery": result_bat.hourly["import"].sum(),

        "export_q3": result_q3.hourly["export"].sum(),
        "export_battery": result_bat.hourly["export"].sum(),

        "battery_charge": charge.sum(),
        "battery_discharge": discharge.sum(),

        "simultaneous_hours": int(simultaneous.sum()),

        "energy_dual_q3": result_q3.duals.get("min_daily_energy"),
        "energy_dual_battery": result_bat.duals.get("min_daily_energy"),
    }
def analyze_q3_energy_sensitivity(
    data: InputData,
    results: list[tuple[float, Results]],
) -> list[dict]:
    """Calculate summary metrics for the Q3 minimum-energy sensitivity."""

    metrics = []

    for Emin, result in results:

        deviation = result.hourly["load"] - data.reference_load

        daily_load = float(result.hourly["load"].sum())

        absolute_deviation = float(abs(deviation).sum())

        squared_deviation = float((deviation**2).sum())

        procurement_cost = calculate_procurement_cost(
            data,
            result,
        )

        disutility = (
            data.quadratic_disutility
            * squared_deviation
        )

        import_total = float(result.hourly["import"].sum())
        export_total = float(result.hourly["export"].sum())
        pv_total = float(result.hourly["pv"].sum())

        above_reference_hours = int(
            (
                result.hourly["load"]
                > data.reference_load + 1e-6
            ).sum()
        )

        energy_dual = result.duals.get(
            "min_daily_energy"
        )

        metrics.append({
            "E_min": Emin,
            "daily_load": daily_load,
            "objective": float(result.objective),
            "procurement_cost": procurement_cost,
            "disutility": disutility,
            "absolute_deviation": absolute_deviation,
            "import": import_total,
            "export": export_total,
            "pv": pv_total,
            "above_reference_hours": above_reference_hours,
            "energy_dual": energy_dual,
            "energy_slack": daily_load - Emin,
            "energy_active": bool(abs(daily_load - Emin) <= 1e-6),
            "energy_binding": bool(abs(daily_load - Emin) <= 1e-6
                                   and energy_dual is not None and energy_dual > 1e-6),
        })

    return metrics

def analyze_q3_cq_sensitivity(
    data: InputData,
    results: list[tuple[float, Results]],
) -> list[dict]:
    """Calculate summary metrics for the Q3 quadratic-disutility sensitivity."""

    metrics = []

    for cQ, result in results:

        deviation = result.hourly["load"] - data.reference_load

        absolute_deviation = float(abs(deviation).sum())
        squared_deviation = float((deviation**2).sum())

        procurement_cost = calculate_procurement_cost(
            data,
            result,
        )

        disutility = cQ * squared_deviation

        daily_load = float(result.hourly["load"].sum())
        import_total = float(result.hourly["import"].sum())
        export_total = float(result.hourly["export"].sum())
        pv_total = float(result.hourly["pv"].sum())

        energy_dual = result.duals.get(
            "min_daily_energy"
        )

        metrics.append({
            "cQ": cQ,
            "daily_load": daily_load,
            "objective": float(result.objective),
            "procurement_cost": procurement_cost,
            "disutility": disutility,
            "absolute_deviation": absolute_deviation,
            "squared_deviation": squared_deviation,
            "import": import_total,
            "export": export_total,
            "pv": pv_total,
            "energy_dual": energy_dual,
        })

    return metrics

def calculate_temporal_disutility(
    data: InputData,
    results: Results,
    c0: float,
    cT: float,
) -> float:
    """Return temporal disutility in DKK, with coefficients in DKK/kWh²."""

    if data.reference_load is None:
        raise ValueError("Temporal disutility requires a reference load.")
    if c0 <= 0 or cT < 0:
        raise ValueError("Require c0 > 0 and cT >= 0.")

    deviation = (
        results.hourly["load"] - data.reference_load
    ).abs().to_numpy()

    # Zero padding gives both boundary hours two individual-square terms.
    pairs = deviation[0] ** 2 + deviation[-1] ** 2
    pairs += ((deviation[:-1] + deviation[1:]) ** 2).sum()

    return float(c0 * (deviation ** 2).sum() + cT * pairs)


def calculate_cost_components(data: InputData, results: Results) -> dict[str, float]:
    """Reconstruct model economics from physical flows and actual deviations."""
    preference = results.meta["preference"]
    if preference == "utility":
        return calculate_q1_components(data, results)
    procurement = calculate_procurement_cost(data, results)
    if preference == "linear":
        discomfort = float(data.linear_disutility * abs(results.hourly["load"] - data.reference_load).sum())
    elif preference == "quadratic":
        discomfort = calculate_quadratic_disutility(data, results)
    else:
        discomfort = calculate_temporal_disutility(data, results, results.meta["c0"], results.meta["cT"])
    return {"procurement_cost": procurement, "disutility": discomfort,
            "surplus": -procurement - discomfort, "objective": float(results.objective)}


def analytical_hourly_solution(data: InputData, preference: str, mu: float = 0.0) -> list[dict]:
    """Solve separable hours by economic breakpoints, independently of Gurobi.

    At fixed load, PV candidates are zero, available PV and load clipped to PV
    availability. Linear objectives attain optima at load bounds/reference/PV
    breakpoints; quadratic objectives also have stationary points obtained from
    each possible procurement marginal cost. Return optimal load intervals for
    linear ties. mu is the Q3 daily-energy multiplier in DKK/kWh.
    """
    if preference not in {"utility", "linear", "quadratic"}:
        raise ValueError("Analytical hourly check requires separable preferences.")
    rows = []
    for t in range(data.n_hours):
        lower, upper = data.load_min_kWh, data.load_max_kWh
        pv = data.pv_available[t]
        buy = data.energy_price[t] + data.import_tariff
        sell = data.energy_price[t] - data.export_tariff
        ref = 0.0 if data.reference_load is None else data.reference_load[t]
        candidates = [lower, upper, float(np.clip(pv, lower, upper))]
        if preference == "linear":
            candidates.append(float(np.clip(ref, lower, upper)))
        if preference == "quadratic":
            candidates.extend(float(np.clip(ref + (mu - slope) / (2 * data.quadratic_disutility), lower, upper))
                              for slope in [buy, sell, data.pv_marginal_cost])

        def cost(load):
            procurement = min(data.pv_marginal_cost * q + buy * max(load - q, 0)
                              - sell * max(q - load, 0)
                              for q in [0.0, pv, float(np.clip(load, 0, pv))])
            if preference == "utility":
                discomfort = -data.consumption_utility * load
            elif preference == "linear":
                discomfort = data.linear_disutility * abs(load - ref)
            else:
                discomfort = data.quadratic_disutility * (load - ref) ** 2
            return float(procurement + discomfort - mu * load)

        scores = np.asarray([cost(load) for load in candidates])
        minimum = float(scores.min())
        optima = np.asarray(candidates)[np.isclose(scores, minimum, atol=1e-9, rtol=0)]
        rows.append({"hour": t, "load": float(candidates[int(scores.argmin())]),
                     "load_min_optimal": float(optima.min()), "load_max_optimal": float(optima.max()),
                     "objective": minimum, "nonunique_load": bool(optima.max() - optima.min() > 1e-6)})
    return rows


def validate_solution(data: InputData, results: Results, tolerance: float = 1e-6) -> dict:
    """Check physics, economic reconstruction, duals and analytical optima.

    Uses absolute kWh feasibility tolerance, objective tolerance 1e-5 DKK plus
    1e-6 relative, and dual/complementarity tolerance 1e-5. Optimal linear ties
    are accepted without enforcing one solver-selected schedule. Battery overlap
    is reported and rejected for these assignment experiments, without adding
    integer variables or changing the convex model.
    """
    h = results.hourly
    flows = h[["import", "export", "pv", "load"]].to_numpy()
    if not np.isfinite(flows).all():
        raise ValueError("Solution contains non-finite physical flows.")
    errors = [max(0.0, float(-h[["import", "export", "pv"]].min().min())),
              max(0.0, float(data.load_min_kWh - h.load.min())),
              max(0.0, float(h.load.max() - data.load_max_kWh)),
              max(0.0, float((h.pv - data.pv_available).max()))]
    balance = h.pv + h["import"] - h.load - h["export"]
    overlap = 0
    if results.meta["battery_enabled"]:
        soc = np.r_[h.battery_soc.to_numpy(), results.meta["final_battery_soc"]]
        charge, discharge = h.battery_charge, h.battery_discharge
        if not np.isfinite(np.r_[soc, charge, discharge]).all():
            raise ValueError("Battery solution contains non-finite values.")
        balance += discharge - charge
        errors.extend([float(np.max(np.abs(np.diff(soc) - data.battery_charging_efficiency * charge
                                           + discharge / data.battery_discharging_efficiency))),
                       abs(float(soc[0] - data.battery_initial_soc_kWh)), abs(float(soc[-1] - soc[0])),
                       max(0.0, float(-soc.min())), max(0.0, float(soc.max() - data.battery_capacity_kWh)),
                       max(0.0, float(-charge.min())), max(0.0, float(-discharge.min())),
                       max(0.0, float(charge.max() - data.battery_max_charge_kW)),
                       max(0.0, float(discharge.max() - data.battery_max_discharge_kW))])
        overlap = int(((charge > tolerance) & (discharge > tolerance)).sum())
        if overlap:
            raise ValueError("Battery charge/discharge overlap: investigate prices or degeneracy.")
    errors.append(float(np.abs(balance).max()))
    if results.meta["daily_energy_enabled"]:
        errors.append(max(0.0, float(data.min_daily_energy_kWh - h.load.sum())))
    feasibility = max(errors)
    if feasibility > tolerance:
        raise ValueError(f"Physical feasibility error: {feasibility:g} kWh")
    components = calculate_cost_components(data, results)
    preference_cost = -components["utility"] if "utility" in components else components["disutility"]
    objective_error = abs(results.objective - components["procurement_cost"] - preference_cost)
    objective_tolerance = 1e-5 + 1e-6 * abs(results.objective)
    if objective_error > objective_tolerance:
        raise ValueError("Objective does not match the reconstructed economic components.")
    if "deviation" in h and np.max(np.abs(h.deviation - abs(h.load - data.reference_load))) > tolerance:
        raise ValueError("Absolute-deviation auxiliary does not equal the actual deviation.")
    if data.import_tariff + data.export_tariff > 0 and np.minimum(h["import"], h["export"]).max() > tolerance:
        raise ValueError("Simultaneous import/export with a strictly positive tariff spread.")

    dual_errors, complementarity = [], []
    for name, slack, sign in [
        ("load_min", h.load - data.load_min_kWh, 1),
        ("load_max", data.load_max_kWh - h.load, -1),
        ("pv_max", data.pv_available - h.pv, -1),
    ]:
        dual = h[f"dual_{name}"]
        dual_errors.append(max(0.0, float((-sign * dual).max())))
        complementarity.append(float(abs(slack * dual).max()))
    for t in range(data.n_hours):
        price = h.dual_balance.iloc[t]
        if h["import"].iloc[t] > 1e-5:
            dual_errors.append(abs(float(price - data.energy_price[t] - data.import_tariff)))
        if h["export"].iloc[t] > 1e-5:
            dual_errors.append(abs(float(price - data.energy_price[t] + data.export_tariff)))
        if results.meta["preference"] == "utility" and h["import"].iloc[t] <= tolerance and h["export"].iloc[t] <= tolerance and data.load_min_kWh + tolerance < h.load.iloc[t] < data.load_max_kWh - tolerance:
            dual_errors.append(abs(float(price - data.consumption_utility)))
    if results.meta["daily_energy_enabled"]:
        mu = results.duals["min_daily_energy"]
        dual_errors.append(max(0.0, -mu))
        complementarity.append(abs(float(mu * (h.load.sum() - data.min_daily_energy_kWh))))
    if max(dual_errors, default=0.0) > 1e-5 or max(complementarity, default=0.0) > 1e-5:
        raise ValueError("Dual sign, marginal-price or complementary-slackness check failed.")

    oracle_error = None
    if results.meta["preference"] != "temporal" and not results.meta["battery_enabled"]:
        preference = results.meta["preference"]
        oracle = analytical_hourly_solution(data, preference)
        mu = 0.0
        if results.meta["daily_energy_enabled"] and sum(row["load"] for row in oracle) < data.min_daily_energy_kWh:
            low, high = 0.0, 1.0
            while sum(row["load"] for row in analytical_hourly_solution(data, preference, high)) < data.min_daily_energy_kWh:
                high *= 2
            for _ in range(60):
                mu = (low + high) / 2
                oracle = analytical_hourly_solution(data, preference, mu)
                if sum(row["load"] for row in oracle) < data.min_daily_energy_kWh:
                    low = mu
                else:
                    high = mu
        oracle_objective = sum(row["objective"] + mu * row["load"] for row in oracle)
        oracle_error = abs(results.objective - oracle_objective)
        if oracle_error > objective_tolerance:
            raise ValueError("Solver objective differs from the independent analytical optimum.")
        for row in oracle:
            t = row["hour"]
            load, pv = float(h.load.iloc[t]), float(h.pv.iloc[t])
            buy, sell = data.energy_price[t] + data.import_tariff, data.energy_price[t] - data.export_tariff
            candidates = np.asarray([0.0, data.pv_available[t], np.clip(load, 0, data.pv_available[t])])
            costs = data.pv_marginal_cost * candidates + buy * np.maximum(load - candidates, 0) - sell * np.maximum(candidates - load, 0)
            optimal_pv = candidates[np.isclose(costs, costs.min(), atol=1e-9, rtol=0)]
            row.update({"solver_load": load, "solver_pv": pv,
                        "solver_import": float(h["import"].iloc[t]), "solver_export": float(h["export"].iloc[t]),
                        "solver_balance_dual": float(h.dual_balance.iloc[t]),
                        "pv_min_optimal_at_solver_load": float(optimal_pv.min()),
                        "pv_max_optimal_at_solver_load": float(optimal_pv.max())})
            if pv < optimal_pv.min() - tolerance or pv > optimal_pv.max() + tolerance:
                raise ValueError("Hourly PV production differs from the analytical procurement optimum.")
            if preference == "utility":
                dual_min, dual_max = float(sell), float(buy)
                if h["import"].iloc[t] > tolerance:
                    dual_min = max(dual_min, float(buy))
                if h["export"].iloc[t] > tolerance:
                    dual_max = min(dual_max, float(sell))
                if load > data.load_min_kWh + tolerance:
                    dual_max = min(dual_max, data.consumption_utility)
                if load < data.load_max_kWh - tolerance:
                    dual_min = max(dual_min, data.consumption_utility)
                if pv > tolerance:
                    dual_min = max(dual_min, data.pv_marginal_cost)
                if pv < data.pv_available[t] - tolerance:
                    dual_max = min(dual_max, data.pv_marginal_cost)
                row.update({"balance_dual_min": dual_min, "balance_dual_max": dual_max,
                            "nonunique_balance_dual": bool(dual_max - dual_min > tolerance)})
                if h.dual_balance.iloc[t] < dual_min - 1e-5 or h.dual_balance.iloc[t] > dual_max + 1e-5:
                    raise ValueError("Q1 balance dual lies outside its KKT price-ladder interval.")
        results.meta["analytical_hourly"] = oracle
    return {"max_feasibility_error_kWh": feasibility, "objective_error_DKK": objective_error,
            "max_dual_error_DKK_per_kWh": max(dual_errors, default=0.0),
            "max_complementarity_error": max(complementarity, default=0.0),
            "analytical_objective_error_DKK": oracle_error,
            "simultaneous_battery_hours": overlap}


def compare_q2_disutility_models(
    runs: dict[str, tuple[InputData, Results]],
    c0: float,
    cT: float,
) -> list[dict]:
    """Compare solved Q2 models using the existing sensitivity metrics.

    Each run maps a model label to its input data and Results. Costs and
    surplus are in DKK/day; load and deviations are in kWh/day, except
    maximum_deviation, which is the largest one-hour deviation in kWh.
    """

    metrics = []

    for label, (data, result) in runs.items():
        if data.reference_load is None:
            raise ValueError("Q2 comparison requires a reference load.")

        if label == "Linear":
            values = analyze_linear_sensitivity(
                data, [(data.linear_disutility, result)]
            )[0]

        elif label == "Quadratic":
            values = analyze_quadratic_sensitivity(
                data, [(data.quadratic_disutility, result)]
            )[0]

        elif label == "Temporal":
            deviation = result.hourly["load"] - data.reference_load
            values = {
                "daily_load": result.hourly["load"].sum(),
                "absolute_deviation": deviation.abs().sum(),
                "procurement_cost": calculate_procurement_cost(data, result),
                "disutility": calculate_temporal_disutility(data, result, c0, cT),
            }

        else:
            raise ValueError(f"Unknown Q2 model label: {label}")

        # Verify the objective using physical procurement and actual deviations.
        reconstructed = values["procurement_cost"] + values["disutility"]
        tolerance = 1e-5 + 1e-6 * abs(result.objective)
        if abs(reconstructed - result.objective) > tolerance:
            raise ValueError(f"{label} objective does not match its cost components.")

        deviation = result.hourly["load"] - data.reference_load

        metrics.append({
            "model": label,
            "daily_load": float(values["daily_load"]),
            "absolute_deviation": float(values["absolute_deviation"]),
            "procurement_cost": float(values["procurement_cost"]),
            "disutility": float(values["disutility"]),
            "objective": float(result.objective),
            "surplus": float(-result.objective),
            "maximum_deviation": float(deviation.abs().max()),
            "import": float(result.hourly["import"].sum()),
            "export": float(result.hourly["export"].sum()),
            "pv": float(result.hourly["pv"].sum()),
            **count_deviation_bound_hours(data, result),
        })

    return metrics
