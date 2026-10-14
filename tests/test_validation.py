"""Numerical regressions for ties, activation and corrupted solver outputs."""

import unittest
from copy import deepcopy
from dataclasses import replace
import numpy as np

from src.data_loader import load_question
from src.model import (
    FlexibleConsumerModel,
    QuadraticDisutilityModel,
    DailyEnergyModel,
    BatteryModel,
)
from src.analysis import (
    analytical_hourly_solution,
    validate_solution,
    analyze_q3_energy_sensitivity,
)
from src.scenarios import (
    scale_pv,
    scale_prices,
    set_load_preferences,
    set_battery_capacity,
)


class ValidationTests(unittest.TestCase):
    def test_invalid_inputs_are_rejected_before_solving(self):
        d = load_question("Q3_battery")
        for changes in [
            {"energy_price": np.full(24, np.nan)},
            {"quadratic_disutility": -1.0},
            {"battery_charging_efficiency": 0.0},
            {"battery_discharging_efficiency": 1.1},
            {"battery_initial_soc_kWh": 5.0},
            {"min_daily_energy_kWh": 145.0},
        ]:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(d, **changes)

    def test_mean_preserving_scale_rejects_clipping(self):
        d = load_question("Q3")
        self.assertAlmostEqual(
            scale_prices(d, 2, True).energy_price.mean(), d.energy_price.mean()
        )
        with self.assertRaisesRegex(ValueError, "changing the mean"):
            scale_prices(d, 3, True)

    def test_capacity_sweep_preserves_initial_fraction(self):
        d = replace(load_question("Q3_battery"), battery_initial_soc_kWh=1.0)
        self.assertEqual(set_battery_capacity(d, 8.0).battery_initial_soc_kWh, 2.0)

    def test_import_tie_accepts_an_optimal_interval(self):
        d = load_question("Q1_caseA")
        d = scale_pv(
            replace(
                d, energy_price=np.full(24, d.consumption_utility - d.import_tariff)
            ),
            0,
        )
        rows = analytical_hourly_solution(d, "utility")
        self.assertTrue(all(row["nonunique_load"] for row in rows))
        self.assertEqual(rows[0]["load_min_optimal"], d.load_min_kWh)
        self.assertEqual(rows[0]["load_max_optimal"], d.load_max_kWh)
        FlexibleConsumerModel(d).build().solve()

    def test_energy_activation_matches_analytical_threshold(self):
        d = load_question("Q3")
        # Derive the threshold independently from input data and hourly economics.
        unconstrained_energy = sum(
            row["load"] for row in analytical_hourly_solution(d, "quadratic")
        )
        step = 0.01  # kWh, larger than the feasibility tolerance
        results = []
        for energy in [unconstrained_energy - step, unconstrained_energy + step]:
            scenario = set_load_preferences(d, min_daily_energy_kWh=energy)
            results.append((energy, DailyEnergyModel(scenario).build().solve()))
        metrics = analyze_q3_energy_sensitivity(d, results)
        self.assertAlmostEqual(
            metrics[0]["daily_load"], unconstrained_energy, delta=1e-5
        )
        self.assertAlmostEqual(metrics[0]["energy_slack"], step, delta=1e-5)
        self.assertFalse(metrics[0]["energy_active"])
        self.assertFalse(metrics[0]["energy_binding"])
        self.assertAlmostEqual(metrics[0]["energy_dual"], 0.0, delta=1e-5)
        self.assertAlmostEqual(
            metrics[1]["daily_load"], unconstrained_energy + step, delta=1e-6
        )
        self.assertTrue(metrics[1]["energy_active"])
        self.assertTrue(metrics[1]["energy_binding"])
        self.assertGreater(metrics[1]["energy_dual"], 1e-6)

    def test_corrupted_balance_is_rejected(self):
        d = load_question("Q2_quadratic")
        result = deepcopy(QuadraticDisutilityModel(d).build().solve())
        result.hourly.loc[0, "import"] += 0.1
        with self.assertRaisesRegex(ValueError, "feasibility"):
            validate_solution(d, result)

    def test_corrupted_terminal_soc_is_rejected(self):
        d = load_question("Q3_battery")
        result = deepcopy(BatteryModel(d).build().solve())
        result.meta["final_battery_soc"] += 0.1
        with self.assertRaisesRegex(ValueError, "feasibility"):
            validate_solution(d, result)

    def test_corrupted_objective_is_rejected(self):
        d = load_question("Q2_quadratic")
        result = deepcopy(QuadraticDisutilityModel(d).build().solve())
        result.objective += 0.1
        with self.assertRaisesRegex(ValueError, "Objective"):
            validate_solution(d, result)


if __name__ == "__main__":
    unittest.main()
