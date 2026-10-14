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
    """Check input guards, mathematical expectations and validation failures.

    Notes
    -----
    Fixtures use the supplied input data. Deliberate result perturbations check
    that the validator detects errors, rather than only accepting valid solves.
    """

    def test_invalid_inputs_are_rejected_before_solving(self):
        """Reject invalid physical/economic inputs before constructing a solver model.

        Notes
        -----
        Replace valid inputs with non-finite values, invalid efficiencies or
        infeasible bounds; each replacement must raise ValueError.
        """
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
        """Preserve the declared mean and reject factors that would require clipping.

        Notes
        -----
        A valid spread factor keeps the original mean. A factor producing
        negative market prices must be rejected instead of silently clipped.
        """
        d = load_question("Q3")
        self.assertAlmostEqual(
            scale_prices(d, 2, True).energy_price.mean(), d.energy_price.mean()
        )
        with self.assertRaisesRegex(ValueError, "changing the mean"):
            scale_prices(d, 3, True)

    def test_capacity_sweep_preserves_initial_fraction(self):
        """Keep initial SoC divided by capacity constant when changing capacity.

        Notes
        -----
        Start with a 25 percent fill ratio; doubling capacity must double the
        initial stored energy while retaining that ratio.
        """
        d = replace(load_question("Q3_battery"), battery_initial_soc_kWh=1.0)
        self.assertEqual(set_battery_capacity(d, 8.0).battery_initial_soc_kWh, 2.0)

    def test_import_tie_accepts_an_optimal_interval(self):
        """Accept any optimal load when utility equals the effective import price.

        Notes
        -----
        Remove PV and set the import price equal to marginal utility. The
        hourly objective is then flat across the entire feasible load interval.
        """
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
        """Check activity on either side of the independently derived energy threshold.

        Notes
        -----
        Derive E0 from hourly analytical optima. A minimum below E0 must be
        slack; a minimum above E0 must be tight with a positive energy dual.
        """
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
        """Detect an import perturbation that breaks hourly power balance.

        Notes
        -----
        Add 0.1 kWh to one solved import without changing demand. The validator
        must reject the resulting physical imbalance.
        """
        d = load_question("Q2_quadratic")
        result = deepcopy(QuadraticDisutilityModel(d).build().solve())
        result.hourly.loc[0, "import"] += 0.1
        with self.assertRaisesRegex(ValueError, "feasibility"):
            validate_solution(d, result)

    def test_corrupted_terminal_soc_is_rejected(self):
        """Detect a terminal-state perturbation that breaks storage feasibility.

        Notes
        -----
        Change only the final stored energy. The validator must detect the
        broken storage dynamics or cyclic boundary condition.
        """
        d = load_question("Q3_battery")
        result = deepcopy(BatteryModel(d).build().solve())
        result.meta["final_battery_soc"] += 0.1
        with self.assertRaisesRegex(ValueError, "feasibility"):
            validate_solution(d, result)

    def test_corrupted_objective_is_rejected(self):
        """Detect a reported objective inconsistent with reconstructed costs.

        Notes
        -----
        Increase the reported objective by 0.1 DKK while retaining all physical
        decisions. Independent cost reconstruction must detect the discrepancy.
        """
        d = load_question("Q2_quadratic")
        result = deepcopy(QuadraticDisutilityModel(d).build().solve())
        result.objective += 0.1
        with self.assertRaisesRegex(ValueError, "Objective"):
            validate_solution(d, result)


if __name__ == "__main__":
    unittest.main()
