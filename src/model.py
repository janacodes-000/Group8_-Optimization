"""Shared parameterized LP/QP and compatible question-specific configurations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)
    metrics: dict[str, float] = field(default_factory=dict)  # daily economic components in DKK

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")
            for k, v in self.metrics.items():
                f.write(f"{k} : {v:.6f} DKK\n")
        (folder / f"{stem}_summary.json").write_text(
            json.dumps({"status": self.status, "objective": self.objective,
                        "metrics": self.metrics, "duals": self.duals,
                        "meta": self.meta}, indent=2, default=_json_value) + "\n",
            encoding="utf-8",
        )
        if "battery_soc" in self.hourly:
            states = np.r_[self.hourly.battery_soc.to_numpy(), self.meta["final_battery_soc"]]
            pd.DataFrame({"boundary_hour": np.arange(len(states)), "soc_kWh": states}).to_csv(
                folder / f"{stem}_battery_states.csv", index=False,
            )
        if "analytical_hourly" in self.meta:
            pd.DataFrame(self.meta["analytical_hourly"]).to_csv(folder / f"{stem}_analytical.csv", index=False)

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


def _json_value(value):
    """Serialize NumPy values and paths without rounding report parameters."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


class FlexibleConsumerModel:
    """One parameterized continuous LP/QP for every assignment question.

    Preference selects utility, linear, quadratic or temporal discomfort.
    Daily-energy and cyclic-battery constraints are independent options. Public
    question classes below only configure this builder; all algebra is shared.
    """

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False,
                 *, preference: str = "utility", daily_energy: bool = False,
                 battery: bool = False, c0: float = 0.5, cT: float = 0.25):
        self.data = data
        self.preference = preference
        self.daily_energy = daily_energy
        self.battery = battery
        self.c0, self.cT = c0, cT
        self.T = range(data.n_hours)
        if preference not in {"utility", "linear", "quadratic", "temporal"}:
            raise ValueError(f"Unknown preference: {preference}")
        if preference == "temporal" and (not np.isfinite([c0, cT]).all() or c0 <= 0 or cT < 0):
            raise ValueError("Require finite c0 > 0 and cT >= 0.")
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = int(verbose)
        self.m.Params.BarConvTol = 1e-12
        self.var: dict[str, gp.tupledict | gp.Var] = {}
        self.con: dict[str, gp.tupledict | gp.Constr] = {}

    def build(self) -> "FlexibleConsumerModel":
        """Build shared physics, configured preferences and optional coupling."""
        if self.var:
            raise RuntimeError("This model is already built; create a new instance for another run.")
        self._build_physical_model()
        if self.preference in {"linear", "temporal"}:
            self._build_absolute_deviations()
        if self.daily_energy:
            if self.data.min_daily_energy_kWh is None:
                raise ValueError("Daily-energy model requires a minimum daily energy.")
            self.con["min_daily_energy"] = self.m.addConstr(
                gp.quicksum(self.var["load"][t] for t in self.T) >= self.data.min_daily_energy_kWh,
                name="min_daily_energy",
            )
        self.m.setObjective(self._procurement_expression() + self._preference_expression(), GRB.MINIMIZE)
        self.m.update()
        return self

    def _build_physical_model(self):
        """Create one shared balance and explicit load/PV limits per hour."""
        d, m, T = self.data, self.m, self.T
        for name in ["import", "export", "pv"]:
            self.var[name] = m.addVars(T, lb=0, name=name)
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, name="load")
        if self.battery:
            self._build_battery()
        charge, discharge = self.var.get("battery_charge", {}), self.var.get("battery_discharge", {})
        self.con["balance"] = m.addConstrs(
            (self.var["pv"][t] + self.var["import"][t] + discharge.get(t, 0)
             == self.var["load"][t] + self.var["export"][t] + charge.get(t, 0) for t in T), name="balance")
        self.con["load_min"] = m.addConstrs((self.var["load"][t] >= d.load_min_kWh for t in T), name="load_min")
        self.con["load_max"] = m.addConstrs((self.var["load"][t] <= d.load_max_kWh for t in T), name="load_max")
        self.con["pv_max"] = m.addConstrs((self.var["pv"][t] <= d.pv_available[t] for t in T), name="pv_max")

    def _build_absolute_deviations(self):
        """Shared epigraph of absolute load-reference deviation."""
        d, m, T = self.data, self.m, self.T
        if d.reference_load is None:
            raise ValueError("Disutility requires a reference profile.")
        a = self.var["deviation"] = m.addVars(T, lb=0, name="deviation")
        load = self.var["load"]
        self.con["deviation_pos"] = m.addConstrs((a[t] >= load[t] - d.reference_load[t] for t in T), name="deviation_pos")
        self.con["deviation_neg"] = m.addConstrs((a[t] >= d.reference_load[t] - load[t] for t in T), name="deviation_neg")

    def _procurement_expression(self):
        """PV cost plus import cost minus export revenue, in DKK/day."""
        d = self.data
        return gp.quicksum(d.pv_marginal_cost * self.var["pv"][t]
                           + (d.energy_price[t] + d.import_tariff) * self.var["import"][t]
                           - (d.energy_price[t] - d.export_tariff) * self.var["export"][t] for t in self.T)

    def _preference_expression(self):
        """Configured utility/disutility; temporal boundaries are zero-padded."""
        d, T = self.data, self.T
        load = self.var["load"]
        if self.preference == "utility":
            if d.consumption_utility is None:
                raise ValueError("Utility model requires a consumption utility.")
            return -d.consumption_utility * gp.quicksum(load[t] for t in T)
        if d.reference_load is None:
            raise ValueError("Disutility requires a reference profile.")
        if self.preference == "quadratic":
            if d.quadratic_disutility is None or d.quadratic_disutility <= 0:
                raise ValueError("Quadratic disutility must be positive.")
            return d.quadratic_disutility * gp.quicksum((load[t] - d.reference_load[t]) ** 2 for t in T)
        a = self.var["deviation"]
        if self.preference == "linear":
            if d.linear_disutility is None or d.linear_disutility <= 0:
                raise ValueError("Linear disutility must be positive.")
            return d.linear_disutility * gp.quicksum(a[t] for t in T)
        return self.c0 * gp.quicksum(a[t] ** 2 for t in T) + self.cT * (
            a[0] ** 2 + gp.quicksum((a[t - 1] + a[t]) ** 2 for t in range(1, d.n_hours)) + a[d.n_hours - 1] ** 2)

    def _build_battery(self):
        """Continuous battery with 25 states, losses, power limits and cyclic closure."""
        d, m, T = self.data, self.m, self.T
        if d.battery_capacity_kWh is None:
            raise ValueError("Battery model requires battery parameters.")
        charge = self.var["battery_charge"] = m.addVars(T, lb=0, ub=d.battery_max_charge_kW, name="battery_charge")
        discharge = self.var["battery_discharge"] = m.addVars(T, lb=0, ub=d.battery_max_discharge_kW, name="battery_discharge")
        soc = self.var["battery_soc"] = m.addVars(range(d.n_hours + 1), lb=0, ub=d.battery_capacity_kWh, name="battery_soc")
        self.con["battery_initial_soc"] = m.addConstr(soc[0] == d.battery_initial_soc_kWh, name="battery_initial_soc")
        self.con["battery_soc_balance"] = m.addConstrs(
            (soc[t + 1] == soc[t] + d.battery_charging_efficiency * charge[t]
             - discharge[t] / d.battery_discharging_efficiency for t in T), name="battery_soc_balance")
        self.con["battery_final_soc"] = m.addConstr(soc[d.n_hours] == soc[0], name="battery_final_soc")

    def solve(self) -> Results:
        """Optimize and return validated primal, dual and economic results."""
        if not self.var:
            raise RuntimeError("Build the model before solving.")
        self.m.optimize()
        status = _status_name(self.m.Status)
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimization ended with {status}; check feasibility (computeIIS can help).")
        result = self._extract_results(status)
        from .analysis import calculate_cost_components, validate_solution
        result.metrics.update(calculate_cost_components(self.data, result))
        result.meta["validation"] = validate_solution(self.data, result)
        return result

    def _extract_results(self, status: str) -> Results:
        """Collect hourly variables/duals and full scenario metadata without rounding."""
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"], hourly["pv_available"] = d.energy_price, d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load
        for name, values in self.var.items():
            hourly[name] = [values[t].X for t in T]
        duals = {}
        for name, constraints in self.con.items():
            if isinstance(constraints, gp.tupledict):
                hourly[f"dual_{name}"] = [_dual(constraints[t]) for t in T]
            else:
                duals[name] = _dual(constraints)
        return Results(question=d.question, status=status, objective=float(self.m.ObjVal), hourly=hourly, duals=duals,
                       meta={"scalar_variables": {}, "final_battery_soc": self.var["battery_soc"][d.n_hours].X if self.battery else None,
                             "preference": self.preference, "daily_energy_enabled": self.daily_energy, "battery_enabled": self.battery,
                             "c0": self.c0 if self.preference == "temporal" else None,
                             "cT": self.cT if self.preference == "temporal" else None,
                             "parameters": {key: value for key, value in vars(d).items() if key != "raw"}})


class LinearDisutilityModel(FlexibleConsumerModel):
    """Configure the shared builder for Q2(b)."""

    def __init__(self, data, name="linear_disutility", verbose=False):
        super().__init__(data, name=name, verbose=verbose, preference="linear")


class QuadraticDisutilityModel(FlexibleConsumerModel):
    """Configure the shared builder for Q2(c)."""

    def __init__(self, data, name="quadratic_disutility", verbose=False):
        super().__init__(data, name=name, verbose=verbose, preference="quadratic")


class TemporalDisutilityModel(FlexibleConsumerModel):
    """Configure Q2(e), with c0/cT in DKK/kWh² and zero-padded boundaries."""

    def __init__(self, data, c0=0.5, cT=0.25, verbose=False):
        super().__init__(data, name="temporal_disutility", verbose=verbose, preference="temporal", c0=c0, cT=cT)


class DailyEnergyModel(FlexibleConsumerModel):
    """Configure Q3: quadratic preference and minimum daily consumption."""

    def __init__(self, data, name="daily_energy", verbose=False):
        super().__init__(data, name=name, verbose=verbose, preference="quadratic", daily_energy=True)


class BatteryModel(FlexibleConsumerModel):
    """Configure Q3(g): quadratic preference, daily minimum and cyclic storage."""

    def __init__(self, data, name="battery", verbose=False):
        super().__init__(data, name=name, verbose=verbose, preference="quadratic", daily_energy=True, battery=True)


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    """Readable solver termination status."""
    return _STATUS.get(code, f"STATUS_{code}")
