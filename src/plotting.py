"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    if "battery_soc" in hr:

        h_soc = np.arange(data.n_hours + 1)

        soc = np.append(
            hr["battery_soc"].to_numpy(),
            results.meta["final_battery_soc"],
        )

        ax.plot(
            h_soc,
            soc,
            color="C2",
            ls="-.",
            marker=".",
            lw=1.5,
            label="battery SoC [kWh]",
        )

        ax.set_xlim(-0.5, 24.5)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} (objective {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)

def plot_linear_disutility_sensitivity(
    results: list[tuple[float, Results]],
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Plot optimal daily load for different linear disutility coefficients."""

    # Extract cL values and corresponding optimal daily load
    cL_values = [cL for cL, _ in results]
    total_load = [r.hourly["load"].sum() for _, r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 4))

    # Optimal daily load from the sensitivity analysis
    ax.plot(
        cL_values,
        total_load,
        marker="o",
        label="optimal daily load",
    )

    # Total reference load
    ax.axhline(
        data.reference_load.sum(),
        ls="--",
        color="grey",
        label="reference daily load",
    )

    # PV marginal cost threshold
    ax.axvline(
        data.pv_marginal_cost,
        ls=":",
        color="grey",
        label=rf"$c^{{PV}}={data.pv_marginal_cost:.2f}$ DKK/kWh",
    )

    # Labels and title
    ax.set(
        xlabel=r"Linear disutility $c^L$ [DKK/kWh]",
        ylabel="Daily load [kWh]",
    )

    ax.legend()

    return _finish(fig, save_to)

def plot_quadratic_disutility_sensitivity(
    results,
    data,
    save_to=None,
):
    """Plot deviation from reference load for different cQ values."""

    cQ_values = [cQ for cQ, _ in results]

    deviations = [
        abs(result.hourly["load"] - data.reference_load).sum()
        for _, result in results
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(cQ_values, deviations, marker="o")

    ax.set(
        xlabel=r"Quadratic disutility $c^Q$ [DKK/kWh$^2$]",
        ylabel="Absolute deviation from reference [kWh]",
    )

    ax.set_xscale("log")

    return _finish(fig, save_to)

def plot_q3_comparison(
    result_q2: Results,
    result_q3: Results,
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Compare the reference load with Q2 quadratic and Q3 optimal load schedules."""

    h = result_q2.hourly.index.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Reference load profile
    ax.step(
        h,
        data.reference_load,
        where="mid",
        ls="--",
        label="reference load",
    )

    # Q2(c): unconstrained quadratic-disutility load
    ax.step(
        h,
        result_q2.hourly["load"],
        where="mid",
        label="Q2 quadratic",
    )

    # Q3: quadratic disutility + minimum daily energy requirement
    ax.step(
        h,
        result_q3.hourly["load"],
        where="mid",
        label="Q3 minimum energy",
    )

    ax.set(
        xlabel="hour",
        ylabel="load [kWh/h]",
        title="Q2(c) and Q3 load comparison",
    )

    ax.legend()

    return _finish(fig, save_to)


def plot_q3_battery_comparison(
    result_q3,
    result_bat,
    data,
    save_to=None,
):
    """Compare optimal load without and with battery."""

    hr_q3 = result_q3.hourly
    hr_bat = result_bat.hourly
    h = hr_q3.index.to_numpy()

    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Reference profile
    ax.step(
        h,
        data.reference_load,
        where="mid",
        color="grey",
        ls=":",
        lw=2,
        label="reference load",
    )

    # Q3 without battery
    ax.step(
        h,
        hr_q3["load"],
        where="mid",
        color="C0",
        lw=1.8,
        label="Q3 without battery",
    )

    # Q3 with battery
    ax.step(
        h,
        hr_bat["load"],
        where="mid",
        color="C2",
        lw=1.8,
        label="Q3 with battery",
    )

    ax.set(
        xlabel="hour",
        ylabel="kWh/h",
        title="Effect of battery on optimal load schedule",
    )

    ax.legend(
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )

    return _finish(fig, save_to)
def plot_q3_energy_sensitivity(
    metrics: list[dict],
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Plot objective and dual value for different minimum daily energy requirements."""

    Emin_values = [m["E_min"] for m in metrics]
    objectives = [m["objective"] for m in metrics]
    energy_duals = [m["energy_dual"] for m in metrics]

    binding_Emin = next(
        (
            m["E_min"]
            for m in metrics
            if m["energy_dual"] is not None
            and m["energy_dual"] > 1e-4
        ),
        None,
    )

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(
        Emin_values,
        objectives,
        marker="o",
        label="optimal objective",
    )

    ax1.set(
        xlabel=r"Minimum daily energy $E^{min}$ [kWh]",
        ylabel="Objective [DKK]",
        title="Q3 sensitivity to minimum daily energy",
    )

    ax2 = ax1.twinx()

    ax2.plot(
        Emin_values,
        energy_duals,
        marker="s",
        ls="--",
        color="orange",
        label=r"dual $\mu$",
    )

    ax2.set_ylabel(
        r"Dual value $\mu$ [DKK/kWh]"
    )
    ax2.set_ylim(0, 3.6)

    ax2.axhline(
        0,
        ls=":",
        lw=1,
    )

    reference_energy = data.reference_load.sum()

    ax1.axvline(
        reference_energy,
        ls="--",
        color="grey",
        label=f"reference load = {reference_energy:.2f} kWh",
    )

    if binding_Emin is not None:
        ax1.axvline(
            binding_Emin,
            ls=":",
            color="grey",
            label=f"binding point ≈ {binding_Emin:.2f} kWh",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
    )

    return _finish(fig, save_to)

def plot_q3_energy_supply_sensitivity(
    metrics: list[dict],
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Plot PV use and grid import for different minimum daily energy requirements."""

    Emin_values = [m["E_min"] for m in metrics]
    pv_use = [m["pv"] for m in metrics]
    grid_import = [m["import"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        Emin_values,
        pv_use,
        marker="o",
        label="PV used",
    )

    ax.plot(
        Emin_values,
        grid_import,
        marker="s",
        ls="--",
        label="grid import",
    )

    ax.axhline(
        data.pv_available.sum(),
        ls=":",
        label=f"available PV = {data.pv_available.sum():.2f} kWh",
    )

    binding_Emin = 22.77
    ax.axvline(
        binding_Emin,
        ls=":",
        color="grey",
        label=f"unconstrained optimum = {binding_Emin:.2f} kWh",
    )

    ax.axvline(
        data.reference_load.sum(),
        ls="--",
        color="grey",
        label=f"reference load = {data.reference_load.sum():.2f} kWh",
    )

    ax.set(
        xlabel=r"Minimum daily energy $E^{min}$ [kWh]",
        ylabel="Daily energy [kWh]",
        title="Q3 energy supply sensitivity",
    )

    ax.legend()

    return _finish(fig, save_to)

def plot_q3_cq_sensitivity(
    metrics: list[dict],
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Plot deviation and procurement cost for different quadratic disutility coefficients."""

    cQ_values = [m["cQ"] for m in metrics]
    deviations = [m["absolute_deviation"] for m in metrics]
    procurement_costs = [m["procurement_cost"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(
        cQ_values,
        deviations,
        marker="o",
        label="absolute deviation",
    )

    ax1.set(
        xlabel=r"Quadratic disutility $c^Q$ [DKK/kWh$^2$]",
        ylabel="Absolute deviation [kWh]",
        title=r"Q3 sensitivity to quadratic disutility $c^Q$",
    )

    ax1.set_xscale("log")

    ax2 = ax1.twinx()

    ax2.plot(
        cQ_values,
        procurement_costs,
        marker="s",
        ls="--",
        label="procurement cost",
    )

    ax2.set_ylabel(
        "Procurement cost [DKK]"
    )

    ax1.axvline(
        1.0,
        ls=":",
        color="grey",
        label=r"base case $c^Q=1$",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="best",
    )

    return _finish(fig, save_to)
