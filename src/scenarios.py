"""Derive sensitivity scenarios from a base ``InputData`` without touching the JSON files.

Each helper returns a *new* ``InputData`` (the base data is never modified), so scenarios can
be chained::

    from src.data_loader import load_question
    from src.scenarios import scale_prices, set_tariffs, scale_pv

    base = load_question("Q1_caseA")
    high_spread = scale_prices(base, factor=2.0, keep_mean=True)      # same mean, doubled spread
    no_export_tariff = set_tariffs(high_spread, export_tariff=0.0)

Add your own helpers here (e.g. shifting the price peak towards the PV peak, changing the
minimum daily energy or the reference profile) and document them in the README.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .data_loader import InputData


def scale_prices(data: InputData, factor: float, keep_mean: bool = False) -> InputData:
    """Return a scenario with scaled price levels or mean-preserving spread.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    factor : float
        Non-negative scaling factor.
    keep_mean : bool, optional
        Whether to scale only deviations from the original mean.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.

    Raises
    ------
    ValueError
        If scaling is invalid or would produce negative market prices.
    """
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("Price scale must be finite and non-negative.")
    p = data.energy_price
    # Scale around the mean so only variability changes in spread experiments.
    new_p = p.mean() + factor * (p - p.mean()) if keep_mean else factor * p
    if (new_p < 0).any():
        raise ValueError(
            "Price scale produces negative prices; reduce the factor rather than clipping and changing the mean."
        )
    return replace(data, energy_price=new_p)


def shift_prices(data: InputData, delta: float) -> InputData:
    """Return a scenario with delta DKK/kWh added to every hourly price.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    delta : float
        Price-level shift, in DKK/kWh.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    if not np.isfinite(delta) or (data.energy_price + delta < 0).any():
        raise ValueError("Price shift must be finite and keep prices non-negative.")
    return replace(data, energy_price=data.energy_price + delta)


def roll_prices(data: InputData, hours: int) -> InputData:
    """Return a scenario with prices shifted by ``hours`` one-hour periods.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    hours : int
        Circular time shift; positive values move prices later.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    return replace(data, energy_price=np.roll(data.energy_price, hours))


def set_tariffs(
    data: InputData,
    import_tariff: float | None = None,
    export_tariff: float | None = None,
) -> InputData:
    """Return a scenario with specified import/export tariffs, in DKK/kWh.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    import_tariff : float or None, optional
        Import tariff in DKK/kWh; None preserves the original.
    export_tariff : float or None, optional
        Export tariff in DKK/kWh; None preserves the original.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    return replace(
        data,
        import_tariff=data.import_tariff if import_tariff is None else import_tariff,
        export_tariff=data.export_tariff if export_tariff is None else export_tariff,
    )


def scale_pv(data: InputData, factor: float) -> InputData:
    """Return a scenario with PV availability multiplied by ``factor``.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    factor : float
        Non-negative scaling factor.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.

    Raises
    ------
    ValueError
        If scaling is invalid or exceeds installed peak availability.
    """
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("PV scale must be finite and non-negative.")
    profile = data.pv_profile * factor
    if (profile > 1).any():
        raise ValueError(
            "PV scaling exceeds installed peak power; use a smaller factor or change capacity explicitly."
        )
    return replace(data, pv_profile=profile, pv_available=data.pv_max_kW * profile)


def set_load_preferences(
    data: InputData,
    load_max_kWh: float | None = None,
    load_min_kWh: float | None = None,
    min_daily_energy_kWh: float | None = None,
) -> InputData:
    """Return a scenario with modified hourly bounds or minimum daily energy.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    load_max_kWh : float or None, optional
        Maximum consumption per one-hour period; None preserves it.
    load_min_kWh : float or None, optional
        Minimum consumption per one-hour period; None preserves it.
    min_daily_energy_kWh : float or None, optional
        Daily minimum consumption, in kWh; None preserves it.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    return replace(
        data,
        load_max_kWh=data.load_max_kWh if load_max_kWh is None else load_max_kWh,
        load_min_kWh=data.load_min_kWh if load_min_kWh is None else load_min_kWh,
        min_daily_energy_kWh=(
            data.min_daily_energy_kWh
            if min_daily_energy_kWh is None
            else min_daily_energy_kWh
        ),
    )


def set_linear_disutility(data, value):
    """Return a scenario with linear coefficient ``value``, in DKK/kWh.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    value : float
        Linear coefficient, in DKK/kWh.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    return replace(data, linear_disutility=value)


def set_quadratic_disutility(data, value):
    """Return a scenario with quadratic coefficient ``value``, in DKK/kWh².

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    value : float
        Quadratic coefficient, in DKK/kWh².

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    return replace(data, quadratic_disutility=value)


def set_battery_capacity(data: InputData, capacity_kWh: float) -> InputData:
    """Return a scenario with new capacity, retaining the initial SoC fraction.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    capacity_kWh : float
        New battery energy capacity, in kWh.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """
    if data.battery_capacity_kWh is None:
        raise ValueError("Capacity scenarios require battery parameters.")
    # Hold the initial fill ratio fixed so capacity is the only changed setting.
    fraction = (
        data.battery_initial_soc_kWh / data.battery_capacity_kWh
        if data.battery_capacity_kWh > 0
        else 0.0
    )
    return replace(
        data,
        battery_capacity_kWh=capacity_kWh,
        battery_initial_soc_kWh=fraction * capacity_kWh,
    )


def set_q2_price_structure(
    data: InputData,
    prices,
    reference_kWh: float,
) -> InputData:
    """Return a Q2 scenario with specified prices, constant reference and no PV.

    Parameters
    ----------
    data : InputData
        Physical inputs and consumer preferences.
    prices : array-like
        One finite energy price per hour, in DKK/kWh.
    reference_kWh : float
        Constant preferred consumption per one-hour period, in kWh.

    Returns
    -------
    InputData
        Validated scenario copy; the original inputs are retained.
    """

    prices = np.asarray(prices, dtype=float)
    if prices.shape != (data.n_hours,) or not np.isfinite(prices).all():
        raise ValueError("Supply one finite price per hour.")
    if not np.isfinite(reference_kWh) or not (
        data.load_min_kWh <= reference_kWh <= data.load_max_kWh
    ):
        raise ValueError("Constant reference must lie within the load bounds.")

    no_pv = scale_pv(data, factor=0.0)

    return replace(
        no_pv,
        energy_price=prices.copy(),
        reference_load=np.full(data.n_hours, reference_kWh),
    )
