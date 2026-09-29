import datetime as dt
import os

import numpy as np
import pandas as pd
import pytest

import ochre
from ochre import Dwelling
from inverter import (
    IEEE1547_CAT_B_VOLT_VAR,
    InverterBattery,
    InverterEV,
)

DEFAULTS = os.path.join(os.path.dirname(ochre.__file__), "defaults")
INPUTS = os.path.join(DEFAULTS, "Input Files")
N_STEPS = 192  # 2 days at 15 minutes


def make_dwelling():
    return Dwelling(
        name="test_house",
        start_time=dt.datetime(2018, 1, 1),
        time_res=dt.timedelta(minutes=15),
        duration=dt.timedelta(days=2),
        hpxml_file=os.path.join(INPUTS, "bldg0112631-up00.xml"),
        hpxml_schedule_file=os.path.join(INPUTS, "bldg0112631_schedule.csv"),
        weather_file=os.path.join(DEFAULTS, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
        save_results=False,
        verbosity=3,
        seed=1,  # fixes the random EV events
        Equipment={
            "Battery": {
                "equipment_class": InverterBattery,
                "volt_var_curve": IEEE1547_CAT_B_VOLT_VAR,
            },
            "EV": {
                "equipment_class": InverterEV,
                "vehicle_type": "BEV",
                "charging_level": "Level 2",
                "range": 250,
            },
        },
    )


def voltage_at(i):
    # one full cycle between 0.9 and 1.1 p.u. per day
    return 1.0 + 0.1 * np.sin(2 * np.pi * i / 96)


@pytest.fixture(scope="module")
def run():
    # Step the dwelling like a co-simulation: voltage in, control in, results out
    dwelling = make_dwelling()
    rows = []
    for i in range(N_STEPS):
        control = {
            "Battery": {"P Setpoint": 2 if (i // 16) % 2 else -2},
            "EV": {"Q Setpoint": -3},
        }
        results = dwelling.update(control, {"Voltage (-)": voltage_at(i)})
        results["Voltage In (-)"] = voltage_at(i)
        results["EV Parked"] = dwelling.equipment["EV"].in_event
        results["Equipment Q Sum (kVAR)"] = sum(
            e.reactive_kvar for e in dwelling.equipment.values()
        )
        rows.append(results)
    return dwelling, pd.DataFrame(rows).set_index("Time")


def test_dwelling_uses_inverter_classes(run):
    dwelling, _ = run
    assert isinstance(dwelling.equipment["Battery"], InverterBattery)
    assert isinstance(dwelling.equipment["EV"], InverterEV)


def test_battery_follows_volt_var_curve(run):
    dwelling, df = run
    s = dwelling.equipment["Battery"].inverter_capacity
    expected = np.interp(df["Voltage In (-)"], *IEEE1547_CAT_B_VOLT_VAR) * s

    np.testing.assert_allclose(df["Battery Reactive Power (kVAR)"], expected, atol=1e-9)
    # volt-VAr is exercised in both directions
    assert df["Battery Reactive Power (kVAR)"].min() < 0 < df["Battery Reactive Power (kVAR)"].max()


def test_battery_q_while_idle(run):
    _, df = run
    idle = df["Battery Electric Power (kW)"] == 0
    active_q = df["Battery Reactive Power (kVAR)"] != 0
    assert (idle & active_q).any()


def test_battery_stays_within_soc_and_kva_limits(run):
    dwelling, df = run
    b = dwelling.equipment["Battery"]
    soc = df["Battery SOC (-)"]
    assert soc.min() >= b.soc_min - 1e-3
    assert soc.max() <= b.soc_max + 1e-3
    s = np.hypot(df["Battery Electric Power (kW)"], df["Battery Reactive Power (kVAR)"])
    assert (s <= b.inverter_capacity + 1e-9).all()


def test_ev_q_only_while_parked(run):
    _, df = run
    parked = df["EV Parked"].astype(bool)
    assert parked.any() and (~parked).any()
    np.testing.assert_allclose(df.loc[parked, "EV Reactive Power (kVAR)"], -3)
    assert (df.loc[~parked, "EV Reactive Power (kVAR)"] == 0).all()


def test_total_reactive_power_includes_inverters(run):
    _, df = run
    np.testing.assert_allclose(
        df["Total Reactive Power (kVAR)"], df["Equipment Q Sum (kVAR)"], atol=1e-9
    )


def test_outage_turns_inverters_off_and_recovers():
    dwelling = make_dwelling()
    battery = dwelling.equipment["Battery"]

    # grid outage: house can't island while the battery charges, so OCHRE turns everything off
    dwelling.update({"Battery": {"P Setpoint": 2}}, {"Voltage (-)": 0})
    assert dwelling.total_p_kw == 0
    assert dwelling.total_q_kvar == 0
    assert battery.electric_kw == 0
    assert battery.reactive_kvar == 0

    # grid restored at high voltage: volt-VAr absorbs Q again
    dwelling.update({"Battery": {"P Setpoint": 0}}, {"Voltage (-)": 1.08})
    assert battery.reactive_kvar == pytest.approx(0.44 * battery.inverter_capacity)
