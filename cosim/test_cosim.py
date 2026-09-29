"""
HELICS co-simulation test for inverter.py: OCHRE dwelling with an InverterBattery on the
IEEE 1547 Cat. B volt-VAr curve, connected to a minimal GridLAB-D feeder.

Runs the full co-simulation once (broker + GridLAB-D + OCHRE) and checks that:
  - all processes finish cleanly and the federates stay in lock-step
  - the voltage GridLAB-D computes is the voltage OCHRE's inverter sees
  - the battery Q follows the volt-VAr curve over the whole voltage sweep
  - the P + jQ that OCHRE publishes is the load GridLAB-D solves with

Requires gridlabd and helics_broker on PATH. Usage:
    pytest cosim/test_cosim.py
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

import run_cosim
from ochre_federate import V_BASE  # also puts ../src on sys.path
from inverter import IEEE1547_CAT_B_VOLT_VAR

pytestmark = pytest.mark.skipif(
    shutil.which("gridlabd") is None or shutil.which("helics_broker") is None,
    reason="gridlabd and helics_broker are required",
)


@pytest.fixture(scope="module")
def cosim():
    codes = run_cosim.run()
    out = run_cosim.OUTPUT
    ochre_df = pd.read_csv(os.path.join(out, "ochre_federate.csv")).set_index("Time (s)")
    ochre_df.index = ochre_df.index.astype(int)
    gld_v = run_cosim.read_gld_recorder(os.path.join(out, "gld_voltage.csv"))
    gld_s = run_cosim.read_gld_recorder(os.path.join(out, "gld_load.csv"))
    return codes, ochre_df, gld_v, gld_s


def test_all_processes_succeed(cosim):
    codes, *_ = cosim
    assert codes == {"broker": 0, "gridlabd": 0, "ochre": 0}


def test_federates_in_lock_step(cosim):
    _, df, gld_v, _ = cosim
    np.testing.assert_array_equal(df["Granted Time (s)"], df.index)
    assert df["Voltage Updated"].all()
    assert set(df.index) <= set(gld_v.index)


def test_ochre_sees_gridlabd_voltage(cosim):
    # OCHRE at time t runs with the voltage GridLAB-D solved at time t
    _, df, gld_v, _ = cosim
    expected = np.abs(gld_v.loc[df.index].values) / V_BASE
    np.testing.assert_allclose(df["Voltage (-)"], expected, rtol=1e-5)


def test_voltage_sweeps_full_curve(cosim):
    _, df, _, _ = cosim
    voltages, _ = IEEE1547_CAT_B_VOLT_VAR
    assert df["Voltage (-)"].min() < voltages[0]
    assert df["Voltage (-)"].max() > voltages[-1]


def test_battery_follows_volt_var_curve(cosim):
    _, df, _, _ = cosim
    s = df["Inverter Capacity (kVA)"]
    expected = np.interp(df["Voltage (-)"], *IEEE1547_CAT_B_VOLT_VAR) * s
    q = df["Battery Reactive Power (kVAR)"]
    np.testing.assert_allclose(q, expected, atol=1e-9)

    # both saturation limits and the deadband are reached
    q_max = IEEE1547_CAT_B_VOLT_VAR[1][-1] * s
    assert np.isclose(q, -q_max).any()
    assert np.isclose(q, q_max).any()
    assert (q == 0).any()


def test_battery_within_kva_rating(cosim):
    _, df, _, _ = cosim
    s = np.hypot(df["Battery Electric Power (kW)"], df["Battery Reactive Power (kVAR)"])
    assert (s <= df["Inverter Capacity (kVA)"] + 1e-9).all()
    # volt-VAr is exercised while charging and discharging
    assert (df["Battery Electric Power (kW)"] > 0).any()
    assert (df["Battery Electric Power (kW)"] < 0).any()


def test_gridlabd_load_matches_ochre_power(cosim):
    # GridLAB-D at time t solves with the P + jQ that OCHRE published at t - 60 s
    _, df, _, gld_s = cosim
    sent = (df["Total Electric Power (kW)"] + 1j * df["Total Reactive Power (kVAR)"]) * 1000
    received = gld_s.loc[df.index[1:]].values
    np.testing.assert_allclose(received.real, sent.values[:-1].real, rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(received.imag, sent.values[:-1].imag, rtol=1e-5, atol=1e-2)
    assert gld_s.loc[0] == 0  # nothing received before the first OCHRE step
