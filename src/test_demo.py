import numpy as np
import pytest

import demo
from inverter import IEEE1547_CAT_B_VOLT_VAR


@pytest.fixture(scope="module")
def volt_var():
    return demo.run_volt_var_scenario()


@pytest.fixture(scope="module")
def control_modes():
    return demo.run_control_mode_scenario()


@pytest.fixture(scope="module")
def priority():
    return demo.run_priority_scenario()


# Scenario 1: volt-VAr in a dwelling


def test_volt_var_q_matches_curve(volt_var):
    battery, df = volt_var
    expected = np.interp(df["Voltage (-)"], *IEEE1547_CAT_B_VOLT_VAR) * battery.inverter_capacity
    np.testing.assert_allclose(df["Q (kVAR)"], expected, atol=1e-9)


def test_volt_var_within_kva_and_soc_limits(volt_var):
    battery, df = volt_var
    assert (np.hypot(df["P (kW)"], df["Q (kVAR)"]) <= battery.inverter_capacity + 1e-9).all()
    assert df["SOC (-)"].between(battery.soc_min - 1e-3, battery.soc_max + 1e-3).all()


def test_volt_var_curtails_discharge_on_kva_circle(volt_var):
    battery, df = volt_var
    curtailed = (
        (df["P Request (kW)"] < 0)
        & (df["P (kW)"] > df["P Request (kW)"] + 0.1)
        & (df["SOC (-)"] > battery.soc_min + 0.01)
    )
    assert curtailed.any()
    s = np.hypot(df.loc[curtailed, "P (kW)"], df.loc[curtailed, "Q (kVAR)"])
    np.testing.assert_allclose(s, battery.inverter_capacity)


def test_volt_var_stops_at_soc_limits(volt_var):
    battery, df = volt_var
    stopped_full = (df["P Request (kW)"] > 0) & (df["P (kW)"] == 0)
    stopped_empty = (df["P Request (kW)"] < 0) & (df["P (kW)"] == 0)
    assert stopped_full.any() and stopped_empty.any()
    assert df.loc[stopped_full, "SOC (-)"].min() == pytest.approx(battery.soc_max, abs=1e-3)
    assert df.loc[stopped_empty, "SOC (-)"].max() == pytest.approx(battery.soc_min, abs=1e-3)


# Scenario 2: control modes in a dwelling


def test_p_setpoint_phase_charges_to_max_soc(control_modes):
    battery, df = control_modes
    phase = df[df["Mode"] == "P Setpoint"]
    below_max = phase["SOC (-)"] < battery.soc_max - 0.05
    np.testing.assert_allclose(phase.loc[below_max, "P (kW)"], 2)
    assert phase["SOC (-)"].iloc[-1] == pytest.approx(battery.soc_max, abs=1e-3)


def test_soc_target_phase_reaches_target(control_modes):
    _, df = control_modes
    phase = df[df["Mode"] == "SOC target"]
    assert phase["SOC (-)"].iloc[-1] == pytest.approx(demo.CM_SOC_TARGET, abs=1e-3)


def test_self_consumption_zeroes_grid_until_min_soc(control_modes):
    _, df = control_modes
    phase = df[df["Mode"] == "Self-consumption"]
    above_min = phase["SOC (-)"] > demo.CM_MIN_SOC + 0.02
    assert above_min.any()
    np.testing.assert_allclose(phase.loc[above_min, "Grid (kW)"], 0, atol=1e-6)

    assert phase["SOC (-)"].min() >= demo.CM_MIN_SOC - 1e-3
    at_min = phase["SOC (-)"] <= demo.CM_MIN_SOC + 1e-3
    assert at_min.any()
    # the step that reaches Min SOC may be partial; after that the battery is idle
    idle = at_min & at_min.shift(fill_value=False)
    # (tolerance of 1 W: the SOC controller can leave a tiny floating-point trickle)
    np.testing.assert_allclose(phase.loc[idle, "P (kW)"], 0, atol=1e-3)


# Scenario 3: inverter priority


def test_priority_modes(priority):
    kva, df = priority
    q_req = df["Q Request (kVAR)"]
    p_req = df["P Request (kW)"]

    watt = df["Priority"] == "Watt"
    np.testing.assert_allclose(df.loc[watt, "P (kW)"], p_req[watt])
    np.testing.assert_allclose(
        df.loc[watt, "Q (kVAR)"], np.minimum(q_req[watt], np.sqrt(kva**2 - p_req[watt] ** 2))
    )

    var = df["Priority"] == "Var"
    q_var = np.minimum(q_req[var], kva)
    np.testing.assert_allclose(df.loc[var, "Q (kVAR)"], q_var)
    np.testing.assert_allclose(
        df.loc[var, "P (kW)"], np.minimum(p_req[var], np.sqrt(kva**2 - q_var**2)), atol=1e-9
    )

    cpf = df["Priority"] == "CPF"
    np.testing.assert_allclose(
        df.loc[cpf, "Q (kVAR)"] * p_req[cpf], df.loc[cpf, "P (kW)"] * q_req[cpf], atol=1e-9
    )

    assert (np.hypot(df["P (kW)"], df["Q (kVAR)"]) <= kva + 1e-9).all()


# Figures


def test_make_figures(tmp_path):
    paths = demo.make_figures(tmp_path)
    assert len(paths) == 4
    for path in paths:
        assert path.endswith(".png")
        assert (tmp_path / path.split("/")[-1]).stat().st_size > 10_000
