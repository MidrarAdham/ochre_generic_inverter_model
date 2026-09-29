"""
OCHRE HELICS federate: one Dwelling with an InverterBattery on the IEEE 1547 Cat. B
volt-VAr curve.

Each step (60 s):
  1. receive house_meter.voltage_12 from GridLAB-D and convert it to p.u.
  2. step the dwelling with that voltage and a battery P setpoint
  3. publish total house P + jQ (in VA) to ochre_house.constant_power_12

Everything sent and received is logged to output/ochre_federate.csv for test_cosim.py.

This is something I need to fix on the actual HELICS setup. Currently, all of the
actual HELICS setup use real power instead of the apparent power.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd  # must be imported before helics (native library conflict)

import ochre
from ochre import Dwelling

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from inverter import IEEE1547_CAT_B_VOLT_VAR, InverterBattery  # noqa: E402

import helics as h  # noqa: E402

DEFAULTS = os.path.join(os.path.dirname(ochre.__file__), "defaults")
INPUTS = os.path.join(DEFAULTS, "Input Files")
OUTPUT = os.path.join(HERE, "output")

START = dt.datetime(2018, 1, 1)
TIME_RES = dt.timedelta(minutes=1)  # must match "period" in the HELICS configs
DURATION = dt.timedelta(hours=4)  # must match the GridLAB-D clock
V_BASE = 240  # voltage_12 base, in V


def make_dwelling():
    return Dwelling(
        name="cosim_house",
        start_time=START,
        time_res=TIME_RES,
        duration=DURATION,
        hpxml_file=os.path.join(INPUTS, "bldg0112631-up00.xml"),
        hpxml_schedule_file=os.path.join(INPUTS, "bldg0112631_schedule.csv"),
        weather_file=os.path.join(DEFAULTS, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
        save_results=False,
        verbosity=3,
        seed=1,
        Equipment={
            "Battery": {
                "equipment_class": InverterBattery,
                "volt_var_curve": IEEE1547_CAT_B_VOLT_VAR,
            },
        },
    )


def battery_setpoint(t_s):
    # alternate +2 kW / -2 kW every hour, so volt-VAr is tested while charging and discharging
    return 2.0 if (t_s // 3600) % 2 == 0 else -2.0


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    dwelling = make_dwelling()  # build before connecting, so initialization doesn't stall the broker
    battery = dwelling.equipment["Battery"]

    fed = h.helicsCreateValueFederateFromConfig(os.path.join(HERE, "ochre_helics_config.json"))
    pub = fed.get_publication_by_name("ochre_house.constant_power_12")
    sub = fed.get_subscription_by_index(0)
    fed.enter_executing_mode()

    n_steps = int(DURATION / TIME_RES)
    v_pu = 1.0  # until GridLAB-D publishes; 0 would put OCHRE in outage mode
    rows = []
    for i in range(n_steps):
        t_req = i * TIME_RES.total_seconds()
        # the federate starts at t = 0 after entering executing mode
        t_granted = fed.request_time(t_req) if i > 0 else fed.current_time

        v_updated = sub.is_updated()
        if v_updated:
            v_pu = abs(sub.complex) / V_BASE

        control = {"Battery": {"P Setpoint": battery_setpoint(t_req)}}
        results = dwelling.update(control, {"Voltage (-)": v_pu})

        p_kw = results["Total Electric Power (kW)"]
        q_kvar = results["Total Reactive Power (kVAR)"]
        pub.publish(complex(p_kw * 1000, q_kvar * 1000))

        rows.append(
            {
                "Time (s)": t_req,
                "Granted Time (s)": t_granted,
                "Voltage Updated": v_updated,
                "Voltage (-)": v_pu,
                "Battery P Setpoint (kW)": battery_setpoint(t_req),
                "Battery Electric Power (kW)": battery.electric_kw,
                "Battery Reactive Power (kVAR)": battery.reactive_kvar,
                "Battery SOC (-)": battery.soc,
                "Inverter Capacity (kVA)": battery.inverter_capacity,
                "Total Electric Power (kW)": p_kw,
                "Total Reactive Power (kVAR)": q_kvar,
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT, "ochre_federate.csv"), index=False)
    dwelling.finalize()
    fed.disconnect()
    h.helicsFederateFree(fed)
    h.helicsCloseLibrary()
    print(f"OCHRE federate done: {n_steps} steps")


if __name__ == "__main__":
    main()
