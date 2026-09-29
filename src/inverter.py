"""
Generic four-quadrant inverter model for OCHRE equipment.

OCHRE only models reactive power for PV, and that logic lives inside
``PV.calculate_power_and_heat`` and assumes generation only (P <= 0). This module
provides a four-quadrant version as a mixin that can wrap any OCHRE device with an
AC real-power setpoint (Battery, ElectricVehicle, ...).

Sign convention (same as OCHRE):
  - P > 0: consuming / charging, P < 0: generating / discharging
  - Q > 0: absorbing (inductive),  Q < 0: injecting (capacitive)

With this convention, an IEEE 1547 Category B volt-VAr curve is
``([0.92, 0.98, 1.02, 1.08], [-0.44, 0, 0, 0.44])`` (Q in p.u. of kVA rating).

To use in a Dwelling, override the equipment class, e.g.
``Equipment={"Battery": {"equipment_class": InverterBattery, "volt_var_curve": ...}}``,
and pass the grid voltage each step with ``dwelling.update(control, {"Voltage (-)": v})``.
"""

import numpy as np

from ochre.utils import OCHREException
from ochre.Equipment import Battery, ElectricVehicle

IEEE1547_CAT_B_VOLT_VAR = ([0.92, 0.98, 1.02, 1.08], [-0.44, 0.0, 0.0, 0.44])
INVERTER_PRIORITIES = ["Watt", "Var", "CPF"]


class GridInverter:
    """
    Mixin that adds a kVA/power-factor limited inverter to an OCHRE device.

    Must be listed before the OCHRE equipment class, e.g.
    ``class InverterBattery(GridInverter, Battery)``, and subclasses must set
    ``setpoint_attr`` and implement ``default_kva``.

    Reactive power control, in order of precedence:
      1. Volt-VAr curve (if set): Q from the latest grid voltage
      2. Power factor (if set): Q proportional to |P|, sign of Q = sign of pf
      3. Q setpoint (default 0), in kVAR

    External control signal options (in addition to the device's own):
      - "Q Setpoint": reactive power setpoint, in kVAR. Clears the power factor setpoint
      - "Power Factor": fixed power factor, sign gives the sign of Q
      - "Volt-VAr Curve": (voltages in p.u., Q in p.u. of kVA), or None to disable
      - "Priority": "Watt", "Var", or "CPF" (constant power factor)
    """

    setpoint_attr = None  # name of the device's AC real power setpoint attribute

    def __init__(
        self,
        inverter_capacity=None,
        inverter_priority="Var",
        inverter_min_pf=None,
        volt_var_curve=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inverter_capacity = (
            inverter_capacity if inverter_capacity is not None else self.default_kva()
        )  # in kVA
        self.inverter_priority = self._check_priority(inverter_priority)
        self.inverter_min_pf = inverter_min_pf
        self.volt_var_curve = volt_var_curve
        self.q_setpoint = 0  # in kVAR
        self.pf_setpoint = None
        self.grid_voltage = 1  # latest grid voltage, in p.u.

    def default_kva(self):
        raise NotImplementedError

    def inverter_available(self):
        # True if the inverter is connected and can provide reactive power
        return True

    def _check_priority(self, priority):
        if priority not in INVERTER_PRIORITIES:
            raise OCHREException(f"Invalid {self.name} inverter priority: {priority}")
        return priority

    def update_inputs(self, schedule_inputs=None):
        super().update_inputs(schedule_inputs)

        # OCHRE drops schedule inputs that the equipment doesn't declare, including voltage.
        # Keep the latest voltage so that co-simulations can send it only when it changes.
        if schedule_inputs and "Voltage (-)" in schedule_inputs:
            self.grid_voltage = schedule_inputs["Voltage (-)"]
        self.current_schedule.setdefault("Voltage (-)", self.grid_voltage)

    def update_external_control(self, control_signal):
        if "Q Setpoint" in control_signal:
            self.q_setpoint = control_signal["Q Setpoint"]
            self.pf_setpoint = None
        if "Power Factor" in control_signal:
            pf = control_signal["Power Factor"]
            if pf is not None and not 0 < abs(pf) <= 1:
                raise OCHREException(f"Invalid {self.name} power factor: {pf}")
            self.pf_setpoint = pf
        if "Volt-VAr Curve" in control_signal:
            self.volt_var_curve = control_signal["Volt-VAr Curve"]
        if "Priority" in control_signal:
            self.inverter_priority = self._check_priority(control_signal["Priority"])

        return super().update_external_control(control_signal)

    def get_desired_q(self, p):
        if self.volt_var_curve is not None:
            v = self.current_schedule.get("Voltage (-)", self.grid_voltage)
            voltages, q_pu = self.volt_var_curve
            return float(np.interp(v, voltages, q_pu)) * self.inverter_capacity
        if self.pf_setpoint is not None:
            pf = self.pf_setpoint
            return abs(p) * np.sqrt(1 / pf**2 - 1) * np.sign(pf)
        return self.q_setpoint

    def calculate_power_and_heat(self):
        s = self.inverter_capacity
        p_set = float(np.clip(getattr(self, self.setpoint_attr), -s, s))
        q = self.get_desired_q(p_set) if self.inverter_available() else 0

        # Limit P before the device calculates losses and SOC, so its state stays consistent
        if self.inverter_priority == "Var":
            q = float(np.clip(q, -s, s))
            p_max = np.sqrt(s**2 - q**2)
            p_set = float(np.clip(p_set, -p_max, p_max))
        elif self.inverter_priority == "CPF" and np.hypot(p_set, q) > s:
            kva_ratio = np.hypot(p_set, q) / s
            p_set /= kva_ratio
            q /= kva_ratio
        setattr(self, self.setpoint_attr, p_set)

        heat_data = super().calculate_power_and_heat()
        p = self.electric_kw  # final P after device limits (SOC, charger, etc.)

        if self.inverter_priority == "Watt":
            q_max = np.sqrt(max(s**2 - p**2, 0))
            q = float(np.clip(q, -q_max, q_max))
        if self.inverter_min_pf is not None:
            q_max_pf = abs(p) * np.tan(np.arccos(self.inverter_min_pf))
            q = float(np.clip(q, -q_max_pf, q_max_pf))

        self.reactive_kvar = q if self.inverter_available() else 0
        return heat_data

    def run_zip(self, v, v0=1):
        # The inverter sets P and Q directly; only a grid disconnect zeroes them.
        # Note: the default ZIP model sets Q to 0 whenever P is 0.
        if v == 0:
            self.electric_kw = 0
            self.reactive_kvar = 0

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 3:
            results[f"{self.results_name} Reactive Power (kVAR)"] = self.reactive_kvar
        if self.verbosity >= 6:
            results[f"{self.results_name} Q Setpoint (kVAR)"] = self.q_setpoint
            results[f"{self.results_name} Inverter Voltage (-)"] = self.current_schedule.get(
                "Voltage (-)", self.grid_voltage
            )
        return results


class InverterBattery(GridInverter, Battery):
    """Stationary battery with a four-quadrant grid inverter."""

    setpoint_attr = "power_setpoint"

    def default_kva(self):
        return self.capacity  # kW rating of the battery


class InverterEV(GridInverter, ElectricVehicle):
    """
    EV charger with reactive power capability. Reactive power is only available while the
    EV is plugged in. Real power is limited to charging (P >= 0) by the OCHRE EV model.
    """

    setpoint_attr = "p_setpoint"

    def default_kva(self):
        return self.max_power  # charger rating

    def inverter_available(self):
        return self.in_event
