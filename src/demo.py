"""
Demonstration scenarios and figures for the OCHRE inverter battery.

Scenarios:
  1. Volt-VAr in a dwelling: battery follows a time-of-day P schedule while its inverter
     runs the IEEE 1547 Cat. B volt-VAr curve against a synthetic feeder voltage.
  2. Control modes in a dwelling: P setpoint, SOC target, then self-consumption with a
     minimum SOC, all sent through the external control signal.
  3. Inverter priority: standalone battery, P = 4 kW with an increasing Q request, for
     Watt, Var, and CPF priority.

Usage:
    python -m ochre_battery_model.demo [output_dir]
"""

import datetime as dt
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ochre  # noqa: E402
from ochre import Dwelling  # noqa: E402
from inverter import (  # noqa: E402
    IEEE1547_CAT_B_VOLT_VAR,
    INVERTER_PRIORITIES,
    InverterBattery,
)

DEFAULTS = os.path.join(os.path.dirname(ochre.__file__), "defaults")
INPUTS = os.path.join(DEFAULTS, "Input Files")
START = dt.datetime(2018, 1, 1)
TIME_RES = dt.timedelta(minutes=15)

# Scenario 1 settings
VV_INVERTER_KVA = 4  # smaller than the 5 kW battery, so Var priority curtails P
VV_CHARGE = (9, 15, 2.0)  # start hour, end hour, kW
VV_DISCHARGE = (18, 21, -4.0)

# Scenario 2 settings
CM_CHARGE_END = 6  # hour: P Setpoint +2 kW before this
CM_SOC_END = 10  # hour: SOC target before this
CM_SOC_TARGET = 0.6
CM_MIN_SOC = 0.3  # self-consumption minimum SOC

# Scenario 3 settings
PR_P_REQUEST = 4
PR_Q_REQUESTS = np.linspace(0, 6, 25)


def make_dwelling(duration=dt.timedelta(days=1), battery_args=None, **kwargs):
    """OCHRE example house (bldg0112631) in Denver with an InverterBattery."""
    return Dwelling(
        name="demo_house",
        start_time=START,
        time_res=TIME_RES,
        duration=duration,
        hpxml_file=os.path.join(INPUTS, "bldg0112631-up00.xml"),
        hpxml_schedule_file=os.path.join(INPUTS, "bldg0112631_schedule.csv"),
        weather_file=os.path.join(DEFAULTS, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
        save_results=False,
        verbosity=3,
        seed=1,
        Equipment={"Battery": {"equipment_class": InverterBattery, **(battery_args or {})}},
        **kwargs,
    )


def feeder_voltage(hour, rng):
    """Synthetic feeder voltage: midday rise (e.g. PV), evening dip (peak load)."""
    rise = 0.08 * np.exp(-(((hour - 13) / 2.5) ** 2))
    dip = 0.08 * np.exp(-(((hour - 19) / 1.8) ** 2))
    return 1.0 + rise - dip + rng.normal(0, 0.003)


def scheduled_power(hour):
    for start, end, power in (VV_CHARGE, VV_DISCHARGE):
        if start <= hour < end:
            return power
    return 0.0


def hour_of(t):
    return t.hour + t.minute / 60


def run_volt_var_scenario():
    """Scenario 1: returns (battery, DataFrame of per-step results)."""
    dwelling = make_dwelling(
        battery_args={
            "volt_var_curve": IEEE1547_CAT_B_VOLT_VAR,
            "inverter_capacity": VV_INVERTER_KVA,
            "inverter_priority": "Var",
            "enable_degradation": False,
        }
    )
    battery = dwelling.equipment["Battery"]
    rng = np.random.default_rng(0)
    rows = []
    for t in dwelling.sim_times:
        v = feeder_voltage(hour_of(t), rng)
        p_request = scheduled_power(hour_of(t))
        dwelling.update({"Battery": {"P Setpoint": p_request}}, {"Voltage (-)": v})
        rows.append(
            {
                "Time": t,
                "Voltage (-)": v,
                "P Request (kW)": p_request,
                "P (kW)": battery.electric_kw,
                "Q (kVAR)": battery.reactive_kvar,
                "SOC (-)": battery.soc,
            }
        )
    return battery, pd.DataFrame(rows).set_index("Time")


def control_mode_signal(t):
    hour = hour_of(t)
    if hour < CM_CHARGE_END:
        return "P Setpoint", {"P Setpoint": 2}
    if hour < CM_SOC_END:
        return "SOC target", {"SOC": CM_SOC_TARGET}
    return "Self-consumption", {"Self Consumption Mode": True, "Min SOC": CM_MIN_SOC}


def run_control_mode_scenario():
    """Scenario 2: returns (battery, DataFrame of per-step results)."""
    dwelling = make_dwelling(battery_args={"enable_degradation": False})
    battery = dwelling.equipment["Battery"]
    rows = []
    for t in dwelling.sim_times:
        mode, signal = control_mode_signal(t)
        dwelling.update({"Battery": signal}, {"Voltage (-)": 1.0})
        rows.append(
            {
                "Time": t,
                "Mode": mode,
                "P (kW)": battery.electric_kw,
                "SOC (-)": battery.soc,
                "Grid (kW)": dwelling.total_p_kw,
                "House Load (kW)": dwelling.total_p_kw - battery.electric_kw,
            }
        )
    return battery, pd.DataFrame(rows).set_index("Time")


def run_priority_scenario():
    """Scenario 3: returns (kVA rating, DataFrame of requested and delivered P/Q)."""
    rows = []
    for priority in INVERTER_PRIORITIES:
        for q_request in PR_Q_REQUESTS:
            battery = InverterBattery(
                enable_degradation=False,
                inverter_priority=priority,
                start_time=START,
                time_res=TIME_RES,
                duration=dt.timedelta(hours=1),
                save_results=False,
                verbosity=0,
            )
            battery.update({"P Setpoint": PR_P_REQUEST, "Q Setpoint": q_request})
            rows.append(
                {
                    "Priority": priority,
                    "P Request (kW)": PR_P_REQUEST,
                    "Q Request (kVAR)": q_request,
                    "P (kW)": battery.electric_kw,
                    "Q (kVAR)": battery.reactive_kvar,
                }
            )
    return battery.inverter_capacity, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
BAND = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (fixed order)


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": TEXT_2,
            "lines.linewidth": 1.6,
            "font.size": 9,
        }
    )


def format_time_axis(ax):
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time of day")


def plot_volt_var_curve(battery, df):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    v_curve = np.linspace(0.9, 1.1, 401)
    q_curve = np.interp(v_curve, *IEEE1547_CAT_B_VOLT_VAR)
    ax.axvspan(0.98, 1.02, color=BAND, zorder=0)
    ax.plot(v_curve, q_curve, color=TEXT_2, lw=1.2, ls="--", label="IEEE 1547 Cat. B curve")
    ax.scatter(
        df["Voltage (-)"],
        df["Q (kVAR)"] / battery.inverter_capacity,
        s=22,
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=0.8,
        zorder=3,
        label="Simulated battery (15-min steps)",
    )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.text(1.0, 0.3, "deadband", ha="center", color=TEXT_2, fontsize=8)
    ax.text(0.902, -0.25, "injects Q", color=TEXT_2, fontsize=8)
    ax.text(1.098, 0.25, "absorbs Q", ha="right", color=TEXT_2, fontsize=8)
    ax.set_xlim(0.9, 1.1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("Grid voltage (p.u.)")
    ax.set_ylabel("Reactive power (p.u. of inverter kVA)")
    ax.set_title("Battery reactive power follows the volt-VAr curve in the dwelling")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_volt_var_timeseries(battery, df):
    fig, axes = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)
    t = df.index

    ax = axes[0]
    ax.axhspan(0.98, 1.02, color=BAND, zorder=0)
    ax.plot(t, df["Voltage (-)"], color=SERIES[0])
    ax.set_ylabel("p.u.")
    ax.set_title("PCC voltage (a fake number, but it will be an input from co-simulation)")
    
    ax.text(t[1], 1.022, "volt-VAr deadband", color=TEXT_2, fontsize=8, va="bottom")

    ax = axes[1]
    ax.step(t, df["P Request (kW)"], where="post", color=MUTED, ls="--", lw=1.2, label="Requested")
    ax.step(t, df["P (kW)"], where="post", color=SERIES[0], label="Delivered")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("kW")
    ax.set_title("Battery real power (+ charge, − discharge)")
    ax.legend(loc="upper left", ncol=2)
    # annotate where delivered P departs from the request, found from the results
    request, delivered = df["P Request (kW)"], df["P (kW)"]
    stopped_full = (request > 0) & (delivered.abs() < 0.01)
    stopped_empty = (request < 0) & (delivered.abs() < 0.01)
    soc_limited = df["SOC (-)"] <= battery.soc_min + 0.01
    curtail = (delivered - request).where((request < 0) & (delivered < -0.01) & ~soc_limited, 0)
    notes = []
    if stopped_full.any():
        notes.append(("stopped at max SOC", stopped_full.idxmax(), dt.timedelta(hours=1.5), 1.2))
    # if curtail.max() > 0.05:
    #     notes.append(
    #         ("curtailed by kVA limit\n(Var priority)", curtail.idxmax(), -dt.timedelta(hours=6), -2.8)
    #     )
    if stopped_empty.any():
        notes.append(("stopped at min SOC", stopped_empty.idxmax(), dt.timedelta(hours=1), -1.6))
    for text, t_n, offset, y_text in notes:
        ax.annotate(
            text,
            xy=(t_n, delivered[t_n]),
            xytext=(t_n + offset, y_text),
            color=TEXT_2,
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
        )

    ax = axes[2]
    ax.step(t, df["Q (kVAR)"], where="post", color=SERIES[0])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("kVAR")
    ax.set_title("Battery reactive power (+ absorb, − inject)")

    ax = axes[3]
    ax.plot(t, df["SOC (-)"], color=SERIES[0])
    for limit, label in ((battery.soc_max, "max SOC"), (battery.soc_min, "min SOC")):
        ax.axhline(limit, color=MUTED, ls=":", lw=1)
        ax.text(t[-1], limit, label, color=TEXT_2, fontsize=8, ha="right", va="bottom")
    ax.set_ylim(0, 1)
    ax.set_ylabel("SOC (-)")
    ax.set_title("State of charge")
    format_time_axis(ax)

    fig.suptitle(
        f"volt-VAr with a {battery.capacity:g} kW battery on a "
        f"{battery.inverter_capacity:g} kVA inverter",
        x=0.02,
        ha="left",
        fontweight="bold",
        color=TEXT,
    )
    fig.tight_layout()
    return fig


def plot_control_modes(battery, df):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    t = df.index

    # shade and label control phases
    phases = df["Mode"].ne(df["Mode"].shift()).cumsum()
    for i, (_, phase) in enumerate(df.groupby(phases)):
        start = phase.index[0]
        end = phase.index[-1] + TIME_RES
        for ax in axes:
            if i % 2 == 0:
                ax.axvspan(start, end, color=BAND, zorder=0, lw=0)
        axes[0].text(
            start + (end - start) / 2,
            1.02,
            phase["Mode"].iloc[0],
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=TEXT_2,
            fontsize=8,
        )

    ax = axes[0]
    ax.step(t, df["House Load (kW)"], where="post", color=SERIES[1], label="House load without battery")
    ax.step(t, df["Grid (kW)"], where="post", color=SERIES[0], label="Grid import with battery")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("kW")
    ax.set_title("Dwelling power", pad=16)
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.step(t, df["P (kW)"], where="post", color=SERIES[0])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("kW")
    ax.set_title("Battery real power (+ charge, − discharge)")

    ax = axes[2]
    ax.plot(t, df["SOC (-)"], color=SERIES[0])
    for limit, label in (
        (battery.soc_max, "max SOC (hard limit)"),
        (CM_SOC_TARGET, "SOC target"),
        (CM_MIN_SOC, "Min SOC (self-consumption)"),
        (battery.soc_min, "min SOC (hard limit)"),
    ):
        ax.axhline(limit, color=MUTED, ls=":", lw=1)
        ax.text(t[-1], limit, label, color=TEXT_2, fontsize=8, ha="right", va="bottom")
    ax.set_ylim(0, 1)
    ax.set_ylabel("SOC (-)")
    ax.set_title("State of charge")
    format_time_axis(ax)

    fig.suptitle(
        "Scenario 2 · battery control modes via the external control signal",
        x=0.02,
        ha="left",
        fontweight="bold",
        color=TEXT,
    )
    fig.tight_layout()
    return fig


def plot_priority(kva, df):
    subtitles = {
        "Watt": "Watt priority: keep P, cut Q",
        "Var": "Var priority: keep Q, cut P",
        "CPF": "CPF priority: scale P and Q together",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    theta = np.linspace(0, np.pi / 2, 200)
    for ax, priority in zip(axes, INVERTER_PRIORITIES):
        d = df[df["Priority"] == priority]
        ax.plot(kva * np.cos(theta), kva * np.sin(theta), color=TEXT_2, ls="--", lw=1.2)
        ax.text(0.15, kva + 0.12, f"{kva:g} kVA limit", color=TEXT_2, fontsize=8)
        # link each request outside the kVA circle to what was delivered
        outside = np.hypot(d["P Request (kW)"], d["Q Request (kVAR)"]) > kva
        for _, row in d[outside].iterrows():
            ax.plot(
                [row["P Request (kW)"], row["P (kW)"]],
                [row["Q Request (kVAR)"], row["Q (kVAR)"]],
                color=GRID,
                lw=0.8,
                zorder=1,
            )
        ax.scatter(
            d["P Request (kW)"],
            d["Q Request (kVAR)"],
            s=18,
            facecolor=SURFACE,
            edgecolor=MUTED,
            zorder=2,
            label="Requested",
        )
        ax.scatter(
            d["P (kW)"],
            d["Q (kVAR)"],
            s=22,
            color=SERIES[0],
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
            label="Delivered",
        )
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 6.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Real power P (kW)")
        ax.set_title(subtitles[priority], fontsize=10)
    axes[0].set_ylabel("Reactive power Q (kVAR)")
    axes[0].legend(loc="upper right")
    fig.suptitle(
        f"Scenario 3 · P = {PR_P_REQUEST} kW with Q requests of 0–6 kVAR on a {kva:g} kVA inverter",
        x=0.02,
        ha="left",
        fontweight="bold",
        color=TEXT,
    )
    fig.tight_layout()
    return fig


def make_figures(output_dir):
    """Run all scenarios and save figures. Returns the list of saved file paths."""
    set_style()
    os.makedirs(output_dir, exist_ok=True)
    battery_vv, df_vv = run_volt_var_scenario()
    battery_cm, df_cm = run_control_mode_scenario()
    kva, df_pr = run_priority_scenario()

    figures = {
        "1_volt_var_curve.png": plot_volt_var_curve(battery_vv, df_vv),
        "2_volt_var_timeseries.png": plot_volt_var_timeseries(battery_vv, df_vv),
        "3_control_modes.png": plot_control_modes(battery_cm, df_cm),
        "4_inverter_priority.png": plot_priority(kva, df_pr),
    }
    paths = []
    for name, fig in figures.items():
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "figures"
    for p in make_figures(out):
        print("Saved", p)
